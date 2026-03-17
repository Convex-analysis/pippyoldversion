"""
Separated stage trainer for EVO-1 two-stage training

This module provides functionality to execute Stage 1 and Stage 2 training
independently, allowing for flexible training with limited GPU resources.
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import wandb
from pathlib import Path

# Import EVO-1 components
from utils.config import EVO1DrivingConfig
from model.evo1_driving import FederatedEVO1Driving
from data.nuscenes_loader import create_dataloader
from data.augmentation import DrivingAugmentation, create_comprehensive_augmentation
from training.utils import TrainingMetrics, CheckpointManager, LearningRateScheduler


@dataclass
class StageTrainingState:
    """Training state for stage-specific training"""
    current_round: int = 0
    total_rounds: int = 50
    stage_name: str = "unknown"
    model_checkpoint_path: Optional[str] = None
    stage_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_metrics is None:
            self.stage_metrics = {}


class StageClientTrainer:
    """Individual client trainer for stage-specific training"""
    
    def __init__(
        self,
        client_id: str,
        config: EVO1DrivingConfig,
        device: str = "cuda",
        stage: int = 1
    ):
        self.client_id = client_id
        self.config = config
        self.device = device
        self.stage = stage
        
        # Setup model
        self.model = FederatedEVO1Driving(
            config=config.model,
            training_config=config.training,
            device=device
        ).to(device)
        
        self.model.set_client_id(client_id)
        
        # Set model stage mode
        if stage == 1:
            self.model.set_stage1_mode()
            self.stage_name = "Stage 1"
            learning_rate = config.training.stage1_lr
        else:
            self.model.set_stage2_mode()
            self.stage_name = "Stage 2"
            learning_rate = config.training.stage2_lr
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            config=config.training
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Setup augmentation
        self.augmentation = create_comprehensive_augmentation()
        
        # Setup data loader
        self.setup_data_loader()
        
        # Setup metrics tracking
        self.metrics = TrainingMetrics()
        
        # Checkpoint manager
        stage_output_dir = os.path.join(config.output_dir, f"stage{stage}_{client_id}")
        self.checkpoint_manager = CheckpointManager(
            output_dir=stage_output_dir,
            max_checkpoints=config.max_checkpoints_to_keep
        )
    
    def setup_data_loader(self):
        """Setup data loader for this client"""
        # Extract client number safely
        try:
            client_num = int(self.client_id.split('_')[1])
        except (IndexError, ValueError):
            client_num = 0  # Default to 0 if extraction fails
            logging.warning(f"Failed to extract client number from {self.client_id}, using default 0")
        
        self.train_loader = create_dataloader(
            config=self.config.data,
            model_config=self.config.model,
            split="train",
            client_id=client_num,
            num_clients=self.config.training.num_clients,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0  # Disable multiprocessing to avoid deadlocks
        )
        
        self.val_loader = create_dataloader(
            config=self.config.data,
            model_config=self.config.model,
            split="val",
            client_id=client_num,
            num_clients=self.config.training.num_clients,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def train_epoch(self, global_round: int) -> Dict[str, float]:
        """Train for one local epoch"""
        self.model.train()
        epoch_metrics = {}
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply augmentation
            if self.config.training.federated_learning:
                batch = self.augmentation.augment_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.training.mixed_precision:
                with autocast('cuda'):
                    output = self.model(
                        images=batch['images'],
                        image_mask=batch['image_mask'],
                        state=batch['state'],
                        instructions=batch['instructions'],
                        future_controls=batch['future_controls'],
                        mode="training"
                    )
                    
                    loss_dict = self.model.compute_loss(
                        output=output,
                        target_controls=batch['future_controls']
                    )
                    loss = loss_dict['total_loss']
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (only on trainable parameters)
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                output = self.model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    future_controls=batch['future_controls'],
                    mode="training"
                )
                
                loss_dict = self.model.compute_loss(
                    output=output,
                    target_controls=batch['future_controls']
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping (only on trainable parameters)
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                logging.info(
                    f"Client {self.client_id} - Round {global_round} - "
                    f"{self.stage_name} - Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
                
                # Log to wandb if enabled
                if wandb.run is not None:
                    wandb.log({
                        f"client_{self.client_id}/batch_loss": loss.item(),
                        f"client_{self.client_id}/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "global_round": global_round,
                        f"client_{self.client_id}/stage": self.stage
                    })
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Compute epoch metrics
        epoch_metrics['train_loss'] = total_loss / max(num_batches, 1)
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        epoch_metrics['stage'] = self.stage
        epoch_metrics['trainable_params_ratio'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / sum(p.numel() for p in self.model.parameters())
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set"""
        self.model.eval()
        val_metrics = {}
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output = self.model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    future_controls=batch['future_controls'],
                    mode="training"
                )
                
                loss_dict = self.model.compute_loss(
                    output=output,
                    target_controls=batch['future_controls']
                )
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        val_metrics['val_loss'] = total_loss / max(num_batches, 1)
        return val_metrics
    
    def train_local(self, num_epochs: int, global_round: int) -> Dict[str, float]:
        """Train locally for specified epochs"""
        all_metrics = {}
        
        for epoch in range(num_epochs):
            logging.info(f"Client {self.client_id} - Round {global_round} - {self.stage_name} - Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_metrics = self.train_epoch(global_round)
            
            # Validate (every other epoch to save time)
            if epoch % 2 == 0:
                val_metrics = self.validate()
                train_metrics.update(val_metrics)
            
            # Update metrics
            all_metrics.update({f"epoch_{epoch}_{k}": v for k, v in train_metrics.items()})
        
        # Save checkpoint
        if (global_round + 1) % self.config.save_frequency == 0:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=global_round,
                metrics=all_metrics
            )
            logging.info(f"Saved checkpoint for client {self.client_id}: {checkpoint_path}")
        
        return all_metrics
    
    def get_model_updates(self) -> Dict[str, torch.Tensor]:
        """Get local model updates for federated aggregation"""
        return self.model.get_local_updates()
    
    def set_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Set global model parameters"""
        self.model.set_global_model(global_state_dict)


class SeparatedStageTrainer:
    """Trainer for executing individual stages of two-stage training"""
    
    def __init__(
        self,
        config: EVO1DrivingConfig,
        stage: int,
        resume_from_checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.stage = stage
        self.device = device
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Initialize training state
        if stage == 1:
            total_rounds = config.training.stage1_rounds
            stage_name = "Stage 1"
        else:
            total_rounds = config.training.stage2_rounds
            stage_name = "Stage 2"
        
        self.training_state = StageTrainingState(
            current_round=0,
            total_rounds=total_rounds,
            stage_name=stage_name
        )
        
        # Setup global model
        self.global_model = FederatedEVO1Driving(
            config=config.model,
            training_config=config.training,
            device=device
        ).to(device)
        
        # Set model stage mode
        if stage == 1:
            self.global_model.set_stage1_mode()
        else:
            self.global_model.set_stage2_mode()
        
        # Initialize client trainers
        self.client_trainers = {}
        self.setup_client_trainers()
        
        # Setup output directories
        self.setup_output_dirs()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb if enabled
        self.setup_wandb()
        
        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
    
    def setup_client_trainers(self):
        """Setup client trainers for the specific stage"""
        num_clients = self.config.training.num_clients
        
        logging.info(f"Setting up {num_clients} client trainers for {self.training_state.stage_name}...")
        
        for client_id in range(num_clients):
            client_name = f"client_{client_id}"
            
            # Create trainer for this stage
            self.client_trainers[client_name] = StageClientTrainer(
                client_id=client_name,
                config=self.config,
                device=self.device,
                stage=self.stage
            )
        
        logging.info(f"Initialized {num_clients} client trainers for {self.training_state.stage_name}")
    
    def setup_output_dirs(self):
        """Setup output directories"""
        stage_output_dir = os.path.join(self.config.output_dir, f"stage_{self.stage}")
        os.makedirs(stage_output_dir, exist_ok=True)
        os.makedirs(os.path.join(stage_output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(stage_output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(stage_output_dir, "metrics"), exist_ok=True)
        
        self.stage_output_dir = stage_output_dir
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.stage_output_dir, "logs", f"stage_{self.stage}_training.log")
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers to avoid duplication
        if not logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Create stream handler for console output
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if os.getenv("WANDB_API_KEY"):
            wandb.init(
                project="evo1-stage-training",
                name=f"{self.config.experiment_name}_stage_{self.stage}",
                config={
                    **asdict(self.config),
                    "stage": self.stage,
                    "stage_name": self.training_state.stage_name
                }
            )
            logging.info("Initialized wandb logging")
        else:
            logging.info("Wandb not configured. Skipping wandb logging.")
    
    def select_clients(self) -> List[str]:
        """Select clients for current federated round"""
        num_clients = self.config.training.num_clients
        client_fraction = self.config.training.client_fraction
        num_selected = max(1, int(num_clients * client_fraction))
        
        # Simple random selection
        all_clients = list(self.client_trainers.keys())
        selected_clients = np.random.choice(
            all_clients, 
            size=num_selected, 
            replace=False
        ).tolist()
        
        return selected_clients
    
    def federated_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one federated learning round"""
        logging.info(f"Starting {self.training_state.stage_name} round {round_idx + 1}/{self.training_state.total_rounds}")
        
        round_start_time = time.time()
        round_metrics = {}
        
        # Select participating clients
        selected_clients = self.select_clients()
        
        logging.info(f"Selected clients for round {round_idx + 1}: {selected_clients}")
        
        # Distribute global model to selected clients
        global_state = self.global_model.state_dict()
        for client_id in selected_clients:
            self.client_trainers[client_id].set_global_model(global_state)
        
        # Local training on selected clients
        client_updates = {}
        client_metrics = {}
        
        for client_id in selected_clients:
            logging.info(f"Training client {client_id}")
            
            # Local training
            metrics = self.client_trainers[client_id].train_local(
                num_epochs=self.config.training.local_epochs,
                global_round=round_idx
            )
            
            # Get model updates
            updates = self.client_trainers[client_id].get_model_updates()
            
            # Apply differential privacy if enabled
            if self.client_trainers[client_id].model.noise_scale > 0:
                updates = self.client_trainers[client_id].model.apply_differential_privacy(updates)
            
            client_updates[client_id] = updates
            client_metrics[client_id] = metrics
        
        # Aggregate updates (FedAvg)
        aggregated_updates = self.aggregate_client_updates(client_updates)
        
        # Update global model
        self.global_model.update_from_aggregation(aggregated_updates)
        
        # Compute round metrics
        round_time = time.time() - round_start_time
        round_metrics['round_time'] = round_time
        round_metrics['num_participating_clients'] = len(selected_clients)
        round_metrics['stage'] = self.stage
        
        # Aggregate client metrics
        if client_metrics:
            avg_train_loss = np.mean([
                max([v for k, v in metrics.items() if 'train_loss' in k], default=0)
                for metrics in client_metrics.values()
            ])
            avg_val_loss = np.mean([
                max([v for k, v in metrics.items() if 'val_loss' in k], default=0)
                for metrics in client_metrics.values()
            ])
            
            round_metrics['avg_train_loss'] = avg_train_loss
            round_metrics['avg_val_loss'] = avg_val_loss
        
        # Update training state
        self.training_state.current_round = round_idx
        self.training_state.stage_metrics = round_metrics
        
        # Log metrics
        self.log_round_metrics(round_idx, round_metrics, client_metrics)
        
        # Save checkpoint
        if (round_idx + 1) % self.config.save_frequency == 0:
            self.save_stage_checkpoint(round_idx)
        
        return round_metrics
    
    def aggregate_client_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates from clients with stage awareness"""
        if not client_updates:
            return {}
        
        # Simple FedAvg - equal weights for all clients
        client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Use first client's update structure as template
        aggregated_updates = {}
        
        first_client = list(client_updates.keys())[0]
        for param_name in client_updates[first_client]:
            # Filter based on stage
            if self.stage == 1 and not self._is_stage1_trainable_param(param_name):
                continue
            
            # Weighted average of parameter updates
            param_updates = [client_updates[client][param_name] for client in client_updates]
            aggregated = sum(w * u for w, u in zip(client_weights, param_updates))
            aggregated_updates[param_name] = aggregated
        
        logging.info(f"[AGGREGATION] {self.training_state.stage_name} - Aggregated {len(aggregated_updates)} parameters")
        
        return aggregated_updates
    
    def _is_stage1_trainable_param(self, param_name: str) -> bool:
        """Check if parameter should be trainable in Stage 1"""
        if self.stage != 1:
            return True
        
        # In Stage 1, only train action expert and integration modules
        trainable_modules = [
            'action_head',
            'state_encoder',
            'control_head',
            'confidence_estimator'
        ]
        
        frozen_modules = [
            'vl_embedder'
        ]
        
        for trainable_module in trainable_modules:
            if param_name.startswith(trainable_module):
                return True
        
        for frozen_module in frozen_modules:
            if param_name.startswith(frozen_module):
                return False
        
        return True
    
    def log_round_metrics(self, round_idx: int, global_metrics: Dict[str, float], 
                         client_metrics: Dict[str, Dict[str, float]]):
        """Log metrics for current round"""
        # Log to console
        logging.info(f"{self.training_state.stage_name} Round {round_idx + 1} completed:")
        for metric, value in global_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "round": round_idx,
                **{f"global/{k}": v for k, v in global_metrics.items()}
            })
            
            # Log client metrics
            for client_id, metrics in client_metrics.items():
                wandb.log({
                    f"client_{client_id}/train_loss": metrics.get('train_loss', 0),
                    f"client_{client_id}/val_loss": metrics.get('val_loss', 0),
                    "round": round_idx
                })
        
        # Save metrics to file
        metrics_file = os.path.join(
            self.stage_output_dir, "metrics", 
            f"stage_{self.stage}_round_{round_idx:04d}_metrics.json"
        )
        
        metrics_data = {
            'round': round_idx,
            'stage': self.stage,
            'stage_name': self.training_state.stage_name,
            'global_metrics': global_metrics,
            'client_metrics': {k: {str(mk): float(mv) for mk, mv in v.items()} 
                              for k, v in client_metrics.items()},
            'timestamp': time.time()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def save_stage_checkpoint(self, round_idx: int):
        """Save stage-specific model checkpoint"""
        checkpoint_path = os.path.join(
            self.stage_output_dir, "checkpoints",
            f"stage_{self.stage}_model_round_{round_idx:04d}.pt"
        )
        
        self.global_model.save_checkpoint(
            filepath=checkpoint_path,
            epoch=round_idx,
            optimizer_state=None  # Global model doesn't have optimizer
        )
        
        logging.info(f"Saved {self.training_state.stage_name} checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load global model state
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set training state
            self.training_state.current_round = checkpoint.get('epoch', 0) + 1
            
            logging.info(f"Loaded {self.training_state.stage_name} checkpoint from round {self.training_state.current_round}")
            
        except Exception as e:
            logging.error(f"Failed to load {self.training_state.stage_name} checkpoint: {e}")
    
    def train(self):
        """Main training loop for the specific stage"""
        print(f"[{self.training_state.stage_name}] Starting training...")
        logging.info(f"Starting {self.training_state.stage_name} training")
        
        start_time = time.time()
        logging.info(f"Starting {self.training_state.stage_name} training for {self.training_state.total_rounds} rounds with {self.config.training.num_clients} clients")
        
        try:
            for round_idx in range(self.training_state.current_round, self.training_state.total_rounds):
                logging.info(f"Starting {self.training_state.stage_name} round {round_idx + 1}/{self.training_state.total_rounds}")
                round_metrics = self.federated_round(round_idx)
                logging.info(f"{self.training_state.stage_name} Round {round_idx + 1} completed: avg_loss={round_metrics.get('avg_train_loss', 0):.4f}")
                
                # Early stopping if loss is low enough
                if round_metrics.get('avg_val_loss', float('inf')) < 0.1:
                    logging.info("Early stopping: validation loss below threshold")
                    break
            
            # Save final model
            self.save_stage_checkpoint(self.training_state.total_rounds - 1)
            
            total_time = time.time() - start_time
            logging.info(f"{self.training_state.stage_name} training completed in {total_time:.2f} seconds")
            
            if wandb.run is not None:
                wandb.log({"total_training_time": total_time})
                wandb.finish()
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_stage_checkpoint(self.training_state.current_round)
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    def get_final_model_path(self) -> str:
        """Get the path to the final model checkpoint"""
        final_round = self.training_state.total_rounds - 1
        return os.path.join(
            self.stage_output_dir, "checkpoints",
            f"stage_{self.stage}_model_round_{final_round:04d}.pt"
        )