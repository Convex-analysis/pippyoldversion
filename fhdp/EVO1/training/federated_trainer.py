"""
Federated training pipeline for EVO-1 autonomous driving model

This module implements the complete federated learning training system
integrating EVO-1 with FHDP architecture for autonomous driving.
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import wandb
from pathlib import Path

# Import FHDP components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
try:
    from core.fhdp_system import FHDPSystem
    from edge_server.server import EdgeServer
    from vehicle_layer.vehicle import Vehicle
except ImportError:
    logging.warning("FHDP core components not found. Running in standalone mode.")

from EVO1.model.evo1_driving import FederatedEVO1Driving, EVO1DrivingOutput
from EVO1.data.nuscenes_loader import create_dataloader
from EVO1.data.augmentation import DrivingAugmentation, create_comprehensive_augmentation
from EVO1.utils.config import EVO1DrivingConfig, TrainingConfig, FHDPConfig
from EVO1.training.utils import TrainingMetrics, CheckpointManager, LearningRateScheduler


@dataclass
class FederatedTrainingState:
    """Training state for federated learning"""
    current_round: int = 0
    total_rounds: int = 100
    participating_clients: List[str] = None
    global_metrics: Dict[str, float] = None
    client_metrics: Dict[str, Dict[str, float]] = None
    aggregated_model_state: Dict[str, torch.Tensor] = None
    
    def __post_init__(self):
        if self.participating_clients is None:
            self.participating_clients = []
        if self.global_metrics is None:
            self.global_metrics = {}
        if self.client_metrics is None:
            self.client_metrics = {}
        if self.aggregated_model_state is None:
            self.aggregated_model_state = {}


class ClientTrainer:
    """Individual client trainer for federated learning"""
    
    def __init__(
        self,
        client_id: str,
        config: EVO1DrivingConfig,
        device: str = "cuda",
        shared_model: Optional[Any] = None
    ):
        self.client_id = client_id
        self.config = config
        self.device = device
        
        # Setup model (use shared model if provided)
        if shared_model is not None:
            self.model = shared_model
            print(f"[CLIENT_TRAINER] Using shared model for client {client_id}")
        else:
            self.model = FederatedEVO1Driving(
                config=config.model,
                training_config=config.training,
                device=device
            ).to(device)
            print(f"[CLIENT_TRAINER] Created new model for client {client_id}")
        
        self.model.set_client_id(client_id)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
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
        print(f"[TRAINER] About to setup data loader...")
        self.setup_data_loader()
        print(f"[TRAINER] Data loader setup complete")
        
        # Setup metrics tracking
        self.metrics = TrainingMetrics()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=os.path.join(config.output_dir, f"client_{client_id}"),
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
                with autocast():
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
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
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
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
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
                    f"Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
                
                # Log to wandb if enabled
                if wandb.run is not None:
                    wandb.log({
                        f"client_{self.client_id}/batch_loss": loss.item(),
                        f"client_{self.client_id}/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "global_round": global_round
                    })
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Compute epoch metrics
        epoch_metrics['train_loss'] = total_loss / max(num_batches, 1)
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
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
            logging.info(f"Client {self.client_id} - Round {global_round} - Epoch {epoch+1}/{num_epochs}")
            
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


class FederatedEVO1Trainer:
    """Main federated training orchestrator"""
    
    def __init__(
        self,
        config: EVO1DrivingConfig,
        fhdp_system: Optional[Any] = None,
        device: str = "cuda"
    ):
        print(f"[TRAINER] Initializing FederatedEVO1Trainer...")
        print(f"[TRAINER] Config loaded, device: {device}")
        self.config = config
        self.fhdp_system = fhdp_system
        self.device = device
        print(f"[TRAINER] Basic initialization done")
        
        # Initialize training state
        self.training_state = FederatedTrainingState(
            current_round=0,
            total_rounds=config.training.aggregation_rounds
        )
        
        # Setup global model (for reference)
        self.global_model = FederatedEVO1Driving(
            config=config.model,
            training_config=config.training,
            device=device
        ).to(device)
        
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
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
    
    def setup_client_trainers(self):
        """Setup client trainers"""
        num_clients = self.config.training.num_clients
        
        logging.info(f"Setting up {num_clients} client trainers (model sharing enabled)...")
        
        for client_id in range(num_clients):
            client_name = f"client_{client_id}"
            
            # Create trainer with shared global model (don't create separate model copies)
            self.client_trainers[client_name] = ClientTrainer(
                client_id=client_name,
                config=self.config,
                device=self.device,
                shared_model=self.global_model  # Pass the global model to avoid duplication
            )
        
        print(f"[TRAINER] About to log client trainer initialization...")
        logging.info(f"Initialized {num_clients} client trainers with shared model")
        print(f"[TRAINER] Client trainer initialization logged")
    
    def setup_output_dirs(self):
        print(f"[TRAINER] Setting up output directories...")
        """Setup output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "metrics"), exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, "logs", "federated_training.log")
        
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
                project="evo1-federated-driving",
                name=self.config.experiment_name,
                config=asdict(self.config)
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
        logging.info(f"Starting federated round {round_idx + 1}/{self.config.training.aggregation_rounds}")
        
        round_start_time = time.time()
        round_metrics = {}
        
        # Select participating clients
        selected_clients = self.select_clients()
        self.training_state.participating_clients = selected_clients
        
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
        self.training_state.aggregated_model_state = self.global_model.state_dict()
        
        # Compute round metrics
        round_time = time.time() - round_start_time
        round_metrics['round_time'] = round_time
        round_metrics['num_participating_clients'] = len(selected_clients)
        
        # Aggregate client metrics
        if client_metrics:
            # Extract the latest epoch's metrics (last epoch has highest number)
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
        self.training_state.global_metrics = round_metrics
        self.training_state.client_metrics = client_metrics
        
        # Log metrics
        self.log_round_metrics(round_idx, round_metrics, client_metrics)
        
        # Save checkpoint
        if (round_idx + 1) % self.config.save_frequency == 0:
            self.save_global_checkpoint(round_idx)
        
        return round_metrics
    
    def aggregate_client_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates from clients (Federated Averaging)"""
        if not client_updates:
            return {}
        
        # Simple FedAvg - equal weights for all clients
        client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Use first client's update structure as template
        aggregated_updates = {}
        
        first_client = list(client_updates.keys())[0]
        for param_name in client_updates[first_client]:
            # Weighted average of parameter updates
            param_updates = [client_updates[client][param_name] for client in client_updates]
            aggregated = sum(w * u for w, u in zip(client_weights, param_updates))
            aggregated_updates[param_name] = aggregated
        
        return aggregated_updates
    
    def log_round_metrics(self, round_idx: int, global_metrics: Dict[str, float], 
                         client_metrics: Dict[str, Dict[str, float]]):
        """Log metrics for current round"""
        # Log to console
        logging.info(f"Round {round_idx + 1} completed:")
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
            self.config.output_dir, "metrics", 
            f"round_{round_idx:04d}_metrics.json"
        )
        
        metrics_data = {
            'round': round_idx,
            'global_metrics': global_metrics,
            'client_metrics': {k: {str(mk): float(mv) for mk, mv in v.items()} 
                              for k, v in client_metrics.items()},
            'timestamp': time.time()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def save_global_checkpoint(self, round_idx: int):
        """Save global model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir, "checkpoints",
            f"global_model_round_{round_idx:04d}.pt"
        )
        
        self.global_model.save_checkpoint(
            filepath=checkpoint_path,
            epoch=round_idx,
            optimizer_state=None  # Global model doesn't have optimizer
        )
        
        logging.info(f"Saved global checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load global model state
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set training state
            self.training_state.current_round = checkpoint.get('epoch', 0) + 1
            
            logging.info(f"Loaded checkpoint from round {self.training_state.current_round}")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    
    def train(self):
        """Main training loop"""
        print("[TRAIN] Starting federated training...")
        logging.info("Starting federated training")
        
        start_time = time.time()
        logging.info(f"Starting federated training for {self.config.training.aggregation_rounds} rounds with {self.config.training.num_clients} clients")
        
        try:
            for round_idx in range(self.training_state.current_round, self.config.training.aggregation_rounds):
                logging.info(f"Starting federated round {round_idx + 1}/{self.config.training.aggregation_rounds}")
                round_metrics = self.federated_round(round_idx)
                logging.info(f"Round {round_idx + 1} completed: avg_loss={round_metrics.get('avg_train_loss', 0):.4f}")
                
                # Early stopping if loss is low enough
                if round_metrics.get('avg_val_loss', float('inf')) < 0.1:
                    logging.info("Early stopping: validation loss below threshold")
                    break
            
            # Save final model
            self.save_global_checkpoint(self.config.training.aggregation_rounds - 1)
            
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time:.2f} seconds")
            
            if wandb.run is not None:
                wandb.log({"total_training_time": total_time})
                wandb.finish()
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_global_checkpoint(self.training_state.current_round)
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on test set"""
        self.global_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        control_errors = []
        waypoint_errors = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                output = self.global_model(
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    mode="inference"
                )
                
                # Compute metrics
                target_controls = batch['future_controls']
                loss_dict = self.global_model.compute_loss(
                    output=output,
                    target_controls=target_controls
                )
                
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
                
                # Compute control errors (handle batch size mismatch)
                batch_size = min(output.controls.size(0), target_controls.size(0))
                control_error = F.mse_loss(
                    output.controls[:batch_size], 
                    target_controls[:batch_size], 
                    reduction='none'
                )
                control_errors.extend(control_error.flatten().cpu().numpy())
                
                # Compute waypoint errors (approximate)
                target_waypoints = self.global_model._controls_to_waypoints(target_controls[:batch_size])
                waypoint_error = F.mse_loss(
                    output.waypoints[:batch_size], 
                    target_waypoints, 
                    reduction='none'
                )
                waypoint_errors.extend(waypoint_error.flatten().cpu().numpy())
        
        metrics = {
            'test_loss': total_loss / max(num_batches, 1),
            'mean_control_error': np.mean(control_errors),
            'std_control_error': np.std(control_errors),
            'mean_waypoint_error': np.mean(waypoint_errors),
            'std_waypoint_error': np.std(waypoint_errors)
        }
        
        return metrics