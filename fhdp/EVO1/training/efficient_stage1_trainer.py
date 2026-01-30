"""
Efficient Stage 1 Trainer for Limited GPU Resources

This trainer implements a memory-efficient approach for Stage 1 training:
- Loads one frozen backbone model
- Creates multiple action expert heads that share the same backbone
- Simulates federated learning with minimal GPU memory usage
- Allows training multiple "clients" on a single GPU
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import wandb
from pathlib import Path

# Import EVO-1 components
from ..utils.config import EVO1DrivingConfig
from ..model.evo1_driving import FederatedEVO1Driving
from ..data.nuscenes_loader import create_dataloader
from ..data.augmentation import DrivingAugmentation, create_comprehensive_augmentation


@dataclass
class EfficientStage1State:
    """Training state for efficient Stage 1 training"""
    current_round: int = 0
    total_rounds: int = 50
    num_clients: int = 4
    client_metrics: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.client_metrics is None:
            self.client_metrics = {}


class SharedBackboneModel(nn.Module):
    """Model with shared backbone and multiple client-specific action experts"""
    
    def __init__(
        self,
        config: EVO1DrivingConfig,
        num_clients: int,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.config = config
        self.num_clients = num_clients
        self.device = device
        
        # Create one shared backbone (frozen)
        self.shared_backbone = self._create_backbone(config)
        
        # Freeze the shared backbone
        for param in self.shared_backbone.parameters():
            param.requires_grad = False
        
        # Create multiple client-specific action experts
        self.client_experts = nn.ModuleDict()
        self.client_state_encoders = nn.ModuleDict()
        self.client_control_heads = nn.ModuleDict()
        self.client_confidence_estimators = nn.ModuleDict()
        
        for client_id in range(num_clients):
            client_name = f"client_{client_id}"
            
            # Action expert (trainable)
            self.client_experts[client_name] = self._create_action_expert(config)
            
            # State encoder (trainable)
            self.client_state_encoders[client_name] = self._create_state_encoder()
            
            # Control head (trainable)
            self.client_control_heads[client_name] = self._create_control_head()
            
            # Confidence estimator (trainable)
            self.client_confidence_estimators[client_name] = self._create_confidence_estimator(config)
        
        # Move all components to device
        self.to(device)
        
        # Log parameter counts
        self._log_parameter_info()
    
    def _create_backbone(self, config):
        """Create the shared vision-language backbone"""
        from ..model.evo1_driving import InternVL3Embedder
        
        # Create backbone without action head
        backbone = InternVL3Embedder(
            model_name=getattr(config.model, 'vision_encoder', 'OpenGVLab/InternVL3-1B'),
            image_size=getattr(config.model, 'image_size', 224),
            device=self.device
        )
        
        return backbone
    
    def _create_action_expert(self, config):
        """Create action expert (flow matching head)"""
        from ..model.evo1_driving import FlowmatchingActionHead
        return FlowmatchingActionHead(config.model)
    
    def _create_state_encoder(self):
        """Create state encoder"""
        from ..model.evo1_driving import StateEncoder
        return StateEncoder(state_dim=12, hidden_dim=256)
    
    def _create_control_head(self):
        """Create control head"""
        from ..model.evo1_driving import ControlHead
        return ControlHead()
    
    def _create_confidence_estimator(self, config):
        """Create confidence estimator"""
        hidden_dim = 256
        max_waypoints = config.model.max_waypoints
        
        return nn.Sequential(
            nn.Linear(max_waypoints * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _log_parameter_info(self):
        """Log parameter information"""
        backbone_params = sum(p.numel() for p in self.shared_backbone.parameters())
        expert_params = sum(p.numel() for p in self.client_experts.parameters())
        other_params = sum(p.numel() for p in self.client_state_encoders.parameters()) + \
                     sum(p.numel() for p in self.client_control_heads.parameters()) + \
                     sum(p.numel() for p in self.client_confidence_estimators.parameters())
        
        total_params = backbone_params + expert_params + other_params
        trainable_params = expert_params + other_params  # Backbone is frozen
        
        print(f"[EFFICIENT_STAGE1] Model Parameter Summary:")
        print(f"  Backbone (frozen): {backbone_params:,} parameters")
        print(f"  Action Experts: {expert_params:,} parameters")
        print(f"  Other Components: {other_params:,} parameters")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Memory Efficiency: ~{100*trainable_params/total_params:.1f}% of normal Stage 1")
    
    def forward(self, client_id: str, images, image_mask, state, instructions, future_controls=None, mode="inference"):
        """Forward pass for specific client"""
        
        # Get shared backbone features (no gradients)
        with torch.no_grad():
            vision_features, text_features = self.shared_backbone(images, instructions)
        
        # Get client-specific components
        action_expert = self.client_experts[client_id]
        state_encoder = self.client_state_encoders[client_id]
        control_head = self.client_control_heads[client_id]
        confidence_estimator = self.client_confidence_estimators[client_id]
        
        # Encode state
        encoded_state = state_encoder(state)
        
        # Predict waypoints using action expert
        if mode == "training" and future_controls is not None:
            waypoints = action_expert(
                fused_tokens=vision_features,
                state=encoded_state,
                actions_gt=future_controls
            )
            
            # Handle different output formats from action expert
            if isinstance(waypoints, tuple):
                waypoints, _ = waypoints
        else:
            waypoints = action_expert(
                fused_tokens=vision_features,
                state=encoded_state
            )
        
        # Handle different possible shapes from action expert
        if waypoints.dim() == 2:
            waypoints = waypoints.view(waypoints.size(0), action_expert.horizon, action_expert.per_action_dim)
        elif waypoints.dim() == 1:
            waypoints = waypoints.view(1, action_expert.horizon, action_expert.per_action_dim)
        
        # Convert waypoints to control signals
        controls = control_head(waypoints)
        
        # Estimate confidence
        waypoints_flat = waypoints.view(waypoints.size(0), -1)
        confidence = confidence_estimator(waypoints_flat)
        
        # Create output structure
        from ..model.evo1_driving import EVO1DrivingOutput
        return EVO1DrivingOutput(
            waypoints=waypoints,
            controls=controls,
            vision_features=vision_features,
            confidence=confidence,
            intermediate_representations={
                'encoded_state': encoded_state,
                'text_features': text_features if text_features is not None else None
            }
        )
    
    def get_client_parameters(self, client_id: str):
        """Get trainable parameters for specific client"""
        params = []
        params.extend(self.client_experts[client_id].parameters())
        params.extend(self.client_state_encoders[client_id].parameters())
        params.extend(self.client_control_heads[client_id].parameters())
        params.extend(self.client_confidence_estimators[client_id].parameters())
        return params
    
    def compute_loss(self, client_id: str, output, target_controls, confidence_weight=0.1):
        """Compute loss for specific client"""
        import torch.nn.functional as F
        
        # Ensure target tensors match output batch size
        output_batch_size = output.controls.size(0)
        if target_controls.size(0) != output_batch_size:
            target_controls = target_controls[:output_batch_size]
        
        # Control prediction loss (MSE)
        control_loss = F.mse_loss(output.controls, target_controls)
        
        # Waypoint trajectory loss (MSE)
        target_waypoints = self._controls_to_waypoints(target_controls)
        waypoint_loss = F.mse_loss(output.waypoints, target_waypoints)
        
        # Confidence regularization
        control_error = F.mse_loss(output.controls, target_controls, reduction='none')
        confidence_target = torch.exp(-control_error.mean(dim=(1, 2))).unsqueeze(1)
        confidence_loss = F.mse_loss(output.confidence, confidence_target.detach())
        
        # Total loss
        total_loss = control_loss + 0.5 * waypoint_loss + confidence_weight * confidence_loss
        
        return {
            'total_loss': total_loss,
            'control_loss': control_loss,
            'waypoint_loss': waypoint_loss,
            'confidence_loss': confidence_loss
        }
    
    def _controls_to_waypoints(self, controls):
        """Convert control sequence to waypoints (simplified)"""
        B, T, _ = controls.shape
        waypoints = torch.zeros(B, T, 3, device=controls.device)
        
        # Simple conversion: use controls as waypoints
        waypoints[:, :, :2] = controls[:, :, :2] * 10.0  # Scale position
        waypoints[:, :, 2] = torch.zeros(B, T, device=controls.device)  # z = 0
        
        return waypoints


class EfficientStage1Trainer:
    """Memory-efficient Stage 1 trainer using shared backbone"""
    
    def __init__(
        self,
        config: EVO1DrivingConfig,
        device: str = "cuda",
        resume_from_checkpoint: Optional[str] = None
    ):
        self.config = config
        self.device = device
        
        # Initialize training state
        self.training_state = EfficientStage1State(
            current_round=0,
            total_rounds=config.training.stage1_rounds,
            num_clients=config.training.num_clients
        )
        
        # Create efficient model with shared backbone
        self.model = SharedBackboneModel(
            config=config,
            num_clients=config.training.num_clients,
            device=device
        )
        
        # Create optimizers for each client
        self.client_optimizers = {}
        self.client_schedulers = {}
        self.client_scalers = {}
        
        for client_id in range(config.training.num_clients):
            client_name = f"client_{client_id}"
            
            # Get client-specific parameters (backbone is frozen)
            client_params = self.model.get_client_parameters(client_name)
            
            # Create optimizer
            optimizer = optim.AdamW(
                client_params,
                lr=config.training.stage1_lr,
                weight_decay=config.training.weight_decay
            )
            self.client_optimizers[client_name] = optimizer
            
            # Create scaler for mixed precision
            if config.training.mixed_precision:
                self.client_scalers[client_name] = GradScaler()
        
        # Setup augmentation
        self.augmentation = create_comprehensive_augmentation()
        
        # Setup data loaders for each client
        self.setup_data_loaders()
        
        # Setup output directories
        self.setup_output_dirs()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb if enabled
        self.setup_wandb()
        
        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
    
    def setup_data_loaders(self):
        """Setup data loaders for each client"""
        self.client_loaders = {}
        
        for client_id in range(self.config.training.num_clients):
            # Extract client number
            client_num = client_id
            
            # Create train loader
            train_loader = create_dataloader(
                config=self.config.data,
                model_config=self.config.model,
                split="train",
                client_id=client_num,
                num_clients=self.config.training.num_clients,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # Create val loader
            val_loader = create_dataloader(
                config=self.config.data,
                model_config=self.config.model,
                split="val",
                client_id=client_num,
                num_clients=self.config.training.num_clients,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            client_name = f"client_{client_id}"
            self.client_loaders[client_name] = {
                'train': train_loader,
                'val': val_loader
            }
        
        logging.info(f"Setup {self.config.training.num_clients} client data loaders")
    
    def setup_output_dirs(self):
        """Setup output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "metrics"), exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, "logs", "efficient_stage1_training.log")
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if os.getenv("WANDB_API_KEY"):
            wandb.init(
                project="evo1-efficient-stage1",
                name=f"{self.config.experiment_name}_efficient_stage1",
                config={
                    **self.config.__dict__,
                    "training_type": "efficient_stage1",
                    "num_clients": self.config.num_clients,
                    "shared_backbone": True
                }
            )
            logging.info("Initialized wandb logging")
        else:
            logging.info("Wandb not configured. Skipping wandb logging.")
    
    def train_client_epoch(self, client_name: str, global_round: int) -> Dict[str, float]:
        """Train one client for one epoch"""
        
        client_id = int(client_name.split('_')[1])
        optimizer = self.client_optimizers[client_name]
        scaler = self.client_scalers.get(client_name)
        loaders = self.client_loaders[client_name]
        
        epoch_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        self.model.train()
        
        for batch_idx, batch in enumerate(loaders['train']):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply augmentation
            if self.config.training.federated_learning:
                batch = self.augmentation.augment_batch(batch)
            
            # Forward pass
            optimizer.zero_grad()
            
            if self.config.training.mixed_precision and scaler is not None:
                with autocast():
                    output = self.model(
                        client_id=client_name,
                        images=batch['images'],
                        image_mask=batch['image_mask'],
                        state=batch['state'],
                        instructions=batch['instructions'],
                        future_controls=batch['future_controls'],
                        mode="training"
                    )
                    
                    loss_dict = self.model.compute_loss(
                        client_id=client_name,
                        output=output,
                        target_controls=batch['future_controls']
                    )
                    loss = loss_dict['total_loss']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_client_parameters(client_name),
                        self.config.training.max_grad_norm
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                output = self.model(
                    client_id=client_name,
                    images=batch['images'],
                    image_mask=batch['image_mask'],
                    state=batch['state'],
                    instructions=batch['instructions'],
                    future_controls=batch['future_controls'],
                    mode="training"
                )
                
                loss_dict = self.model.compute_loss(
                    client_id=client_name,
                    output=output,
                    target_controls=batch['future_controls']
                )
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_client_parameters(client_name),
                        self.config.training.max_grad_norm
                    )
                
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                logging.info(
                    f"Client {client_name} - Round {global_round} - "
                    f"Batch {batch_idx}/{len(loaders['train'])} - "
                    f"Loss: {loss.item():.4f}"
                )
                
                if wandb.run is not None:
                    wandb.log({
                        f"{client_name}/batch_loss": loss.item(),
                        f"{client_name}/learning_rate": optimizer.param_groups[0]['lr'],
                        "global_round": global_round
                    })
        
        # Compute epoch metrics
        epoch_metrics['train_loss'] = total_loss / max(num_batches, 1)
        epoch_metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        
        return epoch_metrics
    
    def train_local(self, client_name: str, num_epochs: int, global_round: int) -> Dict[str, float]:
        """Train specific client locally"""
        
        all_metrics = {}
        
        for epoch in range(num_epochs):
            logging.info(f"Client {client_name} - Round {global_round} - Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_metrics = self.train_client_epoch(client_name, global_round)
            
            # Update metrics
            all_metrics.update({f"epoch_{epoch}_{k}": v for k, v in train_metrics.items()})
        
        return all_metrics
    
    def federated_round(self, round_idx: int) -> Dict[str, float]:
        """Execute one federated learning round"""
        
        logging.info(f"Starting Efficient Stage 1 round {round_idx + 1}/{self.training_state.total_rounds}")
        
        round_start_time = time.time()
        round_metrics = {}
        client_metrics = {}
        
        # Train each client locally
        for client_id in range(self.config.training.num_clients):
            client_name = f"client_{client_id}"
            
            # Local training
            metrics = self.train_local(
                client_name=client_name,
                num_epochs=self.config.training.local_epochs,
                global_round=round_idx
            )
            
            client_metrics[client_name] = metrics
        
        # Simulate federated aggregation (average parameters across clients)
        self.aggregate_client_parameters()
        
        # Compute round metrics
        round_time = time.time() - round_start_time
        round_metrics['round_time'] = round_time
        round_metrics['num_clients'] = self.config.training.num_clients
        
        # Aggregate client metrics
        if client_metrics:
            avg_train_loss = np.mean([
                max([v for k, v in metrics.items() if 'train_loss' in k], default=0)
                for metrics in client_metrics.values()
            ])
            round_metrics['avg_train_loss'] = avg_train_loss
        
        # Update training state
        self.training_state.current_round = round_idx
        self.training_state.client_metrics = client_metrics
        
        # Log metrics
        self.log_round_metrics(round_idx, round_metrics, client_metrics)
        
        # Save checkpoint
        if (round_idx + 1) % self.config.save_frequency == 0:
            self.save_checkpoint(round_idx)
        
        return round_metrics
    
    def aggregate_client_parameters(self):
        """Aggregate parameters across clients (FedAvg)"""
        
        if self.config.training.num_clients <= 1:
            return  # No aggregation needed for single client
        
        # Collect parameter states
        client_params = {}
        for client_id in range(self.config.training.num_clients):
            client_name = f"client_{client_id}"
            param_dict = {}
            
            # Action expert parameters
            for name, param in self.model.client_experts[client_name].named_parameters():
                param_dict[f'action_expert.{name}'] = param.clone().detach()
            
            # State encoder parameters
            for name, param in self.model.client_state_encoders[client_name].named_parameters():
                param_dict[f'state_encoder.{name}'] = param.clone().detach()
            
            # Control head parameters
            for name, param in self.model.client_control_heads[client_name].named_parameters():
                param_dict[f'control_head.{name}'] = param.clone().detach()
            
            # Confidence estimator parameters
            for name, param in self.model.client_confidence_estimators[client_name].named_parameters():
                param_dict[f'confidence_estimator.{name}'] = param.clone().detach()
            
            client_params[client_name] = param_dict
        
        # Compute FedAvg (simple average)
        aggregated_params = {}
        first_client = f"client_{0}"
        
        for param_name in client_params[first_client]:
            # Average across all clients
            avg_param = torch.stack([
                client_params[f"client_{i}"][param_name] 
                for i in range(self.config.training.num_clients)
            ]).mean(dim=0)
            
            aggregated_params[param_name] = avg_param
        
        # Update all clients with averaged parameters
        for client_id in range(self.config.training.num_clients):
            client_name = f"client_{client_id}"
            
            # Update action expert
            for name, param in self.model.client_experts[client_name].named_parameters():
                param_key = f'action_expert.{name}'
                if param_key in aggregated_params:
                    param.data.copy_(aggregated_params[param_key])
            
            # Update state encoder
            for name, param in self.model.client_state_encoders[client_name].named_parameters():
                param_key = f'state_encoder.{name}'
                if param_key in aggregated_params:
                    param.data.copy_(aggregated_params[param_key])
            
            # Update control head
            for name, param in self.model.client_control_heads[client_name].named_parameters():
                param_key = f'control_head.{name}'
                if param_key in aggregated_params:
                    param.data.copy_(aggregated_params[param_key])
            
            # Update confidence estimator
            for name, param in self.model.client_confidence_estimators[client_name].named_parameters():
                param_key = f'confidence_estimator.{name}'
                if param_key in aggregated_params:
                    param.data.copy_(aggregated_params[param_key])
        
        logging.info(f"[AGGREGATION] Aggregated parameters across {self.config.training.num_clients} clients")
    
    def log_round_metrics(self, round_idx: int, global_metrics: Dict[str, float], 
                         client_metrics: Dict[str, Dict[str, float]]):
        """Log metrics for current round"""
        
        # Log to console
        logging.info(f"Efficient Stage 1 Round {round_idx + 1} completed:")
        for metric, value in global_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "round": round_idx,
                **{f"global/{k}": v for k, v in global_metrics.items()}
            })
            
            for client_id, metrics in client_metrics.items():
                wandb.log({
                    f"{client_id}/train_loss": metrics.get('train_loss', 0),
                    "round": round_idx
                })
        
        # Save metrics to file
        metrics_file = os.path.join(
            self.config.output_dir, "metrics", 
            f"efficient_stage1_round_{round_idx:04d}_metrics.json"
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
    
    def save_checkpoint(self, round_idx: int):
        """Save training checkpoint"""
        
        checkpoint_path = os.path.join(
            self.config.output_dir, "checkpoints",
            f"efficient_stage1_round_{round_idx:04d}.pt"
        )
        
        # Save model state and optimizers
        checkpoint = {
            'round': round_idx,
            'model_state_dict': self.model.state_dict(),
            'client_optimizers': {
                f"client_{i}": self.client_optimizers[f"client_{i}"].state_dict()
                for i in range(self.config.num_clients)
            },
            'training_state': self.training_state.__dict__,
            'config': self.config.__dict__
        }
        
        if self.config.training.mixed_precision:
            checkpoint['client_scalers'] = {
                f"client_{i}": self.client_scalers[f"client_{i}"].state_dict()
                for i in range(self.config.num_clients)
            }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved Efficient Stage 1 checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizers
            if 'client_optimizers' in checkpoint:
                for i in range(self.config.num_clients):
                    client_name = f"client_{i}"
                    if client_name in checkpoint['client_optimizers']:
                        self.client_optimizers[client_name].load_state_dict(
                            checkpoint['client_optimizers'][client_name]
                        )
            
            # Load scalers
            if self.config.training.mixed_precision and 'client_scalers' in checkpoint:
                for i in range(self.config.num_clients):
                    client_name = f"client_{i}"
                    if client_name in checkpoint['client_scalers']:
                        self.client_scalers[client_name].load_state_dict(
                            checkpoint['client_scalers'][client_name]
                        )
            
            # Set training state
            if 'training_state' in checkpoint:
                for key, value in checkpoint['training_state'].items():
                    setattr(self.training_state, key, value)
            
            logging.info(f"Loaded Efficient Stage 1 checkpoint from round {self.training_state.current_round}")
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    
    def create_final_models(self):
        """Create separate model files for each client"""
        
        output_dir = os.path.join(self.config.output_dir, "final_models")
        os.makedirs(output_dir, exist_ok=True)
        
        for client_id in range(self.config.training.num_clients):
            client_name = f"client_{client_id}"
            
            # Create individual model
            individual_model = FederatedEVO1Driving(
                config=self.config.model,
                training_config=self.config.training,
                device=self.device
            )
            
            # Copy shared backbone state
            individual_model.vl_embedder.load_state_dict(
                self.model.shared_backbone.state_dict()
            )
            
            # Copy client-specific components
            individual_model.action_head.load_state_dict(
                self.model.client_experts[client_name].state_dict()
            )
            individual_model.state_encoder.load_state_dict(
                self.model.client_state_encoders[client_name].state_dict()
            )
            individual_model.control_head.load_state_dict(
                self.model.client_control_heads[client_name].state_dict()
            )
            
            # Set to Stage 1 mode (frozen backbone)
            individual_model.set_stage1_mode()
            
            # Save individual model
            model_path = os.path.join(output_dir, f"stage1_{client_name}_model.pt")
            individual_model.save_checkpoint(
                filepath=model_path,
                epoch=self.training_state.total_rounds - 1,
                optimizer_state=None
            )
            
            logging.info(f"Saved individual model for {client_name}: {model_path}")
    
    def train(self):
        """Main training loop"""
        
        print("[EFFICIENT_STAGE1] Starting efficient Stage 1 training...")
        print("[EFFICIENT_STAGE1] Using shared backbone with multiple action experts")
        print("[EFFICIENT_STAGE1] This maximizes GPU utilization for limited resources")
        logging.info("Starting efficient Stage 1 training")
        
        start_time = time.time()
        
        try:
            for round_idx in range(self.training_state.current_round, self.training_state.total_rounds):
                logging.info(f"Starting efficient Stage 1 round {round_idx + 1}/{self.training_state.total_rounds}")
                
                round_metrics = self.federated_round(round_idx)
                logging.info(f"Round {round_idx + 1} completed: avg_loss={round_metrics.get('avg_train_loss', 0):.4f}")
                
                # Early stopping if loss is low enough
                if round_metrics.get('avg_train_loss', float('inf')) < 0.05:
                    logging.info("Early stopping: training loss below threshold")
                    break
            
            # Save final checkpoint
            self.save_checkpoint(self.training_state.total_rounds - 1)
            
            # Create individual models for each client
            self.create_final_models()
            
            total_time = time.time() - start_time
            logging.info(f"Efficient Stage 1 training completed in {total_time:.2f} seconds")
            
            if wandb.run is not None:
                wandb.log({"total_training_time": total_time})
                wandb.finish()
            
            print("\n[EFFICIENT_STAGE1] Training completed successfully!")
            print("[EFFICIENT_STAGE1] Individual models created for each client")
            print("[EFFICIENT_STAGE1] Ready for Stage 2 fine-tuning")
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(self.training_state.current_round)
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise