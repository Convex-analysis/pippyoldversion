"""
Training utilities for EVO-1 federated learning

This module contains utility classes for training metrics tracking,
checkpoint management, and learning rate scheduling.
"""

import os
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, Optional
from utils.config import TrainingConfig


class TrainingMetrics:
    """Metrics tracking for training"""
    
    def __init__(self):
        self.metrics = {}
        self.step = 0
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.step += 1
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of specified metric"""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values) if values else 0.0


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, 
            f"checkpoint_epoch_{epoch:04d}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints"""
        checkpoints = sorted([
            f for f in os.listdir(self.output_dir) 
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ])
        
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                os.remove(os.path.join(self.output_dir, checkpoint))


class LearningRateScheduler:
    """Learning rate scheduler for federated training"""
    
    def __init__(self, optimizer: optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        
        # Setup scheduler
        if config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.aggregation_rounds * config.local_epochs,
                eta_min=config.min_lr
            )
        elif config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.aggregation_rounds // 3,
                gamma=0.5
            )
        else:
            self.scheduler = None
    
    def step(self):
        """Step the scheduler"""
        if self.scheduler:
            self.scheduler.step()
        self.step_count += 1
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']