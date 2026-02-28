"""
EVO-1 model adapted for autonomous driving in FHDP framework

This module implements the EVO-1 vision-language-action model specifically
for autonomous driving tasks, with integration for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

# Import original EVO-1 components (adapted path)
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Evo-1/Evo_1'))

try:
    from .internvl3.internvl3_embedder import InternVL3Embedder
    from .action_head.flow_matching import FlowmatchingActionHead
except ImportError:
    # Fallback implementation for standalone usage
    logging.warning("EVO-1 original components not found. Using fallback implementation.")
    
    class InternVL3Embedder(nn.Module):
        """Fallback vision-language embedder"""
        def __init__(self, model_name="OpenGVLab/InternVL3-1B", device="cuda", **kwargs):
            super().__init__()
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2048)
            ).to(device)
            self.text_projection = nn.Linear(512, 2048).to(device)
            self.device = device
        
        def forward(self, images, prompts=None):
            # Move images to device
            images = images.to(self.device)
            
            # Flatten batch and camera dimensions
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W)
            
            # Encode images
            vision_features = self.vision_encoder(images_flat)  # [B*N, 2048]
            vision_features = vision_features.view(B, N, -1)  # [B, N, 2048]
            
            # Simple text encoding fallback
            if prompts is not None:
                # This is a simplified version
                text_features = torch.zeros(B, 2048, device=images.device)
            else:
                text_features = torch.zeros(B, 2048, device=images.device)
            
            return vision_features, text_features
    
    class FlowmatchingActionHead(nn.Module):
        """Fallback action head for flow matching"""
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            # Simplified transformer architecture
            self.input_projection = nn.Linear(2048 + 256, 512)  # vision + state
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=6
            )
            self.action_projection = nn.Linear(512, config.max_waypoints * 3)  # waypoints
            
        def forward(self, fused_tokens, state, actions_gt=None):
            B, N, D = fused_tokens.shape
            
            # Combine vision features (average across views)
            vision_avg = fused_tokens.mean(dim=1)  # [B, 2048]
            
            # Combine with state
            combined = torch.cat([vision_avg, state], dim=-1)  # [B, 2048 + state_dim]
            
            # Project to transformer dimension
            x = self.input_projection(combined)  # [B, 512]
            
            # Apply transformer
            x = x.unsqueeze(0)  # [1, B, 512] for transformer
            x = self.transformer(x)  # [1, B, 512]
            x = x.squeeze(0)  # [B, 512]
            
            # Predict waypoints
            waypoints = self.action_projection(x)  # [B, max_waypoints * 3]
            waypoints = waypoints.view(B, self.config.max_waypoints, 3)
            
            return waypoints


# from ..utils.config import ModelConfig, TrainingConfig
# Config classes defined inline for standalone operation

@dataclass
class ModelConfig:
    vision_encoder: str = "OpenGVLab/InternVL3-1B"
    language_model: str = "Qwen/Qwen2.5-0.5B"
    sequence_length: int = 32
    hidden_dim: int = 4096
    vision_model_name: str = "OpenGVLab/InternVL3-1B"
    image_size: int = 224
    max_waypoints: int = 20
    action_dim: int = 8
    per_action_dim: int = 7

@dataclass  
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 20
    gradient_checkpointing: bool = False


@dataclass
class EVO1DrivingOutput:
    """Output structure for EVO-1 driving model"""
    waypoints: torch.Tensor  # [B, max_waypoints, 3] - predicted trajectory
    controls: torch.Tensor   # [B, max_waypoints, 3] - [steering, throttle, brake]
    vision_features: torch.Tensor  # [B, N, 2048] - vision embeddings
    confidence: torch.Tensor  # [B, 1] - confidence score
    intermediate_representations: Optional[Dict[str, torch.Tensor]] = None


class StateEncoder(nn.Module):
    """Encoder for vehicle state information"""
    
    def __init__(self, state_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.acceleration_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode vehicle state
        
        Args:
            state: [B, 12] - [x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az]
        
        Returns:
            encoded_state: [B, hidden_dim]
        """
        pose = state[:, :6]      # [x, y, z, roll, pitch, yaw]
        velocity = state[:, 6:9]  # [vx, vy, vz]
        acceleration = state[:, 9:12]  # [ax, ay, az]
        
        pose_encoded = self.pose_encoder(pose)
        velocity_encoded = self.velocity_encoder(velocity)
        acceleration_encoded = self.acceleration_encoder(acceleration)
        
        # Combine all encodings
        combined = torch.cat([pose_encoded, velocity_encoded, acceleration_encoded], dim=-1)
        return self.fusion_layer(combined)


class ControlHead(nn.Module):
    """Convert waypoints to control signals"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.waypoint_processor = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.temporal_aggregator = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.control_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 3)  # [steering, throttle, brake]
        )
    
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Convert waypoints to control sequence
        
        Args:
            waypoints: [B, max_waypoints, 3]
        
        Returns:
            controls: [B, max_waypoints, 3] - [steering, throttle, brake]
        """
        B, T, _ = waypoints.shape
        
        # Process each waypoint
        waypoint_features = []
        for t in range(T):
            wp = waypoints[:, t, :]  # [B, 3]
            wp_feat = self.waypoint_processor(wp)  # [B, hidden_dim//4]
            waypoint_features.append(wp_feat)
        
        # Stack and process temporally
        wp_sequence = torch.stack(waypoint_features, dim=1)  # [B, T, hidden_dim//4]
        
        # LSTM processing
        lstm_out, _ = self.temporal_aggregator(wp_sequence)  # [B, T, hidden_dim//2]
        
        # Predict controls
        controls = []
        for t in range(T):
            ctrl = self.control_predictor(lstm_out[:, t, :])  # [B, 3]
            controls.append(ctrl)
        
        controls = torch.stack(controls, dim=1)  # [B, T, 3]
        
        # Apply tanh activation to constrain controls
        controls = torch.tanh(controls)
        
        return controls


class EVO1Driving(nn.Module):
    """EVO-1 model adapted for autonomous driving"""
    
    def __init__(
        self,
        config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device
        
        # Vision-Language Encoder
        self.vl_embedder = InternVL3Embedder(
            model_name=getattr(config, 'vision_encoder', 'OpenGVLab/InternVL3-1B'),
            image_size=getattr(config, 'image_size', 224),
            device=device
        )
        
        # State Encoder
        self.state_encoder = StateEncoder(state_dim=12, hidden_dim=256).to(device)
        
        # Action Head (Waypoint Prediction)
        self.action_head = FlowmatchingActionHead(config).to(device)
        
        # Control Head
        self.control_head = ControlHead().to(device)
        
        # Confidence Estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.max_waypoints * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Initialize weights
        self._initialize_weights()
        
        # Setup for gradient checkpointing if enabled
        if self.training_config.gradient_checkpointing:
            self.gradient_checkpointing = True
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def set_stage1_mode(self):
        """Stage 1: Freeze vision-language backbone, only train action expert and integration module"""
        print(f"[STAGE1] Freezing vision-language backbone, training only action expert and integration")
        
        # Freeze vision-language encoder (backbone)
        for param in self.vl_embedder.parameters():
            param.requires_grad = False
        
        # Unfreeze action head (action expert)
        for param in self.action_head.parameters():
            param.requires_grad = True
        
        # Unfreeze state encoder (integration module)
        for param in self.state_encoder.parameters():
            param.requires_grad = True
        
        # Unfreeze control head (integration module)
        for param in self.control_head.parameters():
            param.requires_grad = True
        
        # Unfreeze confidence estimator
        for param in self.confidence_estimator.parameters():
            param.requires_grad = True
        
        # Log parameter counts
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[STAGE1] Trainable parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def set_stage2_mode(self):
        """Stage 2: Unfreeze all components for full-scale fine-tuning"""
        print(f"[STAGE2] Unfreezing all components for full-scale fine-tuning")
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        
        # Log parameter counts
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[STAGE2] Trainable parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def is_stage1(self, round_idx: int, stage1_rounds: int) -> bool:
        """Check if current round is in Stage 1"""
        return round_idx < stage1_rounds
    
    def forward(
        self,
        images: torch.Tensor,
        image_mask: torch.Tensor,
        state: torch.Tensor,
        instructions: Optional[List[str]] = None,
        future_controls: Optional[torch.Tensor] = None,
        mode: str = "inference"
    ) -> EVO1DrivingOutput:
        """Forward pass
        
        Args:
            images: [B, N, C, H, W] - Multi-view images
            image_mask: [B, N] - Availability mask for cameras
            state: [B, 12] - Vehicle state
            instructions: List of natural language instructions
            future_controls: [B, T, 3] - Ground truth controls for training
            mode: "inference" or "training"
        
        Returns:
            EVO1DrivingOutput
        """
        batch_size = images.shape[0]
        
        # Extract vision and language features
        text_features = None
        if hasattr(self.vl_embedder, 'get_fused_image_text_embedding_from_tensor_images'):
            # Original EVO-1 method - convert (B, N, C, H, W) to list of (N, C, H, W)
            # and then flatten each view to individual (C, H, W) images
            B, N, C, H, W = images.shape
            image_list = []
            mask_list = []
            for b in range(batch_size):
                for n in range(N):
                    # Add individual (C, H, W) images
                    image_list.append(images[b, n])  # Shape: (C, H, W)
                    # Expand mask for each image view
                    mask_list.append(image_mask[b, n].cpu().item())
            
            vision_features = self.vl_embedder.get_fused_image_text_embedding_from_tensor_images(
                image_tensors=image_list,
                image_mask=torch.tensor(mask_list, dtype=torch.bool),
                text_prompt=instructions[0] if instructions else "",
                return_cls_only=False
            )
        else:
            # Fallback method
            vision_features, text_features = self.vl_embedder(images, instructions)
        
        # Encode vehicle state
        encoded_state = self.state_encoder(state)  # [B, hidden_dim]
        
        # Predict waypoints using action head
        if mode == "training" and future_controls is not None:
            # Training mode with ground truth - action_head returns (pred_velocity, noise)
            result = self.action_head(
                fused_tokens=vision_features,
                state=encoded_state,
                actions_gt=future_controls
            )
            if isinstance(result, tuple):
                # In training mode, use the velocity prediction
                pred_velocity, noise = result
                # Reshape pred_velocity to match expected waypoints format [B, T, per_action_dim]
                if pred_velocity.dim() == 2:
                    # pred_velocity is [B, horizon*per_action_dim], reshape to [B, horizon, per_action_dim]
                    waypoints = pred_velocity.view(pred_velocity.size(0), self.action_head.horizon, self.action_head.per_action_dim)
                else:
                    waypoints = pred_velocity
            else:
                waypoints = result
        else:
            # Inference mode
            waypoints = self.action_head(
                fused_tokens=vision_features,
                state=encoded_state
            )
            
            # Handle different possible shapes from inference
            if waypoints.dim() == 2:
                # waypoints is [B, action_dim_total], reshape to [B, horizon, per_action_dim]
                waypoints = waypoints.view(waypoints.size(0), self.action_head.horizon, self.action_head.per_action_dim)
            elif waypoints.dim() == 1:
                # waypoints is [action_dim_total], add batch dimension and reshape
                waypoints = waypoints.view(1, self.action_head.horizon, self.action_head.per_action_dim)
        
        # Convert waypoints to control signals
        controls = self.control_head(waypoints)
        
        # Estimate confidence
        actual_batch_size = waypoints.shape[0]
        waypoints_flat = waypoints.view(actual_batch_size, -1)
        confidence = self.confidence_estimator(waypoints_flat)
        
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
    
    def compute_loss(
        self,
        output: EVO1DrivingOutput,
        target_controls: torch.Tensor,
        confidence_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        
        # Ensure target tensors match output batch size
        output_batch_size = output.controls.size(0)
        if target_controls.size(0) != output_batch_size:
            target_controls = target_controls[:output_batch_size]
        
        # Control prediction loss (MSE)
        control_loss = F.mse_loss(output.controls, target_controls)
        
        # Waypoint trajectory loss (MSE)
        # Convert target controls to waypoints (approximate)
        target_waypoints = self._controls_to_waypoints(target_controls)
        waypoint_loss = F.mse_loss(output.waypoints, target_waypoints)
        
        # Confidence regularization (encourage appropriate confidence)
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
    
    def _controls_to_waypoints(self, controls: torch.Tensor) -> torch.Tensor:
        """Convert control sequence to waypoints (simplified kinematic model)"""
        B, T, _ = controls.shape
        
        waypoints = torch.zeros(B, T, 3, device=controls.device)
        
        # Initialize state variables
        current_pos = torch.zeros(B, 2, device=controls.device)
        current_heading = torch.zeros(B, device=controls.device)
        current_speed = torch.zeros(B, device=controls.device)
        
        dt = 0.1  # 100ms timestep
        
        for t in range(T):
            # Save current position as waypoint
            waypoints[:, t, :2] = current_pos
            waypoints[:, t, 2] = 0.0  # z = 0
            
            if t < T - 1:  # Update for next timestep
                # Get control inputs
                steering = controls[:, t, 0]  # Normalized steering angle
                throttle = torch.relu(controls[:, t, 1])  # Normalized throttle
                brake = torch.relu(controls[:, t, 2])  # Normalized brake
                
                # Convert normalized controls to physical values
                steering_angle = steering * self.config.max_steering  # Convert to radians
                throttle_force = throttle * 10.0  # Simplified throttle force
                brake_force = brake * 5.0  # Simplified brake force
                
                # Update speed
                acceleration = throttle_force - brake_force - 0.1 * current_speed  # Add drag
                current_speed = torch.clamp(current_speed + acceleration * dt, 0.0, self.config.max_speed)
                
                # Update heading
                turning_rate = steering_angle * 2.0  # Simplified turning model
                current_heading = (current_heading + turning_rate * dt) % (2 * torch.pi)
                
                # Update position
                dx = current_speed * dt * torch.cos(current_heading)
                dy = current_speed * dt * torch.sin(current_heading)
                current_pos += torch.stack([dx, dy], dim=1)
        
        return waypoints
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'device': self.device
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "cuda"):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        config = checkpoint['config']
        model = cls(config, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['epoch']


class FederatedEVO1Driving(EVO1Driving):
    """EVO-1 model with federated learning capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Federated learning components
        self.client_id = None
        self.global_model_state = None
        self.local_updates = []
        
        # For differential privacy
        self.noise_scale = 0.0
        self.clip_norm = 1.0
    
    def set_client_id(self, client_id: str):
        """Set client identifier for federated learning"""
        self.client_id = client_id
    
    def set_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Set global model parameters"""
        self.global_model_state = global_state_dict
        self.load_state_dict(global_state_dict)
    
    def get_local_updates(self) -> Dict[str, torch.Tensor]:
        """Get local model updates (difference from global model)"""
        if self.global_model_state is None:
            return self.state_dict()
        
        local_updates = {}
        for name, param in self.named_parameters():
            if name in self.global_model_state:
                local_updates[name] = param.data - self.global_model_state[name]
        
        return local_updates
    
    def apply_differential_privacy(self, updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to model updates"""
        if self.noise_scale == 0.0:
            return updates
        
        noisy_updates = {}
        for name, param in updates.items():
            # Clip gradients
            param_norm = param.norm()
            if param_norm > self.clip_norm:
                param = param * (self.clip_norm / param_norm)
            
            # Add Gaussian noise
            noise = torch.randn_like(param) * self.noise_scale
            noisy_updates[name] = param + noise
        
        return noisy_updates
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                         weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Aggregate updates from multiple clients (FedAvg)"""
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        aggregated = {}
        
        # Initialize with first client
        for name, param in client_updates[0].items():
            aggregated[name] = weights[0] * param
        
        # Add weighted contributions from other clients
        for i, updates in enumerate(client_updates[1:], 1):
            for name, param in updates.items():
                if name in aggregated:
                    aggregated[name] += weights[i] * param
        
        return aggregated
    
    def update_from_aggregation(self, aggregated_updates: Dict[str, torch.Tensor]):
        """Update model with aggregated parameters"""
        current_state = self.state_dict()
        
        for name, update in aggregated_updates.items():
            if name in current_state:
                current_state[name] = current_state[name] + update
        
        self.load_state_dict(current_state)
    
    def compute_client_drift(self) -> float:
        """Compute drift from global model (for fairness analysis)"""
        if self.global_model_state is None:
            return 0.0
        
        total_drift = 0.0
        param_count = 0
        
        for name, param in self.named_parameters():
            if name in self.global_model_state:
                drift = (param.data - self.global_model_state[name]).norm().item()
                total_drift += drift
                param_count += 1
        
        return total_drift / max(param_count, 1)