"""
Vehicle Encoder Client with FHDP Integration

This module implements vehicle-side encoder training using FHDP's
native vehicle architecture and training engine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from vehicle_layer.vehicle import Vehicle
from vehicle_layer.training_engine import TrainingExecutor
from core.types import VehicleInfo, VehicleState, TrainingMode, ModelUpdate
from core.hardware_adapter import HardwareAdapter
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any
import logging

# Import EVO-1 components
try:
    from model.evo1_driving import FlowmatchingActionHead
except ImportError:
    logging.warning("EVO-1 action head not found. Using fallback implementation.")
    
    class FlowmatchingActionHead(nn.Module):
        """Fallback action head for vehicle encoder"""
        def __init__(self, input_dim=2048, hidden_dim=1024, num_actions=100):
            super().__init__()
            self.action_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_actions)
            )
        
        def forward(self, features):
            return self.action_head(features)


class VehicleEncoderClient:
    """
    Vehicle-side encoder training client using FHDP's native architecture
    
    This class extends FHDP's Vehicle with EVO-1 specific encoder training
    while maintaining full compatibility with FHDP's native training system.
    """
    
    def __init__(self, vehicle_id: str, edge_server_url: str, 
                 encoder_config: Optional[Dict] = None):
        self.vehicle_id = vehicle_id
        self.edge_server_url = edge_server_url
        
        # Initialize FHDP vehicle
        self.fhdp_vehicle = Vehicle(
            vehicle_id=vehicle_id,
            initial_position=(0.0, 0.0),  # Default position
            resources=self._get_vehicle_resources()
        )
        
        # Initialize EVO-1 encoder components
        self.encoder = self._initialize_encoder(encoder_config)
        self.action_head = FlowmatchingActionHead(
            input_dim=self._get_encoder_output_dim()
        )
        
        # Use FHDP's native training executor
        self.training_executor = self.fhdp_vehicle.training_executor
        
        # Hardware adapter from FHDP
        self.hardware_adapter = HardwareAdapter(vehicle_id)
        
        # Training state
        self.training_state = {
            'current_round': 0,
            'encoder_optimizer': optim.AdamW(
                self.encoder.parameters(), lr=1e-3
            ),
            'action_head_optimizer': optim.AdamW(
                self.action_head.parameters(), lr=1e-3
            ),
            'mixed_precision': torch.cuda.is_available()
        }
        
        # Communication with edge server
        self.edge_server_connection = None
        
        logging.info(f"Vehicle encoder client initialized: {vehicle_id}")
    
    def _initialize_encoder(self, config: Optional[Dict]) -> nn.Module:
        """Initialize encoder based on FHDP vehicle capabilities"""
        if config is None:
            config = {}
        
        encoder_type = config.get('encoder_type', 'resnet18')
        
        if encoder_type == 'resnet18':
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                # Simplified ResNet-like structure
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 512)
            )
        else:
            # Default simple encoder
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 256)
            )
    
    def _get_encoder_output_dim(self) -> int:
        """Get the output dimension of the encoder"""
        # This should match the actual encoder output
        return 512  # Default for ResNet18-like encoder
    
    def _get_vehicle_resources(self) -> Dict[str, Any]:
        """Get vehicle resources using FHDP's native resource detection"""
        hardware_info = HardwareAdapter.get_hardware_info()
        
        return {
            'cpu_cores': hardware_info.get('cpu_count', 4),
            'memory_gb': hardware_info.get('memory_gb', 8),
            'gpu_available': hardware_info.get('gpu_available', False),
            'gpu_memory_gb': hardware_info.get('gpu_memory_gb', 0),
            'storage_gb': hardware_info.get('storage_gb', 100),
            'network_bandwidth': hardware_info.get('network_bandwidth', 100)
        }
    
    def connect_to_edge_server(self) -> bool:
        """Connect to edge server using FHDP's native communication"""
        try:
            # Use FHDP's native communication manager
            self.edge_server_connection = (
                self.fhdp_vehicle.communication_manager.establish_connection(
                    self.edge_server_url
                )
            )
            return True
        except Exception as e:
            logging.error(f"Failed to connect to edge server: {e}")
            return False
    
    def train_encoder_epoch(self, data_loader) -> Dict[str, float]:
        """Train encoder for one epoch using FHDP's training engine"""
        self.encoder.train()
        self.action_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            # Process batch using FHDP's training capabilities
            batch_loss = self._train_batch(batch_data)
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Update training state
        self.training_state['current_round'] += 1
        
        return {
            'loss': avg_loss,
            'batches_processed': num_batches,
            'vehicle_id': self.vehicle_id
        }
    
    def _train_batch(self, batch_data: Dict[str, Any]) -> float:
        """Train a single batch"""
        # Use FHDP's native training executor
        training_config = {
            'model_components': [self.encoder, self.action_head],
            'optimizers': [
                self.training_state['encoder_optimizer'],
                self.training_state['action_head_optimizer']
            ],
            'mixed_precision': self.training_state['mixed_precision']
        }
        
        # Execute training using FHDP's training engine
        result = self.training_executor.execute_training_step(
            data=batch_data,
            config=training_config
        )
        
        return result.get('loss', 0.0)
    
    def request_vlm_features(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Request VLM features from edge server"""
        if not self.edge_server_connection:
            if not self.connect_to_edge_server():
                return None
        
        try:
            # Use FHDP's native communication
            request_data = {
                'vehicle_id': self.vehicle_id,
                'images': images,
                'request_type': 'vlm_inference'
            }
            
            response = self.fhdp_vehicle.communication_manager.send_request(
                self.edge_server_connection, request_data
            )
            
            if response and 'vlm_features' in response:
                return response['vlm_features']
            
        except Exception as e:
            logging.error(f"Failed to request VLM features: {e}")
        
        return None
    
    def generate_model_update(self) -> ModelUpdate:
        """Generate model update for FHDP aggregation"""
        return ModelUpdate(
            vehicle_id=self.vehicle_id,
            model_state={
                'encoder_state': self.encoder.state_dict(),
                'action_head_state': self.action_head.state_dict()
            },
            optimizer_state={
                'encoder_optimizer': self.training_state['encoder_optimizer'].state_dict(),
                'action_head_optimizer': self.training_state['action_head_optimizer'].state_dict()
            },
            metadata={
                'round': self.training_state['current_round'],
                'vehicle_resources': self._get_vehicle_resources(),
                'training_metrics': self._get_training_metrics()
            }
        )
    
    def _get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics using FHDP's monitoring capabilities"""
        vehicle_monitor = self.fhdp_vehicle.vehicle_monitor
        hardware_metrics = self.hardware_adapter.get_performance_metrics()
        
        return {
            'cpu_usage': hardware_metrics.get('cpu_usage', 0),
            'memory_usage': hardware_metrics.get('memory_usage', 0),
            'gpu_usage': hardware_metrics.get('gpu_usage', 0),
            'training_time': vehicle_monitor.get_training_time(),
            'communication_overhead': vehicle_monitor.get_communication_overhead()
        }
    
    def apply_aggregated_model(self, aggregated_update: ModelUpdate) -> bool:
        """Apply aggregated model update from FHDP"""
        try:
            if 'model_state' in aggregated_update.metadata:
                model_state = aggregated_update.metadata['model_state']
                
                if 'encoder_state' in model_state:
                    self.encoder.load_state_dict(model_state['encoder_state'])
                
                if 'action_head_state' in model_state:
                    self.action_head.load_state_dict(model_state['action_head'])
                
                if 'optimizer_state' in model_state:
                    opt_state = model_state['optimizer_state']
                    
                    if 'encoder_optimizer' in opt_state:
                        self.training_state['encoder_optimizer'].load_state_dict(
                            opt_state['encoder_optimizer']
                        )
                    
                    if 'action_head_optimizer' in opt_state:
                        self.training_state['action_head_optimizer'].load_state_dict(
                            opt_state['action_head_optimizer']
                        )
            
            logging.info(f"Applied aggregated model update to {self.vehicle_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply aggregated model: {e}")
            return False
    
    def get_vehicle_status(self) -> Dict[str, Any]:
        """Get vehicle status using FHDP's native status reporting"""
        fhdp_status = self.fhdp_vehicle.get_vehicle_info()
        
        return {
            'vehicle_id': self.vehicle_id,
            'fhdp_status': fhdp_status,
            'encoder_training_state': self.training_state,
            'hardware_status': self.hardware_adapter.get_hardware_info(),
            'connection_status': 'connected' if self.edge_server_connection else 'disconnected'
        }
    
    def start_training_session(self) -> bool:
        """Start training session using FHDP's native session management"""
        return self.fhdp_vehicle.start_training_session()
    
    def stop_training_session(self) -> bool:
        """Stop training session using FHDP's native session management"""
        return self.fhdp_vehicle.stop_training_session()
    
    def cleanup(self):
        """Cleanup resources using FHDP's native cleanup"""
        if self.fhdp_vehicle:
            self.fhdp_vehicle.cleanup()
        
        if self.edge_server_connection:
            self.fhdp_vehicle.communication_manager.close_connection(
                self.edge_server_connection
            )