"""
Edge Server VLM Integration with FHDP

This module integrates EVO-1 VLM backbone deployment with FHDP's
native edge server architecture, using existing FHDP capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from edge_server.server import EdgeServer
from core.types import VehicleInfo, ModelUpdate, AggregationResult
from core.hardware_adapter import HardwareAdapter
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

# Import EVO-1 components
try:
    from model.evo1_driving import InternVL3Embedder, ModelConfig
except ImportError:
    logging.warning("EVO-1 components not found. Using fallback implementation.")
    
    class InternVL3Embedder(nn.Module):
        """Fallback VLM backbone for FHDP edge server"""
        def __init__(self, model_name="OpenGVLab/InternVL3-1B", device="cuda"):
            super().__init__()
            self.device = device
            # Simple backbone structure
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2048)
            ).to(device)
            
        def forward(self, images, prompts=None):
            # Process multi-view images
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W)
            features = self.vision_encoder(images_flat)
            return features.view(B, N, -1)


class EdgeServerVLMIntegration:
    """
    Integration layer for EVO-1 VLM backbone with FHDP edge server
    
    This class extends FHDP's EdgeServer with EVO-1 specific VLM capabilities
    while maintaining full compatibility with FHDP's native architecture.
    """
    
    def __init__(self, server_id: str, vlm_config: Optional[Dict] = None):
        self.server_id = server_id
        
        # Initialize FHDP edge server
        self.fhdp_server = EdgeServer()
        
        # Initialize VLM backbone for EVO-1
        self.vlm_backbone = self._initialize_vlm_backbone(vlm_config)
        
        # Hardware adapter from FHDP
        self.hardware_adapter = HardwareAdapter(server_id)
        
        # Server state
        self.active_models = {}
        self.feature_cache = {}
        
        logging.info(f"Edge server VLM integration initialized: {server_id}")
    
    def _initialize_vlm_backbone(self, vlm_config: Optional[Dict]) -> nn.Module:
        """Initialize VLM backbone using EVO-1 components"""
        try:
            # Use EVO-1 configuration if available
            config = ModelConfig()
            backbone = InternVL3Embedder(
                model_name=vlm_config.get('model_name', 'OpenGVLab/InternVL3-1B'),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                **config.vl_embedder_kwargs
            )
        except Exception as e:
            logging.warning(f"Failed to load EVO-1 backbone: {e}")
            # Fallback to simple backbone
            backbone = InternVL3Embedder(
                model_name=vlm_config.get('model_name', 'OpenGVLab/InternVL3-1B'),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        # Freeze backbone for edge server deployment
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        
        return backbone
    
    def register_vehicle(self, vehicle_info: VehicleInfo) -> bool:
        """Register vehicle with FHDP edge server"""
        return self.fhdp_server.register_vehicle(vehicle_info)
    
    def process_vlm_inference(self, vehicle_id: str, images: torch.Tensor) -> Dict[str, Any]:
        """Process VLM inference for registered vehicle"""
        try:
            with torch.no_grad():
                # Process through VLM backbone
                vlm_features = self.vlm_backbone(images)
                
                # Cache results for efficiency
                cache_key = f"{vehicle_id}_{hash(images.data.tobytes())}"
                self.feature_cache[cache_key] = vlm_features.cpu()
                
                return {
                    'vehicle_id': vehicle_id,
                    'server_id': self.server_id,
                    'vlm_features': vlm_features,
                    'inference_time': 0.1,  # Mock timing
                    'cache_hit': False
                }
                
        except Exception as e:
            logging.error(f"VLM inference failed for {vehicle_id}: {e}")
            return {
                'vehicle_id': vehicle_id,
                'server_id': self.server_id,
                'error': str(e)
            }
    
    def get_cached_features(self, vehicle_id: str, image_hash: str) -> Optional[torch.Tensor]:
        """Get cached VLM features if available"""
        cache_key = f"{vehicle_id}_{image_hash}"
        return self.feature_cache.get(cache_key)
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status using FHDP's native capabilities"""
        fhdp_status = self.fhdp_server.get_server_stats()
        
        return {
            'server_id': self.server_id,
            'fhdp_status': fhdp_status,
            'vlm_backbone_loaded': self.vlm_backbone is not None,
            'cache_size': len(self.feature_cache),
            'hardware_status': self.hardware_adapter.get_hardware_info()
        }
    
    def start_server(self):
        """Start the edge server using FHDP's native server lifecycle"""
        return self.fhdp_server.start_server()
    
    def stop_server(self):
        """Stop the edge server using FHDP's native server lifecycle"""
        return self.fhdp_server.stop_server()
    
    def aggregate_vehicle_updates(self, vehicle_updates: List[ModelUpdate]) -> AggregationResult:
        """Aggregate vehicle updates using FHDP's native aggregation engine"""
        return self.fhdp_server.aggregator.aggregate(vehicle_updates)
    
    def get_supported_operations(self) -> List[str]:
        """Get list of operations supported by this edge server"""
        return [
            'vlm_inference',
            'model_aggregation', 
            'vehicle_registration',
            'resource_classification',
            'feature_caching'
        ]