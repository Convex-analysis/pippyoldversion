"""
Pipeline Parallel Federated Learning for EVO-1 with FHDP Integration

This module extends EVO-1 training with FHDP's native pipeline parallel capabilities:
- Uses FHDP's existing edge server architecture
- Leverages FHDP's vehicle coordination system  
- Integrates with FHDP's asynchronous aggregation
- Maintains compatibility with FHDP's resource management
"""

from .edge_server.edge_integration import EdgeServerVLMIntegration
from .vehicle_client.vehicle_encoder import VehicleEncoderClient
from .coordinator.fhdp_coordinator import FHDPipelineCoordinator
from .adapter.hardware_integration import HardwareResourceAdapter

__version__ = "1.0.0"
__author__ = "EVO-1 FHDP Integration Team"

__all__ = [
    'EdgeServerVLMIntegration',
    'VehicleEncoderClient', 
    'FHDPipelineCoordinator',
    'HardwareResourceAdapter'
]