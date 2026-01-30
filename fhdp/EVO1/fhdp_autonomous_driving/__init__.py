"""
EVO-1 FHDP Autonomous Driving Integration

This module provides the core integration between EVO-1 autonomous driving model
and FHDP (Federated Hierarchical Dynamic Pipeline) system.

Key Features:
- Vision-based autonomous driving with EVO-1
- Federated learning coordination through FHDP
- Dynamic resource management for distributed training
- Real-time vehicle management and coordination
- Hybrid participation (individual + pipeline training)
"""

from .evo1_trainer import FHDAutonomousDrivingTrainer
from .vehicle_manager import FHDAutonomousVehicleManager
from .coordination import FHDAutonomousCoordination
from .deployment import FHDPDeploymentConfig

__version__ = "1.0.0"
__author__ = "EVO-1 FHDP Team"

__all__ = [
    'FHDAutonomousDrivingTrainer',
    'FHDAutonomousVehicleManager', 
    'FHDAutonomousCoordination',
    'FHDPDeploymentConfig'
]