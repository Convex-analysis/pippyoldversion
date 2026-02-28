"""
EVO-1 Autonomous Driving Integration for FHDP Framework

This module provides the complete training pipeline for EVO-1 model
integration into the FHDP federated learning system, specifically
adapted for autonomous driving tasks using nuScenes dataset.
"""

__version__ = "1.0.0"
__author__ = "FHDP-EVO1 Integration Team"

from .model.evo1_driving import EVO1Driving, FederatedEVO1Driving
from .data.nuscenes_loader import NuScenesDrivingLoader
from .training.stage_trainer import SeparatedStageTrainer
from .evaluation.driving_metrics import DrivingMetricsEvaluator
from .utils.config import EVO1DrivingConfig

__all__ = [
    "EVO1Driving",
    "FederatedEVO1Driving",
    "NuScenesDrivingLoader", 
    "SeparatedStageTrainer",
    "DrivingMetricsEvaluator",
    "EVO1DrivingConfig"
]