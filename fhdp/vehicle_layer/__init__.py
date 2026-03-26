"""
Vehicle Layer Components for FHDP System

Provides:
- Neighbor discovery and V2V communication protocols
- Pipeline formation with greedy selection
- Training execution engine with communication optimization
- Resource monitoring and participation tracking
- Training client for pipeline training
"""

from .communication import V2VCommunicationManager, ProtocolManager, NeighborDiscovery, MessageRouter
from .pipeline_formation import PipelineFormation, GreedySelector, PipelineCandidate
from .training_engine import TrainingExecutor, CommunicationOptimizer, LazyErrorPropagation
from .monitor import VehicleMonitor, ResourceMonitor, ParticipationTracker
from .vehicle import Vehicle
from .training_client import TrainingClient

__all__ = [
    "V2VCommunicationManager",
    "PipelineFormation", 
    "TrainingExecutor",
    "VehicleMonitor",
    "Vehicle",
    "TrainingClient"
]