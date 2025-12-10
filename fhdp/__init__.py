"""
FHDP (Federated Highway-based Distributed Pipeline) System Architecture

A two-level federated learning system designed for vehicular environments with:
- Decentralized two-level decision hierarchy
- Transient pipeline constructs  
- Persistent edge server state
- Resource-aware participation
- Communication-efficient training

Components:
- Edge Server: Mobility prediction, template generation, async aggregation
- Vehicle Layer: Neighbor discovery, pipeline formation, training execution
- Core System: Hybrid participation, fairness mechanisms, lazy error propagation
"""

from .edge_server.server import EdgeServer
from .vehicle_layer.vehicle import Vehicle
from .core.fhdp_system import FHDPSystem
from .core.types import *

__version__ = "1.0.0"
__all__ = [
    "EdgeServer",
    "Vehicle", 
    "FHDPSystem"
]