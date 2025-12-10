"""
Core type definitions for FHDP system
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from torch import Tensor

class VehicleState(Enum):
    """Vehicle participation state"""
    IDLE = "idle"
    INDIVIDUAL = "individual"
    PIPELINE = "pipeline"
    AGGREGATING = "aggregating"
    DISCONNECTED = "disconnected"

class TrainingMode(Enum):
    """Training participation mode"""
    INDIVIDUAL = "individual"
    PIPELINE = "pipeline"

class CommunicationProtocol(Enum):
    """V2V communication protocols"""
    DSRC = "dsrc"
    C-V2X = "cv2x"
    WIFI_DIRECT = "wifi_direct"

class ResourceClass(Enum):
    """Vehicle resource classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class VehicleInfo:
    """Vehicle information and capabilities"""
    vehicle_id: str
    position: Tuple[float, float]  # (x, y) coordinates
    velocity: float  # m/s
    direction: float  # radians
    resources: Dict[str, Any]  # CPU, memory, battery, etc.
    state: VehicleState = VehicleState.IDLE
    last_seen: float = field(default_factory=time.time)
    training_capability: float = 1.0  # 0.0-1.0 capability score
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    local_data_size: int = 1000
    model_size: int = 0  # bytes
    communication_budget: int = 1024 * 1024  # 1MB
    
@dataclass
class PipelineTemplate:
    """Pipeline template for efficient matching"""
    template_id: str
    resource_requirements: List[ResourceClass]
    expected_duration: float  # seconds
    communication_pattern: List[Tuple[int, int]]  # (from_stage, to_stage)
    training_config: TrainingConfig
    model_fragment_size: int = 0  # bytes per fragment

@dataclass 
class Pipeline:
    """Active pipeline instance"""
    pipeline_id: str
    template_id: str
    vehicles: List[str]  # vehicle IDs in order
    stages: List[str]  # stage identifiers
    current_stage: int = 0
    start_time: float = field(default_factory=time.time)
    expected_completion: float = 0.0
    intermediate_results: List[Tensor] = field(default_factory=list)
    communication_overhead: int = 0  # bytes
    
@dataclass
class ModelUpdate:
    """Model update from vehicle or pipeline"""
    source_id: str  # vehicle ID or pipeline ID
    update_data: Union[Tensor, Dict[str, Tensor]]
    metadata: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    training_mode: TrainingMode = TrainingMode.INDIVIDUAL
    fidelity_score: float = 1.0

@dataclass
class AggregationResult:
    """Result of federated aggregation"""
    global_model: Union[Tensor, Dict[str, Tensor]]
    participating_sources: List[str]
    aggregation_weight: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class MobilityPrediction:
    """Mobility prediction result"""
    vehicle_id: str
    predicted_position: Tuple[float, float]
    predicted_time: float  # seconds in future
    confidence: float  # 0.0-1.0
    transition_probabilities: Dict[str, float]

@dataclass
class CommunicationBundle:
    """Communication bundle for efficiency"""
    messages: List[Dict[str, Any]]
    target_ids: List[str]
    protocol: CommunicationProtocol
    compression_ratio: float = 1.0
    bundle_size: int = 0  # bytes

@dataclass
class ResourceMetrics:
    """Vehicle resource utilization metrics"""
    cpu_usage: float  # 0.0-1.0
    memory_usage: float  # 0.0-1.0
    battery_level: float  # 0.0-1.0
    network_quality: float  # 0.0-1.0
    thermal_state: float  # 0.0-1.0
    
@dataclass
class FairnessMetrics:
    """Participation fairness metrics"""
    vehicle_id: str
    participation_count: int
    last_participation: float
    contribution_score: float
    priority_weight: float

@dataclass
class ErrorPropagation:
    """Lazy error propagation data"""
    error_signals: Dict[str, Tensor]
    accumulation_threshold: float
    propagation_targets: List[str]
    propagation_count: int = 0