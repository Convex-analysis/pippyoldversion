"""
Configuration system for EVO-1 autonomous driving integration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """EVO-1 model configuration for driving tasks"""
    # Vision encoder config
    vision_model_name: str = "OpenGVLab/InternVL3-1B"
    vision_layers: int = 14  # Reduced for efficiency
    image_size: int = 448
    num_views: int = 3  # Front, left, right cameras
    
    # Action head config
    action_dim: int = 60  # 20 * 3 (horizon * per_action_dim)
    per_action_dim: int = 3  # [steering, throttle, brake]
    horizon: int = 20  # Time horizon for action sequence
    action_hidden_dim: int = 512
    action_num_layers: int = 6
    flow_matching_steps: int = 50
    
    # Driving specific
    max_waypoints: int = 20
    trajectory_horizon: float = 3.0  # seconds
    control_frequency: float = 10.0  # Hz
    max_speed: float = 30.0  # m/s
    max_steering: float = 0.6 # radians


@dataclass  
class DataConfig:
    """NuScenes dataset configuration"""
    data_root: str = "/data/nuscenes"
    version: str = "v1.0-trainval"
    split: str = "train"  # train/val/test
    
    # Image preprocessing
    image_size: Tuple[int, int] = (448, 448)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Sequence settings
    sequence_length: int = 10  # frames
    sequence_stride: int = 5   # frames between sequences
    
    # Driving task specific
    max_speed: float = 30.0  # m/s
    min_speed: float = 0.0    # m/s
    max_steering: float = 0.6 # radians
    max_acceleration: float = 3.0 # m/s^2
    max_waypoints: int = 20   # Number of future waypoints to predict

    # Camera and scene options (backward compatible)
    num_views: int = 3                # 3 (front only) or 6 (surround)
    use_6_cameras: bool = False       # True to load all 6 NuScenes cameras
    filter_by_keywords: bool = False  # Filter scenes by driving-related keywords


@dataclass
class TrainingConfig:
    """Training configuration for federated learning"""
    # Two-stage training strategy
    use_stage_training: bool = True
    stage1_rounds: int = 50
    stage2_rounds: int = 50
    stage1_lr: float = 1e-4
    stage2_lr: float = 5e-5
    
    # Federated learning
    federated_learning: bool = True
    num_clients: int = 10
    local_epochs: int = 2
    aggregation_rounds: int = 100
    client_fraction: float = 0.3
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # LR scheduling
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # Optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Resource constraints
    max_memory_gb: float = 8.0


@dataclass
class FHDPConfig:
    """FHDP integration configuration"""
    # Pipeline parallelism
    use_pipeline_parallel: bool = True
    num_pipeline_stages: int = 4
    pipeline_chunks: int = 32
    
    # Communication
    communication_protocol: str = "websocket"
    aggregation_protocol: str = "federated_avg"
    compression_enabled: bool = True
    
    # Resource constraints
    max_memory_gb: float = 8.0
    max_latency_ms: float = 100.0
    target_fps: float = 10.0
    
    # Fairness
    fairness_enabled: bool = True
    min_participation_interval: float = 5.0


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    # Core metrics
    evaluate_l2_error: bool = True
    evaluate_collision_rate: bool = True
    evaluate_offroad_rate: bool = True
    evaluate_traffic_violations: bool = True
    
    # Trajectory metrics
    evaluate_ade: bool = True  # Average Displacement Error
    evaluate_fde: bool = True  # Final Displacement Error
    evaluate_miss_rate: bool = True
    
    # Safety metrics
    safety_margin_m: float = 2.0
    time_horizon_s: float = 3.0
    
    # Evaluation settings
    eval_frequency: int = 10  # every N training steps
    num_eval_episodes: int = 100
    save_predictions: bool = True


@dataclass
class EVO1DrivingConfig:
    """Main configuration class combining all sub-configs"""
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fhdp: FHDPConfig = field(default_factory=FHDPConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    experiment_name: str = "evo1_driving_federated"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda"
    log_level: str = "INFO"
    
    # Checkpointing
    save_frequency: int = 1000
    resume_from_checkpoint: Optional[str] = None
    max_checkpoints_to_keep: int = 5
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "EVO1DrivingConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        fhdp_config = FHDPConfig(**config_dict.get('fhdp', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            fhdp=fhdp_config,
            evaluation=evaluation_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'data', 'training', 'fhdp', 'evaluation']}
        )
    
    def to_yaml(self, save_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'fhdp': self.fhdp.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
            'log_level': self.log_level,
            'save_frequency': self.save_frequency,
            'max_checkpoints_to_keep': self.max_checkpoints_to_keep
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Default configurations for different scenarios
DEFAULT_CONFIGS = {
    'simulation': EVO1DrivingConfig(
        training=TrainingConfig(
            batch_size=16,
            local_epochs=1,
            max_memory_gb=16.0
        ),
        fhdp=FHDPConfig(
            use_pipeline_parallel=False,
            max_latency_ms=200.0
        )
    ),
    
    'jetson_realtime': EVO1DrivingConfig(
        model=ModelConfig(
            vision_layers=8,  # Reduced for real-time
            action_hidden_dim=256
        ),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=8,
            max_memory_gb=6.0
        ),
        fhdp=FHDPConfig(
            max_latency_ms=50.0,
            target_fps=20.0
        )
    ),
    
    'multi_vehicle': EVO1DrivingConfig(
        training=TrainingConfig(
            num_clients=20,
            client_fraction=0.5
        ),
        fhdp=FHDPConfig(
            use_pipeline_parallel=True,
            num_pipeline_stages=8,
            communication_protocol="cv2x"
        )
    )
}