#!/usr/bin/env python3
"""
Main training script for federated EVO-1 autonomous driving

This script provides the complete training pipeline for federated learning
with EVO-1 model on nuScenes dataset integrated with FHDP architecture.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.distributed as dist
import wandb

# Import EVO-1 FHDP components
from EVO1.utils.config import EVO1DrivingConfig, DEFAULT_CONFIGS
from EVO1.training.federated_trainer import FederatedEVO1Trainer
from EVO1.evaluation.driving_metrics import DrivingMetricsEvaluator
from EVO1.data.nuscenes_loader import create_dataloader


def setup_logging(log_level: str, log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Federated EVO-1 Autonomous Driving Training"
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--preset', type=str, choices=['simulation', 'jetson_realtime', 'multi_vehicle'],
        help='Use preset configuration'
    )
    
    # Data
    parser.add_argument(
        '--data_root', type=str, default='/data/nuscenes',
        help='Root directory for nuScenes dataset'
    )
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    
    # Training
    parser.add_argument(
        '--num_rounds', type=int, default=100,
        help='Number of federated learning rounds'
    )
    parser.add_argument(
        '--num_clients', type=int, default=10,
        help='Number of federated clients'
    )
    parser.add_argument(
        '--client_fraction', type=float, default=0.3,
        help='Fraction of clients to select per round'
    )
    parser.add_argument(
        '--local_epochs', type=int, default=2,
        help='Number of local epochs per client'
    )
    
    # Model
    parser.add_argument(
        '--model_name', type=str, default='OpenGVLab/InternVL3-1B',
        help='Vision-language model name'
    )
    parser.add_argument(
        '--max_waypoints', type=int, default=20,
        help='Maximum number of waypoints to predict'
    )
    
    # Output
    parser.add_argument(
        '--output_dir', type=str, default='./outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment_name', type=str, default='evo1_driving_federated',
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--resume_from_checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    
    # FHDP Integration
    parser.add_argument(
        '--use_fhdp', action='store_true',
        help='Enable FHDP integration'
    )
    parser.add_argument(
        '--use_pipeline_parallel', action='store_true',
        help='Enable pipeline parallel training'
    )
    parser.add_argument(
        '--num_pipeline_stages', type=int, default=4,
        help='Number of pipeline parallel stages'
    )
    
    # Evaluation
    parser.add_argument(
        '--eval_frequency', type=int, default=10,
        help='Frequency of evaluation (rounds)'
    )
    parser.add_argument(
        '--num_eval_episodes', type=int, default=100,
        help='Number of evaluation episodes'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--enable_wandb', action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> EVO1DrivingConfig:
    """Create configuration from command line arguments"""
    
    # Load base configuration
    if args.config and os.path.exists(args.config):
        config = EVO1DrivingConfig.from_yaml(args.config)
    elif args.preset:
        config = DEFAULT_CONFIGS[args.preset]
    else:
        config = EVO1DrivingConfig()
    
    # Override with command line arguments
    if args.data_root:
        config.data.data_root = args.data_root
    
    if args.num_rounds:
        config.training.aggregation_rounds = args.num_rounds
    
    if args.num_clients:
        config.training.num_clients = args.num_clients
    
    if args.client_fraction:
        config.training.client_fraction = args.client_fraction
    
    if args.local_epochs:
        config.training.local_epochs = args.local_epochs
    
    if args.model_name:
        config.model.vision_model_name = args.model_name
    
    if args.max_waypoints:
        config.model.max_waypoints = args.max_waypoints
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.use_pipeline_parallel:
        config.fhdp.use_pipeline_parallel = args.use_pipeline_parallel
        config.fhdp.num_pipeline_stages = args.num_pipeline_stages
    
    if args.eval_frequency:
        config.evaluation.eval_frequency = args.eval_frequency
    
    if args.num_eval_episodes:
        config.evaluation.num_eval_episodes = args.num_eval_episodes
    
    if args.seed:
        config.seed = args.seed
    
    if args.device:
        config.device = args.device
    
    if args.log_level:
        config.log_level = args.log_level
    
    return config


def validate_configuration(config: EVO1DrivingConfig):
    """Validate configuration parameters"""
    errors = []
    
    # Data validation
    if not os.path.exists(config.data.data_root):
        errors.append(f"Data root does not exist: {config.data.data_root}")
    
    # Model validation
    if config.model.action_dim < 1:
        errors.append("Action dimension must be positive")
    
    if config.model.max_waypoints < 1:
        errors.append("Max waypoints must be positive")
    
    # Training validation
    if config.training.num_clients < 1:
        errors.append("Number of clients must be positive")
    
    if config.training.client_fraction <= 0 or config.training.client_fraction > 1:
        errors.append("Client fraction must be between 0 and 1")
    
    if config.training.aggregation_rounds < 1:
        errors.append("Aggregation rounds must be positive")
    
    # Device validation
    if config.device == 'cuda' and not torch.cuda.is_available():
        errors.append("CUDA requested but not available")
        config.device = 'cpu'
    
    if errors:
        for error in errors:
            logging.error(error)
        raise ValueError("Configuration validation failed")
    
    logging.info("Configuration validation passed")


def setup_reproducibility(seed: int):
    """Setup reproducibility settings"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Set random seed to {seed}")


def setup_wandb(config: EVO1DrivingConfig, enable: bool):
    """Setup Weights & Biases logging"""
    if not enable:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        return
    
    if os.getenv('WANDB_API_KEY') is None:
        logging.warning("WANDB_API_KEY not found. Wandb logging disabled.")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        return
    
    wandb.init(
        project="evo1-federated-driving",
        name=config.experiment_name,
        config=config.__dict__
    )
    logging.info("Wandb logging initialized")


def save_experiment_summary(config: EVO1DrivingConfig, start_time: float, end_time: float):
    """Save experiment summary"""
    summary = {
        'experiment_name': config.experiment_name,
        'config': config.__dict__,
        'start_time': start_time,
        'end_time': end_time,
        'duration_seconds': end_time - start_time,
        'success': True
    }
    
    summary_path = os.path.join(config.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Experiment summary saved to {summary_path}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Setup logging
    log_dir = os.path.join(config.output_dir, 'logs')
    setup_logging(config.log_level, log_dir)
    
    # Validate configuration
    validate_configuration(config)
    
    # Setup reproducibility
    setup_reproducibility(config.seed)
    
    # Setup wandb
    setup_wandb(config, args.enable_wandb)
    
    # Log experiment info
    logging.info("Starting Federated EVO-1 Autonomous Driving Training")
    logging.info(f"Experiment: {config.experiment_name}")
    logging.info(f"Device: {config.device}")
    logging.info(f"Output directory: {config.output_dir}")
    logging.info(f"Number of clients: {config.training.num_clients}")
    logging.info(f"Aggregation rounds: {config.training.aggregation_rounds}")
    
    # Setup output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'metrics'), exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, 'config.yaml')
    config.to_yaml(config_path)
    logging.info(f"Configuration saved to {config_path}")
    
    # Setup FHDP system if enabled
    fhdp_system = None
    if args.use_fhdp:
        try:
            # Import FHDP components
            sys.path.append(os.path.join(project_root, 'core'))
            sys.path.append(os.path.join(project_root, 'edge_server'))
            sys.path.append(os.path.join(project_root, 'vehicle_layer'))
            
            from core.fhdp_system import FHDPSystem
            from edge_server.server import EdgeServer
            
            # Initialize FHDP system
            fhdp_system = FHDPSystem()
            logging.info("FHDP system initialized")
            
        except ImportError as e:
            logging.warning(f"Failed to initialize FHDP system: {e}")
            logging.warning("Continuing without FHDP integration")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Initialize federated trainer
        trainer = FederatedEVO1Trainer(
            config=config,
            fhdp_system=fhdp_system,
            device=config.device
        )
        
        # Start training
        trainer.train()
        
        # Final evaluation
        logging.info("Starting final evaluation...")
        test_loader = create_dataloader(
            config=config.data,
            model_config=config.model,
            split='test',
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        if test_loader:
            final_metrics = trainer.evaluate_global_model(test_loader)
            logging.info("Final evaluation metrics:")
            for metric, value in final_metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
            
            # Save final metrics
            metrics_path = os.path.join(config.output_dir, 'final_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=4)
        
        logging.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    finally:
        # Record end time
        end_time = time.time()
        
        # Save experiment summary
        save_experiment_summary(config, start_time, end_time)
        
        # Finish wandb
        if args.enable_wandb and wandb.run is not None:
            wandb.finish()
        
        logging.info(f"Training duration: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()