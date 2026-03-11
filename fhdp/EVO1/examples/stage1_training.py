#!/usr/bin/env python3
"""
Stage 1 Training: Action Expert Alignment

This script executes only Stage 1 of the EVO-1 two-stage training strategy.
It freezes the vision-language backbone and trains only the action expert 
and integration modules.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add FHDP to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from EVO1.utils.config import EVO1DrivingConfig
from EVO1.training.stage_trainer import SeparatedStageTrainer


def create_stage1_config():
    """Create configuration optimized for Stage 1 training"""
    
    config = EVO1DrivingConfig()
    
    # Stage 1 specific settings
    config = override_for_stage1(config)
    
    # Default output configuration
    config.experiment_name = "evo1_stage1_action_expert"
    config.output_dir = "./outputs/evo1_stage1"
    
    return config


def override_for_stage1(config: EVO1DrivingConfig) -> EVO1DrivingConfig:
    """Apply Stage 1 specific settings to configuration"""
    # Stage 1 specific settings
    config.training.use_stage_training = False  # We'll handle stages manually
    config.training.stage1_rounds = 50  # Will be overridden by CLI if provided
    config.training.stage2_rounds = 0  # Not used in Stage 1
    config.training.aggregation_rounds = 50  # Will be overridden by CLI if provided
    
    # Learning rate for action expert training
    config.training.stage1_lr = 1e-4
    
    # Training parameters (only set if not already configured)
    if config.training.num_clients == 4:  # Default value, override
        config.training.num_clients = 4  # Will be overridden by CLI if provided
    if config.training.client_fraction == 0.3:  # Default value, override
        config.training.client_fraction = 0.75  # Use more clients per round
    config.training.local_epochs = 3  # More local epochs for Stage 1
    config.training.batch_size = 4  # Will be overridden by CLI if provided
    config.training.save_frequency = 10  # Save more frequently
    
    return config


def main():
    """Main function for Stage 1 training"""
    
    parser = argparse.ArgumentParser(description="EVO-1 Stage 1 Training: Action Expert Alignment")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to configuration YAML file")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Resume from checkpoint path")
    parser.add_argument("--gpu", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of training rounds")
    parser.add_argument("--clients", type=int, default=None,
                       help="Number of federated clients (uses config file if not specified)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size per client (uses config file if not specified)")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVO-1 Stage 1 Training: Action Expert Alignment")
    print("=" * 80)
    print()
    
    # Create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = EVO1DrivingConfig.from_yaml(args.config)
        # Apply Stage 1 specific overrides
        config = override_for_stage1(config)
    else:
        print("Using default Stage 1 configuration")
        config = create_stage1_config()
    
    # Override with command line arguments (only if provided)
    config.training.stage1_rounds = args.rounds
    config.training.aggregation_rounds = args.rounds
    if args.clients is not None:
        config.training.num_clients = args.clients
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # Override output settings if provided
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Stage 1 Configuration:")
    print(f"  Rounds: {config.training.stage1_rounds}")
    print(f"  Clients: {config.training.num_clients}")
    print(f"  Client Fraction: {config.training.client_fraction}")
    print(f"  Local Epochs: {config.training.local_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.stage1_lr}")
    print(f"  Output Dir: {config.output_dir}")
    print()
    
    print("Stage 1 Training Details:")
    print("  ✓ Freeze vision-language backbone (vl_embedder)")
    print("  ✓ Train action expert (action_head)")
    print("  ✓ Train integration modules (state_encoder, control_head)")
    print("  ✓ Higher learning rate for fast alignment")
    print("  ✓ Focus on action expert convergence")
    print()
    
    # Initialize Stage 1 trainer
    print("Initializing Stage 1 trainer...")
    trainer = SeparatedStageTrainer(
        config=config,
        stage=1,
        resume_from_checkpoint=args.resume,
        device=args.gpu
    )
    print("Stage 1 trainer initialized successfully!")
    print()
    
    # Show model state
    total_params = sum(p.numel() for p in trainer.global_model.parameters())
    trainable_params = sum(p.numel() for p in trainer.global_model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable in Stage 1: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print()
    
    # Start training
    print("Starting Stage 1 training...")
    print("This will train only the action expert while keeping backbone frozen.")
    print("-" * 50)
    
    try:
        trainer.train()
        
        print("-" * 50)
        print("Stage 1 training completed successfully!")
        
        final_model_path = trainer.get_final_model_path()
        print(f"Final model saved to: {final_model_path}")
        print()
        print("Next Steps:")
        print("1. Use this model for inference with frozen backbone")
        print("2. Continue to Stage 2 training for full fine-tuning")
        print(f"3. Stage 2 command: python3 fhdp/EVO1/examples/stage2_training.py --resume {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nStage 1 training interrupted by user. Checkpoint saved.")
    except Exception as e:
        print(f"\nStage 1 training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()