#!/usr/bin/env python3
"""
Stage 2 Training: Full-scale Fine-Tuning

This script executes only Stage 2 of the EVO-1 two-stage training strategy.
It unfreezes all components for end-to-end fine-tuning.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add FHDP and EVO1 to path with EVO1 taking precedence
evo1_root = str(Path(__file__).parent.parent)
fhdp_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, evo1_root)
sys.path.append(fhdp_root)

from utils.config import EVO1DrivingConfig
from training.stage_trainer import SeparatedStageTrainer


def create_stage2_config():
    """Create configuration optimized for Stage 2 training"""
    
    config = EVO1DrivingConfig()
    
    # Stage 2 specific settings
    config.training.use_stage_training = False  # We'll handle stages manually
    config.training.stage1_rounds = 0  # Not used in Stage 2
    config.training.stage2_rounds = 50
    config.training.aggregation_rounds = 50  # Only Stage 2 rounds
    
    # Learning rate for fine-tuning (lower than Stage 1)
    config.training.stage2_lr = 5e-5
    
    # Training parameters (can be more aggressive in Stage 2)
    config.training.num_clients = 4
    config.training.client_fraction = 0.5  # Standard federated fraction
    config.training.local_epochs = 2  # Fewer epochs for Stage 2
    config.training.batch_size = 4  # Can increase if GPU memory allows
    config.training.save_frequency = 5  # Save more frequently
    
    # Output configuration
    config.experiment_name = "evo1_stage2_fine_tuning"
    config.output_dir = "./outputs/evo1_stage2"
    
    return config


def main():
    """Main function for Stage 2 training"""
    
    parser = argparse.ArgumentParser(description="EVO-1 Stage 2 Training: Full-scale Fine-Tuning")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from Stage 1 checkpoint path")
    parser.add_argument("--gpu", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of training rounds")
    parser.add_argument("--clients", type=int, default=4,
                       help="Number of federated clients")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per client")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVO-1 Stage 2 Training: Full-scale Fine-Tuning")
    print("=" * 80)
    print()
    
    # Create configuration
    config = create_stage2_config()
    
    # Override with command line arguments
    config.training.stage2_rounds = args.rounds
    config.training.aggregation_rounds = args.rounds
    config.training.num_clients = args.clients
    config.training.batch_size = args.batch_size
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Stage 2 Configuration:")
    print(f"  Rounds: {config.training.stage2_rounds}")
    print(f"  Clients: {config.training.num_clients}")
    print(f"  Client Fraction: {config.training.client_fraction}")
    print(f"  Local Epochs: {config.training.local_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.stage2_lr}")
    print(f"  Output Dir: {config.output_dir}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print()
    
    print("Stage 2 Training Details:")
    print("  ✓ Unfreeze all model components")
    print("  ✓ Train vision-language backbone (vl_embedder)")
    print("  ✓ Train action expert (action_head)")
    print("  ✓ Train integration modules (state_encoder, control_head)")
    print("  ✓ Lower learning rate for stable fine-tuning")
    print("  ✓ End-to-end optimization")
    print()
    
    # Initialize Stage 2 trainer
    print("Initializing Stage 2 trainer...")
    trainer = SeparatedStageTrainer(
        config=config,
        stage=2,
        resume_from_checkpoint=args.resume,
        device=args.gpu
    )
    print("Stage 2 trainer initialized successfully!")
    print()
    
    # Show model state
    total_params = sum(p.numel() for p in trainer.global_model.parameters())
    trainable_params = sum(p.numel() for p in trainer.global_model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable in Stage 2: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print()
    
    # Start training
    print("Starting Stage 2 training...")
    print("This will fine-tune all components end-to-end.")
    if args.resume:
        print("Resuming from Stage 1 checkpoint - model should be well-aligned.")
    print("-" * 50)
    
    try:
        trainer.train()
        
        print("-" * 50)
        print("Stage 2 training completed successfully!")
        
        final_model_path = trainer.get_final_model_path()
        print(f"Final model saved to: {final_model_path}")
        print()
        print("Training Complete!")
        print("The model is now fully fine-tuned and ready for deployment.")
        print()
        print("Performance Testing:")
        print(f"  - Inference: python3 fhdp/EVO1/examples/test_model.py --model {final_model_path}")
        print(f"  - Evaluation: python3 fhdp/EVO1/examples/evaluate_model.py --model {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nStage 2 training interrupted by user. Checkpoint saved.")
    except Exception as e:
        print(f"\nStage 2 training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()