#!/usr/bin/env python3
"""
Evaluation script for EVO-1 autonomous driving model

This script provides comprehensive evaluation of the trained model
using the driving metrics evaluator on the validation dataset.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

import torch
import numpy as np

# Import EVO-1 components
from EVO1.utils.config import EVO1DrivingConfig
from EVO1.model.evo1_driving import EVO1Driving
from EVO1.data.nuscenes_loader import create_dataloader
from EVO1.evaluation.driving_metrics import DrivingMetricsEvaluator


def setup_logging(log_level: str, log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'evaluation.log')
    
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
        description="Evaluate EVO-1 Autonomous Driving Model"
    )
    
    # Model
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config_path', type=str, required=True,
        help='Path to configuration file'
    )
    
    # Data
    parser.add_argument(
        '--split', type=str, default='val', choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--data_root', type=str, default='/home/xta/fhdp/EVO1/data/nuscenes',
        help='Root directory for nuScenes dataset'
    )
    
    # Evaluation
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--eval_steps', type=int, default=None,
        help='Number of steps to evaluate (None for full dataset)'
    )
    
    # Output
    parser.add_argument(
        '--output_dir', type=str, default='./evaluation_outputs',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--experiment_name', type=str, default='evaluation_run',
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization of evaluation results'
    )
    
    # Misc
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def load_model(model_path: str, config: EVO1DrivingConfig, device: str) -> EVO1Driving:
    """Load the trained model"""
    # Initialize model
    model = EVO1Driving(
        config=config.model,
        training_config=config.training,
        device=device
    ).to(device)
    
    # Load checkpoint
    logging.info(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model: EVO1Driving, dataloader, evaluator: DrivingMetricsEvaluator, 
                   device: str, max_steps: int = None) -> dict:
    """Evaluate model on dataset"""
    logging.info("Starting evaluation...")
    
    total_batches = len(dataloader)
    if max_steps:
        total_batches = min(max_steps, total_batches)
    
    start_time = time.time()
    evaluation_times = []
    
    for batch_idx, batch in enumerate(dataloader):
        if max_steps and batch_idx >= max_steps:
            break
        
        batch_start = time.time()
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Run inference
        with torch.no_grad():
            output = model(
                images=batch['images'],
                image_mask=batch['image_mask'],
                state=batch['state'],
                instructions=batch['instructions'],
                mode="inference"
            )
        
        # Prepare model output dictionary
        model_output = {
            'waypoints': output.waypoints,
            'controls': output.controls
        }
        
        # Evaluate batch
        batch_results = evaluator.evaluate_batch(batch, model_output)
        
        # Log progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
            logging.info(f"Batch {batch_idx+1}/{total_batches} - ETA: {eta:.2f}s")
        
        evaluation_times.append(time.time() - batch_start)
    
    total_time = time.time() - start_time
    avg_inference_time = np.mean(evaluation_times)
    
    logging.info(f"Evaluation completed in {total_time:.2f} seconds")
    logging.info(f"Average inference time: {avg_inference_time:.4f} seconds per batch")
    
    return {
        'total_time': total_time,
        'avg_inference_time': avg_inference_time,
        'batches_evaluated': total_batches
    }


def save_evaluation_results(results: dict, evaluator: DrivingMetricsEvaluator, 
                            config: EVO1DrivingConfig, output_dir: str):
    """Save evaluation results"""
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    
    # Prepare results data
    evaluation_data = {
        'experiment_name': config.experiment_name,
        'model_path': config.model_path,
        'dataset_split': config.split,
        'evaluation_time': results['total_time'],
        'avg_inference_time': results['avg_inference_time'],
        'batches_evaluated': results['batches_evaluated'],
        'config': config.__dict__,
        'trajectory_metrics': [
            {
                'ade': eval.ade,
                'fde': eval.fde,
                'miss_rate': eval.miss_rate,
                'minade': eval.minade,
                'maxade': eval.maxade,
                'trajectory_length_error': eval.trajectory_length_error,
                'heading_error': eval.heading_error
            } for eval in evaluator.trajectory_evaluations
        ],
        'control_metrics': [
            {
                'steering_mae': eval.steering_mae,
                'steering_rmse': eval.steering_rmse,
                'throttle_mae': eval.throttle_mae,
                'throttle_rmse': eval.throttle_rmse,
                'brake_mae': eval.brake_mae,
                'brake_rmse': eval.brake_rmse,
                'control_smoothness': eval.control_smoothness,
                'control_frequency_error': eval.control_frequency_error
            } for eval in evaluator.control_evaluations
        ],
        'safety_metrics': [
            {
                'collision_rate': eval.collision_rate,
                'offroad_rate': eval.offroad_rate,
                'traffic_violation_rate': eval.traffic_violation_rate,
                'safety_margin_violations': eval.safety_margin_violations,
                'emergency_brake_rate': eval.emergency_brake_rate,
                'comfort_score': eval.comfort_score
            } for eval in evaluator.safety_evaluations
        ],
        'efficiency_metrics': [
            {
                'trip_time_error': eval.trip_time_error,
                'fuel_efficiency_score': eval.fuel_efficiency_score,
                'average_speed_error': eval.average_speed_error,
                'path_efficiency': eval.path_efficiency,
                'stop_count_error': eval.stop_count_error
            } for eval in evaluator.efficiency_evaluations
        ]
    }
    
    # Calculate summary metrics
    if evaluator.trajectory_evaluations:
        avg_ade = np.mean([eval.ade for eval in evaluator.trajectory_evaluations])
        avg_fde = np.mean([eval.fde for eval in evaluator.trajectory_evaluations])
        evaluation_data['summary_metrics'] = {
            'avg_ade': avg_ade,
            'avg_fde': avg_fde,
            'overall_score': evaluator.compute_overall_score({
                'trajectory': evaluator.trajectory_evaluations[0],
                'control': evaluator.control_evaluations[0] if evaluator.control_evaluations else None,
                'safety': evaluator.safety_evaluations[0] if evaluator.safety_evaluations else None,
                'efficiency': evaluator.efficiency_evaluations[0] if evaluator.efficiency_evaluations else None
            })
        }
    
    # Save to file
    with open(results_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2, default=str)
    
    logging.info(f"Evaluation results saved to: {results_path}")
    
    return results_path


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = EVO1DrivingConfig.from_yaml(args.config_path)
    
    # Override configuration with command line arguments
    config.model_path = args.model_path
    config.split = args.split
    config.data.data_root = args.data_root
    config.experiment_name = args.experiment_name
    config.device = args.device
    config.log_level = args.log_level
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(output_dir, 'logs')
    setup_logging(config.log_level, log_dir)
    
    # Log experiment info
    logging.info("Starting EVO-1 Model Evaluation")
    logging.info(f"Experiment: {config.experiment_name}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Dataset split: {args.split}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(args.model_path, config, args.device)
    
    # Create dataloader
    logging.info("Creating validation dataloader...")
    dataloader = create_dataloader(
        config=config.data,
        model_config=config.model,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info(f"Dataloader created with {len(dataloader)} batches")
    
    # Initialize evaluator
    evaluator = DrivingMetricsEvaluator(
        config=config.evaluation,
        model_config=config.model,
        output_dir=output_dir
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=args.device,
        max_steps=args.eval_steps
    )
    
    # Save results
    results_path = save_evaluation_results(
        evaluation_results,
        evaluator,
        config,
        output_dir
    )
    
    # Visualize results if enabled
    if args.visualize:
        viz_path = os.path.join(output_dir, 'evaluation_visualization.png')
        evaluator.visualize_results(viz_path)
    
    # Print summary
    logging.info("\n=== Evaluation Summary ===")
    logging.info(f"Total evaluation time: {evaluation_results['total_time']:.2f} seconds")
    logging.info(f"Average inference time: {evaluation_results['avg_inference_time']:.4f} seconds per batch")
    logging.info(f"Batches evaluated: {evaluation_results['batches_evaluated']}")
    
    if hasattr(evaluator, 'trajectory_evaluations') and evaluator.trajectory_evaluations:
        avg_ade = np.mean([eval.ade for eval in evaluator.trajectory_evaluations])
        avg_fde = np.mean([eval.fde for eval in evaluator.trajectory_evaluations])
        logging.info(f"\n=== Key Metrics ===")
        logging.info(f"Average Displacement Error (ADE): {avg_ade:.4f} meters")
        logging.info(f"Final Displacement Error (FDE): {avg_fde:.4f} meters")
    
    logging.info(f"\nDetailed results saved to: {results_path}")
    if args.visualize:
        logging.info(f"Visualization saved to: {viz_path}")


if __name__ == '__main__':
    main()
