#!/usr/bin/env python3
"""
Inference script for EVO-1 autonomous driving model

This script provides command-line interface for running inference
with trained EVO-1 models in various modes.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from EVO1.inference.driving_inference import (
    DrivingInferencePipeline, 
    InferenceConfig, 
    parse_inference_arguments,
    create_inference_config
)
from EVO1.data.nuscenes_loader import create_dataloader


def main():
    """Main inference function"""
    # Parse arguments
    args = parse_inference_arguments()
    
    # Create inference configuration
    config = create_inference_config(args)
    
    # Create inference pipeline
    pipeline = DrivingInferencePipeline(config)
    
    print(f"Starting inference pipeline on device: {config.device}")
    print(f"Model loaded from: {config.model_path}")
    print(f"Output directory: {config.output_dir}")
    
    try:
        if args.evaluate:
            # Run evaluation on test set
            print("Running evaluation on test set...")
            
            # Create test data loader
            from EVO1.utils.config import EVO1DrivingConfig
            
            # Load model config
            if args.config_path and os.path.exists(args.config_path):
                model_config = EVO1DrivingConfig.from_yaml(args.config_path)
            else:
                model_config = EVO1DrivingConfig()
            
            test_loader = create_dataloader(
                config=model_config.data,
                model_config=model_config.model,
                split='test',
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            if test_loader:
                # Run evaluation
                results = pipeline.evaluate_model(test_loader)
                
                print("Evaluation Results:")
                for metric, value in results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            else:
                print("No test data found. Skipping evaluation.")
        
        if args.enable_streaming:
            # Start streaming server
            print(f"WebSocket server running on port {config.streaming_port}")
            print("Send JSON messages with 'images', 'state', and 'instruction' fields")
            print("Example message:")
            print(json.dumps({
                "images": ["image1.jpg", "image2.jpg", "image3.jpg"],
                "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "instruction": "Continue driving safely"
            }))
            
            # Keep server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down streaming server...")
        
        if not args.evaluate and not args.enable_streaming:
            # Run demo inference
            print("Running demo inference...")
            
            # Create dummy input for demo
            demo_images = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)
            ]
            demo_state = np.array([
                0.0, 0.0, 0.0,  # position [x, y, z]
                0.0, 0.0, 0.0,  # orientation [roll, pitch, yaw]
                5.0, 0.0, 0.0,  # velocity [vx, vy, vz]
                0.0, 0.0, 0.0   # acceleration [ax, ay, az]
            ], dtype=np.float32)
            demo_instruction = "Continue driving straight at 5 m/s"
            
            # Run inference
            start_time = time.time()
            result = pipeline.inference(demo_images, demo_state, demo_instruction)
            inference_time = time.time() - start_time
            
            print("Demo Inference Results:")
            print(f"  Inference time: {inference_time:.4f} seconds")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Waypoints shape: {result['waypoints'].shape}")
            print(f"  Controls shape: {result['controls'].shape}")
            print(f"  First waypoint: {result['waypoints'][0].tolist()}")
            print(f"  First control: {result['controls'][0].tolist()}")
            
            # Save demo results
            pipeline.save_predictions([result], 'demo_prediction.json')
    
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    
    finally:
        # Print performance stats
        stats = pipeline.get_performance_stats()
        if stats:
            print("\nPerformance Statistics:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.4f}")


if __name__ == '__main__':
    main()