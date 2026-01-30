#!/usr/bin/env python3
"""
FHDP Pipeline Training Example for EVO-1

This example demonstrates EVO-1 pipeline parallel training using
FHDP's native architecture and coordination capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pipeline_trainer import FHDPipelineTrainer, FHDPipelineConfig, run_fhd_pipeline_training


def create_demo_config(scale: str = "small") -> FHDPipelineConfig:
    """Create configuration based on demo scale"""
    
    if scale == "small":
        return FHDPipelineConfig(
            experiment_name=f"evo1_fhd_small_{int(time.time())}",
            max_vehicles=5,
            max_vehicles_per_region=10,
            batch_size=4,
            encoder_learning_rate=1e-3,
            mixed_precision=True,
            fairness_enabled=True
        )
    
    elif scale == "medium":
        return FHDPipelineConfig(
            experiment_name=f"evo1_fhd_medium_{int(time.time())}",
            max_vehicles=10,
            max_vehicles_per_region=20,
            batch_size=8,
            encoder_learning_rate=5e-4,
            mixed_precision=True,
            fairness_enabled=True
        )
    
    elif scale == "large":
        return FHDPipelineConfig(
            experiment_name=f"evo1_fhd_large_{int(time.time())}",
            max_vehicles=20,
            max_vehicles_per_region=50,
            batch_size=16,
            encoder_learning_rate=1e-4,
            mixed_precision=True,
            fairness_enabled=True
        )
    
    else:
        raise ValueError(f"Unknown scale: {scale}")


def create_edge_server_configs(num_servers: int) -> list:
    """Create edge server configurations"""
    configs = []
    
    for i in range(num_servers):
        config = {
            'server_id': f'edge_server_{i}',
            'vlm_config': {
                'model_name': 'OpenGVLab/InternVL3-1B',
                'device': 'cuda' if i < 2 else 'cpu'  # First 2 servers get GPU
            }
        }
        configs.append(config)
    
    return configs


def create_vehicle_configs(num_vehicles: int, num_servers: int) -> list:
    """Create vehicle configurations"""
    configs = []
    
    for i in range(num_vehicles):
        config = {
            'vehicle_id': f'vehicle_{i}',
            'edge_server_id': f'edge_server_{i % num_servers}',
            'encoder_config': {
                'encoder_type': 'resnet18' if i % 2 == 0 else 'resnet34',
                'batch_size': 8 if i < 5 else 4,  # First 5 vehicles get larger batch
                'mixed_precision': True,
                'gradient_checkpointing': True
            },
            'resources': {
                'cpu_cores': 4 if i < 10 else 2,
                'memory_gb': 8 if i < 5 else 4,
                'gpu_available': i < 3,  # First 3 vehicles have GPU
                'gpu_memory_gb': 4 if i < 3 else 0,
                'network_bandwidth': 100 if i < 8 else 50,
                'battery_level': 80 - i * 2,  # Varying battery levels
                'thermal_state': 'normal' if i < 15 else 'warm'
            }
        }
        configs.append(config)
    
    return configs


async def run_demonstration(scale: str, num_rounds: int = 20):
    """Run FHDP pipeline demonstration"""
    
    print("=" * 80)
    print("EVO-1 FHDP Pipeline Parallel Training Demonstration")
    print("=" * 80)
    print(f"Scale: {scale}")
    print(f"Training Rounds: {num_rounds}")
    print("=" * 80)
    
    # Create configuration
    config = create_demo_config(scale)
    
    # Create server and vehicle configurations
    num_servers = min(3, config.max_vehicles // 3 + 1)
    edge_configs = create_edge_server_configs(num_servers)
    vehicle_configs = create_vehicle_configs(config.max_vehicles, num_servers)
    
    print(f"\nConfiguration:")
    print(f"  Max Vehicles: {config.max_vehicles}")
    print(f"  Edge Servers: {num_servers}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Fairness Enabled: {config.fairness_enabled}")
    print()
    
    # Initialize and run trainer
    trainer = FHDPipelineTrainer(config)
    
    try:
        print("Initializing FHDP pipeline system...")
        init_success = await trainer.initialize_system(edge_configs, vehicle_configs)
        
        if not init_success:
            print("Failed to initialize FHDP pipeline system")
            return
        
        print("Starting pipeline parallel training...")
        training_success = await trainer.start_pipeline_training(num_rounds)
        
        if not training_success:
            print("Pipeline training failed")
            return
        
        # Get and display results
        stats = trainer.get_training_statistics()
        
        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)
        print(f"Total Rounds: {stats['total_rounds']}")
        print(f"Total Time: {stats['total_training_time']:.2f} seconds")
        print(f"Average Round Time: {stats['average_round_time']:.2f} seconds")
        print(f"Final Loss: {stats['final_loss']:.4f}")
        print(f"Final Accuracy: {stats['final_accuracy']:.4f}")
        print(f"Vehicles Participated: {stats['vehicles_participated']}")
        print(f"Edge Servers Used: {stats['edge_servers_used']}")
        
        # Display fairness metrics
        if 'coordinator_status' in stats and 'fairness_metrics' in stats['coordinator_status']:
            fairness = stats['coordinator_status']['fairness_metrics']
            print(f"\nFairness Metrics:")
            print(f"  Overall Fairness: {fairness.get('overall_fairness', 0):.3f}")
            print(f"  Encoder Training Fairness: {fairness.get('encoder_training_fairness', 0):.3f}")
            print(f"  Resource Utilization Fairness: {fairness.get('resource_utilization_fairness', 0):.3f}")
            print(f"  Pipeline Participation Fairness: {fairness.get('pipeline_participation_fairness', 0):.3f}")
        
        print(f"\nResults saved to: {trainer.experiment_dir}")
        return stats
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        logging.error(f"Demonstration error: {e}")
        raise
    finally:
        await trainer.shutdown_system()


async def run_component_demos():
    """Run demonstrations of individual components using FHDP"""
    print("\n" + "=" * 80)
    print("INDIVIDUAL FHDP COMPONENT DEMONSTRATIONS")
    print("=" * 80)
    
    # Demo 1: FHDP System
    print("\n1. FHDP System Demo")
    print("-" * 40)
    
    from core.fhdp_system import FHDPSystem, SystemConfiguration
    from core.types import VehicleInfo, VehicleState
    
    fhdp_config = SystemConfiguration(
        max_vehicles_per_region=10,
        fairness_enabled=True
    )
    
    fhdp_system = FHDPSystem(fhdp_config)
    print(f"FHDP System initialized: {fhdp_config.max_vehicles_per_region} max vehicles")
    
    # Demo 2: Edge Server Integration
    print("\n2. Edge Server Integration Demo")
    print("-" * 40)
    
    from edge_server.edge_integration import EdgeServerVLMIntegration
    
    edge_server = EdgeServerVLMIntegration("demo_edge_server")
    print(f"Edge server status: {edge_server.get_server_status()}")
    
    # Demo 3: Vehicle Client
    print("\n3. Vehicle Client Demo")
    print("-" * 40)
    
    from vehicle_client.vehicle_encoder import VehicleEncoderClient
    
    vehicle_client = VehicleEncoderClient("demo_vehicle", "http://localhost:8080")
    print(f"Vehicle client status: {vehicle_client.get_vehicle_status()}")
    
    # Demo 4: Hardware Adapter
    print("\n4. Hardware Adapter Demo")
    print("-" * 40)
    
    from adapter.hardware_integration import HardwareResourceAdapter
    
    hardware_adapter = HardwareResourceAdapter("demo_device")
    adaptation_summary = hardware_adapter.get_adaptation_summary()
    print(f"Hardware adaptation: {adaptation_summary['resource_class']}")
    print(f"Adaptation strategies: {adaptation_summary['adaptation_strategies']}")
    
    # Demo 5: Coordinator
    print("\n5. FHDP Coordinator Demo")
    print("-" * 40)
    
    from coordinator.fhdp_coordinator import FHDPipelineCoordinator
    
    coordinator = FHDPipelineCoordinator()
    print(f"Coordinator status: {coordinator.get_coordinator_status()}")


def main():
    """Main function for demonstration script"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="EVO-1 FHDP Pipeline Training Demo")
    
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small",
                        help="Scale of demonstration (default: small)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of training rounds (default: 20)")
    parser.add_argument("--components", action="store_true", help="Run individual component demos")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run component demos if requested
    if args.components:
        asyncio.run(run_component_demos())
        return
    
    # Run main demonstration
    asyncio.run(run_demonstration(args.scale, args.rounds))


if __name__ == "__main__":
    main()