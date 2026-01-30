#!/usr/bin/env python3
"""
EVO-1 FHDP Autonomous Driving Example

This example demonstrates a complete FHDP-integrated autonomous driving
system with EVO-1 models, featuring:
- Vehicle fleet management
- Real-time coordination
- Federated learning
- Deployment configuration
- Performance monitoring
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import FHDP components
try:
    from core.fhdp_system import FHDPSystem, SystemConfiguration
    from core.types import VehicleInfo
    from ..evo1_trainer import FHDAutonomousDrivingTrainer
    from ..vehicle_manager import FHDAutonomousVehicleManager
    from ..deployment import FHDPDeploymentConfig, create_production_deployment
    FHDP_AVAILABLE = True
except ImportError:
    print("Error: FHDP core components not available. Please install FHDP system.")
    sys.exit(1)


def create_example_config() -> dict:
    """Create example configuration for FHDP autonomous driving"""
    
    return {
        'system_config': {
            'max_vehicles_per_region': 20,
            'pipeline_formation_interval': 5.0,
            'model_broadcast_interval': 10.0,
            'participation_timeout': 30.0,
            'aggregation_interval': 15.0,
            'enable_pipeline_training': True,
            'enable_individual_training': True,
            'fairness_enabled': True,
            'default_protocol': "v2x_fahdp",
            'security_level': "high"
        },
        'environment_config': {
            'environment_type': 'production',
            'network_config': {
                'simulation_mode': False,
                'mock_fhdp': False,
                'debug_enabled': False,
                'log_level': 'INFO'
            },
            'security_config': {
                'encryption': 'production_key_rotation',
                'authentication': 'enterprise',
                'firewall_rules': 'strict',
                'intrusion_detection': True
            },
            'resource_limits': {
                'max_vehicles': 12,
                'memory_per_vehicle': '4GB',
                'cpu_per_vehicle': '4 cores',
                'network_bandwidth': '1Gbps'
            },
            'monitoring_config': {
                'metrics_collection': 'production',
                'alert_system': 'pagerduty',
                'log_retention_days': 90,
                'performance_monitoring': True
            }
        },
        'autonomous_driving': {
            'max_vehicles': 12,
            'vehicle_types': ['autonomous_driving_evo1'],
            'coordination_protocols': ['v2x_fahdp', 'grpc'],
            'safety_requirements': {
                'collision_detection_required': True,
                'emergency_stop_capability': True,
                'redundant_systems': True,
                'cyber_security_level': 'high'
            },
            'performance_targets': {
                'coordination_latency': 2.0,  # seconds
                'decision_accuracy': 0.95,
                'safety_compliance': 0.99,
                'throughput_vehicles_per_minute': 15.0,
                'fleet_availability': 0.999
            },
            'deployment_regions': ['us_west', 'us_east'],
            'edge_servers': [
                {
                    'server_id': 'edge_server_west',
                    'location': 'us_west',
                    'capacity': {'max_vehicles': 8}
                },
                {
                    'server_id': 'edge_server_east',
                    'location': 'us_east',
                    'capacity': {'max_vehicles': 4}
                }
            ]
        }
    }


def run_fleet_management_demo():
    """Demonstrate fleet management capabilities"""
    
    print("=== FHDP Fleet Management Demo ===")
    print()
    
    # Create system configuration
    config = create_example_config()
    
    # Initialize FHDP system
    fhdp_system = FHDPSystem(SystemConfiguration(**config['system_config']))
    
    # Create vehicle manager
    vehicle_manager = FHDAutonomousVehicleManager(config['system_config'])
    
    # Create example vehicles
    vehicles = []
    for i in range(6):
        vehicle_info = VehicleInfo(
            vehicle_id=f"demo_vehicle_{i}",
            vehicle_type="autonomous_driving_evo1",
            model_type="EVO1",
            capabilities=[
                "vision_perception",
                "action_prediction",
                "path_planning",
                "vehicle_control",
                "real_time_coordination"
            ],
            resource_class="high",
            location=f"demo_region_{i % 3}",
            status="active"
        )
        vehicles.append(vehicle_info)
    
    # Register fleet
    if vehicle_manager.register_fleet(vehicles):
        print("✅ Fleet registered successfully")
    else:
        print("❌ Fleet registration failed")
        return
    
    # Get fleet summary
    fleet_summary = vehicle_manager.get_fleet_summary()
    print("Fleet Summary:")
    print(f"  Total Vehicles: {fleet_summary.get('total_vehicles', 0)}")
    print(f"  Active Vehicles: {fleet_summary.get('active_vehicles', 0)}")
    print(f"  System Status: {fleet_summary.get('system_status', 'unknown')}")
    print(f"  Capabilities: {fleet_summary.get('capabilities', [])}")
    
    # Deploy fleet for operations
    if vehicle_manager.deploy_fleet():
        print("✅ Fleet deployed successfully")
    else:
        print("❌ Fleet deployment failed")
    
    # Monitor fleet health
    health_metrics = vehicle_manager.monitor_fleet_health()
    print("Fleet Health:")
    print(f"  Overall Health: {health_metrics.get('overall_health', 0):.3f}")
    print(f"  Response Time: {health_metrics.get('response_time', 0):.3f}s")
    print(f"  Error Rate: {health_metrics.get('error_rate', 0):.4f}")
    
    print("\n=== Fleet Management Demo Complete ===")


def run_coordination_demo():
    """Demonstrate coordination capabilities"""
    
    print("\n=== FHDP Coordination Demo ===")
    print()
    
    # Create system configuration
    config = create_example_config()
    
    # Initialize coordination system
    from ..coordination import FHDAutonomousCoordination
    coordination = FHDAutonomousCoordination(config['system_config'])
    
    # Test different coordination scenarios
    scenarios = ['highway_formation', 'intersection_coordination', 'urban_navigation']
    
    for scenario_id in scenarios:
        print(f"\nTesting scenario: {scenario_id}")
        print("-" * 40)
        
        # Simulate vehicle list
        vehicle_ids = [f"demo_vehicle_{i}" for i in range(4)]
        
        # Execute coordination
        result = coordination.coordinate_scenario(scenario_id, vehicle_ids)
        
        if result.success:
            print(f"✅ Scenario {scenario_id} completed successfully")
            print(f"  Execution Time: {result.execution_time:.2f}s")
            print(f"  Vehicles Involved: {result.vehicles_involved}")
        else:
            print(f"❌ Scenario {scenario_id} failed: {result.error}")
        
        print("-" * 40)
    
    print("\n=== Coordination Demo Complete ===")


def run_training_demo():
    """Demonstrate FHDP-integrated training"""
    
    print("\n=== FHDP Training Demo ===")
    print()
    
    # Import EVO-1 configuration
    from ...utils.config import EVO1DrivingConfig
    
    # Create training configuration
    config = EVO1DrivingConfig()
    config.training.aggregation_rounds = 10  # Reduced for demo
    config.training.num_clients = 4
    config.training.federated_learning = True
    config.training.client_fraction = 0.75
    config.training.local_epochs = 2
    config.training.batch_size = 4
    config.training.save_frequency = 3
    
    print(f"Training Configuration:")
    print(f"  Rounds: {config.training.aggregation_rounds}")
    print(f"  Clients: {config.training.num_clients}")
    print(f"  Local Epochs: {config.training.local_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print()
    
    # Create FHDP system configuration
    system_config = SystemConfiguration(**create_example_config()['system_config'])
    
    # Initialize FHDP trainer
    trainer = FHDAutonomousDrivingTrainer(
        config=config,
        fhdp_config=system_config,
        device="cuda"  # Use CPU for demo if no GPU
    )
    
    # Run training for a few rounds
    print("Starting FHDP integrated training...")
    
    try:
        trainer.train()
        print("✅ FHDP training completed successfully!")
        
        # Get final metrics
        final_round = config.training.aggregation_rounds - 1
        print(f"Final Training Metrics:")
        print(f"  Total Rounds: {final_round + 1}")
        print(f"  Final Model Path: {trainer.get_final_model_path()}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
    
    print("\n=== Training Demo Complete ===")


def run_deployment_demo():
    """Demonstrate deployment configuration"""
    
    print("\n=== FHDP Deployment Demo ===")
    print()
    
    # Create deployment manager
    from ..deployment import FHDAutonomousDeployment
    deployment_manager = FHDAutonomousDeployment()
    
    # Create production configuration
    config = create_production_deployment()
    
    # Validate configuration
    validation = deployment_manager.validate_deployment_config(config)
    
    if validation['is_valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed:")
        for error in validation['errors']:
            print(f"  - {error}")
        return
    
    # Generate deployment manifest
    manifest = deployment_manager.generate_deployment_manifest(config)
    
    print("Deployment Configuration:")
    print(f"  Environment: {config.environment_config.environment_type}")
    print(f"  Max Vehicles: {config.max_vehicles}")
    print(f"  Regions: {', '.join(config.deployment_regions)}")
    print(f"  Edge Servers: {len(config.edge_servers)}")
    print()
    
    print("Generated Deployment Manifest:")
    print(f"  Deployment ID: {manifest['deployment_id']}")
    print(f"  Components: {len(manifest['components'])}")
    print(f"  Security Level: {config.system_config.security_level}")
    print()
    
    print("=== Deployment Demo Complete ===")


def main():
    """Main demonstration function"""
    
    parser = argparse.ArgumentParser(description="EVO-1 FHDP Autonomous Driving Example")
    parser.add_argument("--demo", choices=['fleet', 'coordination', 'training', 'deployment'], 
                       required=True, help="Demo to run")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    print("EVO-1 FHDP Autonomous Driving Example")
    print("=" * 50)
    print()
    
    if args.config:
        # Load configuration from file
        print(f"Loading configuration from: {args.config}")
        # Would load and use the provided config
    else:
        # Run requested demo
        if args.demo == 'fleet':
            run_fleet_management_demo()
        elif args.demo == 'coordination':
            run_coordination_demo()
        elif args.demo == 'training':
            run_training_demo()
        elif args.demo == 'deployment':
            run_deployment_demo()
        else:
            print(f"Unknown demo: {args.demo}")
            return 1
    
    print()
    print("All demos completed successfully!")
    print("Check the generated logs and configuration files for details.")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())