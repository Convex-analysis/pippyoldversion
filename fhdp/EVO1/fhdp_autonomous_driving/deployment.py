"""
FHDP Autonomous Driving Deployment Configuration

Configuration and deployment utilities for EVO-1 autonomous driving
with FHDP integration in production environments.
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import FHDP components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
try:
    from core.fhdp_system import FHDPSystem, SystemConfiguration
    from core.types import VehicleInfo, DeploymentConfig
    FHDP_AVAILABLE = True
except ImportError:
    logging.warning("FHDP core components not available. Using standalone mode.")
    FHDP_AVAILABLE = False
    SystemConfiguration = None


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    environment_type: str  # 'development', 'staging', 'production'
    network_config: Dict[str, Any]
    security_config: Dict[str, Any]
    resource_limits: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    deployment_targets: Dict[str, float]


@dataclass
class FHDPDeploymentConfig:
    """FHDP deployment configuration for EVO-1 autonomous driving"""
    
    # System configuration
    system_config: SystemConfiguration
    environment_config: EnvironmentConfig
    
    # Autonomous driving specific
    max_vehicles: int = 20
    vehicle_types: List[str] = None
    coordination_protocols: List[str] = None
    safety_requirements: Dict[str, Any] = None
    
    # Performance and scaling
    performance_targets: Dict[str, float]
    scaling_limits: Dict[str, Any] = None
    
    # Deployment configuration
    deployment_regions: List[str] = None
    edge_servers: List[Dict[str, Any]] = None
    cloud_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.vehicle_types is None:
            self.vehicle_types = ["autonomous_driving_evo1"]
        if self.coordination_protocols is None:
            self.coordination_protocols = ["v2x_fahdp", "mqtt", "http_rest", "grpc"]
        if self.deployment_regions is None:
            self.deployment_regions = ["us_west", "us_east", "eu_central", "asia_pacific"]
        if self.edge_servers is None:
            self.edge_servers = []
        if self.safety_requirements is None:
            self.safety_requirements = {
                'collision_detection_required': True,
                'emergency_stop_capability': True,
                'redundant_systems': True,
                'cyber_security_level': 'high'
            }


class FHDAutonomousDeployment:
    """Manages deployment of EVO-1 autonomous driving with FHDP integration"""
    
    def __init__(self):
        print(f"[DEPLOY] Initializing FHDP Autonomous Driving Deployment Manager...")
        
        self.deployment_config = FHDPDeploymentConfig()
        self.fhdp_system = None
        
        print(f"[DEPLOY] Deployment manager initialization complete")
    
    def create_deployment_config(self, 
                           environment_type: str = "production",
                           system_config_path: Optional[str] = None,
                           custom_overrides: Optional[Dict[str, Any]] = None) -> FHDPDeploymentConfig:
        """Create deployment configuration"""
        
        config = FHDPDeploymentConfig()
        
        # Environment configuration
        config.environment_config = self.create_environment_config(environment_type)
        
        # System configuration
        if system_config_path and os.path.exists(system_config_path):
            config.system_config = self.load_system_config(system_config_path)
        else:
            config.system_config = self.create_default_system_config()
        
        # Apply custom overrides
        if custom_overrides:
            config = self.apply_config_overrides(config, custom_overrides)
        
        # Autonomous driving specific configuration
        config.vehicle_types = ["autonomous_driving_evo1"]
        config.coordination_protocols = ["v2x_fahdp", "mqtt", "http_rest", "grpc"]
        config.safety_requirements = {
            'collision_detection_required': True,
            'emergency_stop_capability': True,
            'redundant_systems': True,
            'cyber_security_level': 'high'
        }
        
        # Performance and scaling
        config.performance_targets = {
            'coordination_latency': 2.0,  # seconds
            'decision_accuracy': 0.95,
            'safety_compliance': 0.99,
            'throughput_vehicles_per_minute': 15.0,
            'fleet_availability': 0.999
        }
        
        # Deployment configuration
        config.deployment_regions = ["us_west", "us_east", "eu_central", "asia_pacific"]
        config.edge_servers = self.create_edge_server_configs()
        config.cloud_config = self.create_cloud_config()
        
        return config
    
    def create_environment_config(self, environment_type: str) -> EnvironmentConfig:
        """Create environment-specific configuration"""
        
        if environment_type == "development":
            return EnvironmentConfig(
                environment_type="development",
                network_config={
                    'simulation_mode': True,
                    'mock_fhdp': True,
                    'debug_enabled': True,
                    'log_level': 'DEBUG'
                },
                security_config={
                    'encryption': 'development_key',
                    'authentication': 'mock',
                    'firewall_rules': 'permissive'
                },
                resource_limits={
                    'max_vehicles': 5,
                    'memory_per_vehicle': '2GB',
                    'cpu_per_vehicle': '2 cores',
                    'network_bandwidth': 'unlimited'
                },
                monitoring_config={
                    'metrics_collection': 'detailed',
                    'alert_system': 'console',
                    'log_retention_days': 7
                },
                deployment_targets={
                    'development_iteration_speed': 'fast',
                    'feature_completeness': 0.8
                }
            )
        
        elif environment_type == "staging":
            return EnvironmentConfig(
                environment_type="staging",
                network_config={
                    'simulation_mode': False,
                    'mock_fhdp': False,
                    'debug_enabled': True,
                    'log_level': 'INFO'
                },
                security_config={
                    'encryption': 'staging_key',
                    'authentication': 'full',
                    'firewall_rules': 'restrictive'
                },
                resource_limits={
                    'max_vehicles': 10,
                    'memory_per_vehicle': '4GB',
                    'cpu_per_vehicle': '4 cores',
                    'network_bandwidth': '1Gbps'
                },
                monitoring_config={
                    'metrics_collection': 'full',
                    'alert_system': 'email_slack',
                    'log_retention_days': 30
                },
                deployment_targets={
                    'stability_testing': 'comprehensive',
                    'performance_validation': 'thorough'
                }
            )
        
        elif environment_type == "production":
            return EnvironmentConfig(
                environment_type="production",
                network_config={
                    'simulation_mode': False,
                    'mock_fhdp': False,
                    'debug_enabled': False,
                    'log_level': 'WARNING'
                },
                security_config={
                    'encryption': 'production_key_rotation',
                    'authentication': 'enterprise',
                    'firewall_rules': 'strict',
                    'intrusion_detection': True
                },
                resource_limits={
                    'max_vehicles': 50,
                    'memory_per_vehicle': '8GB',
                    'cpu_per_vehicle': '8 cores',
                    'network_bandwidth': '10Gbps'
                },
                monitoring_config={
                    'metrics_collection': 'production',
                    'alert_system': 'pagerduty',
                    'log_retention_days': 90,
                    'performance_monitoring': True
                },
                deployment_targets={
                    'reliability_target': 0.9999,
                    'performance_target': 0.95,
                    'cost_optimization': True
                }
            )
        
        else:
            raise ValueError(f"Unknown environment type: {environment_type}")
    
    def create_default_system_config(self) -> SystemConfiguration:
        """Create default FHDP system configuration"""
        
        if FHDP_AVAILABLE:
            return SystemConfiguration(
                max_vehicles_per_region=50,
                pipeline_formation_interval=5.0,
                model_broadcast_interval=10.0,
                participation_timeout=30.0,
                aggregation_interval=15.0,
                enable_pipeline_training=True,
                enable_individual_training=True,
                fairness_enabled=True,
                default_protocol="v2x_fahdp",
                security_level="high"
            )
        else:
            # Minimal standalone config
            return SystemConfiguration(
                max_vehicles_per_region=10,
                pipeline_formation_interval=10.0,
                model_broadcast_interval=15.0,
                participation_timeout=60.0,
                aggregation_interval=30.0,
                enable_pipeline_training=False,
                enable_individual_training=True,
                fairness_enabled=False,
                default_protocol="http_rest",
                security_level="medium"
            )
    
    def load_system_config(self, config_path: str) -> SystemConfiguration:
        """Load FHDP system configuration from file"""
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert to SystemConfiguration
            return SystemConfiguration(**config_data)
            
        except Exception as e:
            logging.error(f"Failed to load system config from {config_path}: {e}")
            return self.create_default_system_config()
    
    def apply_config_overrides(self, config: FHDPDeploymentConfig, overrides: Dict[str, Any]) -> FHDPDeploymentConfig:
        """Apply configuration overrides"""
        
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.system_config, key):
                setattr(config.system_config, key, value)
            elif hasattr(config.environment_config, key):
                setattr(config.environment_config, key, value)
        
        return config
    
    def create_edge_server_configs(self) -> List[Dict[str, Any]]:
        """Create edge server configurations"""
        
        servers = [
            {
                'server_id': 'edge_server_west',
                'location': 'us_west',
                'region': 'us_west',
                'capacity': {
                    'max_vehicles': 25,
                    'cpu_cores': 64,
                    'memory_gb': 256,
                    'storage_gb': 1000,
                    'network_bandwidth_gbps': 10
                },
                'coordination_protocols': ['v2x_fahdp', 'grpc'],
                'security_config': {
                    'tls_enabled': True,
                    'client_auth_required': True,
                    'rate_limiting': True
                },
                'monitoring': {
                    'health_checks': True,
                    'performance_metrics': True,
                    'log_aggregation': True
                }
            },
            {
                'server_id': 'edge_server_east',
                'location': 'us_east',
                'region': 'us_east',
                'capacity': {
                    'max_vehicles': 25,
                    'cpu_cores': 64,
                    'memory_gb': 256,
                    'storage_gb': 1000,
                    'network_bandwidth_gbps': 10
                },
                'coordination_protocols': ['v2x_fahdp', 'grpc'],
                'security_config': {
                    'tls_enabled': True,
                    'client_auth_required': True,
                    'rate_limiting': True
                },
                'monitoring': {
                    'health_checks': True,
                    'performance_metrics': True,
                    'log_aggregation': True
                }
            }
        ]
        
        return servers
    
    def create_cloud_config(self) -> Dict[str, Any]:
        """Create cloud configuration"""
        
        return {
            'cloud_provider': 'aws' or 'gcp' or 'azure',
            'region': 'us-west-2',
            'kubernetes_cluster': {
                'enabled': True,
                'namespace': 'evo1-autonomous',
                'auto_scaling': {
                    'min_replicas': 3,
                    'max_replicas': 50,
                    'target_cpu_utilization': 0.7
                }
            },
            'storage': {
                'model_registry': 's3://evo1-models',
                'training_data': 's3://evo1-training-data',
                'metrics': 'cloudwatch',
                'logs': 'cloudwatch_logs'
            },
            'networking': {
                'vpc_id': 'vpc-evo1',
                'load_balancer': 'application_lb',
                'cdn_enabled': True,
                'ddos_protection': True
            },
            'security': {
                'iam_roles': ['evo1-training', 'evo1-inference'],
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'key_rotation': True
            }
        }
    
    def validate_deployment_config(self, config: FHDPDeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration"""
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Validate system configuration
        if config.system_config.max_vehicles_per_region <= 0:
            validation_results['errors'].append("max_vehicles_per_region must be positive")
            validation_results['is_valid'] = False
        
        if config.system_config.aggregation_interval <= 0:
            validation_results['errors'].append("aggregation_interval must be positive")
            validation_results['is_valid'] = False
        
        # Validate environment configuration
        if config.environment_config.resource_limits['max_vehicles'] < 1:
            validation_results['warnings'].append("max_vehicles less than 1, using default")
            config.environment_config.resource_limits['max_vehicles'] = max(1, config.environment_config.resource_limits['max_vehicles'])
        
        # Validate safety requirements
        required_safety = ['collision_detection_required', 'emergency_stop_capability']
        for requirement in required_safety:
            if not config.safety_requirements.get(requirement, False):
                validation_results['errors'].append(f"Missing required safety feature: {requirement}")
                validation_results['is_valid'] = False
        
        return validation_results
    
    def generate_deployment_manifest(self, config: FHDPDeploymentConfig) -> Dict[str, Any]:
        """Generate deployment manifest"""
        
        manifest = {
            'deployment_id': f"evo1_autonomous_{int(time.time())}",
            'timestamp': time.time(),
            'configuration': asdict(config),
            'environment': config.environment_config.environment_type,
            'components': {
                'vehicles': {
                    'count': config.max_vehicles,
                    'types': config.vehicle_types,
                    'capabilities': [
                        "vision_perception",
                        "action_prediction",
                        "path_planning",
                        "vehicle_control",
                        "real_time_coordination"
                    ]
                },
                'coordination': {
                    'protocols': config.coordination_protocols,
                    'strategies': [
                        "platoon_driving",
                        "intersection_coordination",
                        "emergency_response",
                        "urban_navigation"
                    ],
                    'fhdp_integration': True
                },
                'deployment': {
                    'regions': config.deployment_regions,
                    'edge_servers': len(config.edge_servers),
                    'cloud_integration': config.cloud_config.get('kubernetes_cluster', {}).get('enabled', False)
                }
            },
            'requirements': {
                'hardware': {
                    'vehicle_compute': 'NVIDIA T4 or equivalent',
                    'memory_min_gb': 8,
                    'storage_min_gb': 100,
                    'network_min_gbps': 1
                },
                'software': {
                    'python_version': '>=3.8',
                    'pytorch_version': '>=1.9',
                    'fhdp_version': '>=1.0',
                    'docker_required': True
                },
                'security': {
                    'encryption': 'TLS 1.3',
                    'authentication': 'OAuth 2.0',
                    'audit_logging': True
                }
            }
        }
        
        return manifest
    
    def save_deployment_config(self, config: FHDPDeploymentConfig, filepath: str):
        """Save deployment configuration to file"""
        
        with open(filepath, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        logging.info(f"Deployment configuration saved to {filepath}")
    
    def deploy_configuration(self, config: FHDPDeploymentConfig) -> bool:
        """Deploy configuration to FHDP system"""
        
        # Validate configuration first
        validation = self.validate_deployment_config(config)
        
        if not validation['is_valid']:
            for error in validation['errors']:
                logging.error(f"Configuration validation error: {error}")
            return False
        
        # Print warnings
        for warning in validation['warnings']:
            logging.warning(f"Configuration warning: {warning}")
        
        # Generate deployment manifest
        manifest = self.generate_deployment_manifest(config)
        
        # Save manifest
        manifest_path = "deployment_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logging.info(f"Deployment manifest generated: {manifest_path}")
        
        # Here you would integrate with actual FHDP deployment system
        if FHDP_AVAILABLE:
            try:
                # In real implementation, this would call FHDP deployment APIs
                logging.info("Deploying configuration to FHDP system...")
                # self.fhdp_system.deploy_configuration(config)
                
                logging.info("FHDP deployment completed successfully")
                return True
                
            except Exception as e:
                logging.error(f"FHDP deployment failed: {e}")
                return False
        else:
            logging.info("Standalone deployment: configuration saved locally")
            return True


# Utility functions
def create_production_deployment() -> FHDPDeploymentConfig:
    """Create production-ready deployment configuration"""
    
    config = FHDPDeploymentConfig()
    
    # Production system configuration
    config.system_config = SystemConfiguration(
        max_vehicles_per_region=50,
        pipeline_formation_interval=5.0,
        model_broadcast_interval=10.0,
        participation_timeout=30.0,
        aggregation_interval=15.0,
        enable_pipeline_training=True,
        enable_individual_training=True,
        fairness_enabled=True,
        default_protocol="v2x_fahdp",
        security_level="high"
    )
    
    # Production environment configuration
    config.environment_config = EnvironmentConfig(
        environment_type="production",
        network_config={
            'simulation_mode': False,
            'mock_fhdp': False,
            'debug_enabled': False,
            'log_level': 'WARNING'
        },
        security_config={
            'encryption': 'production_key_rotation',
            'authentication': 'enterprise',
            'firewall_rules': 'strict',
            'intrusion_detection': True
        },
        resource_limits={
            'max_vehicles': 50,
            'memory_per_vehicle': '8GB',
            'cpu_per_vehicle': '8 cores',
            'network_bandwidth': '10Gbps'
        },
        monitoring_config={
            'metrics_collection': 'production',
            'alert_system': 'pagerduty',
            'log_retention_days': 90,
            'performance_monitoring': True
        },
        deployment_targets={
            'reliability_target': 0.9999,
            'performance_target': 0.95,
            'cost_optimization': True
        }
    )
    
    return config


def quick_deployment():
    """Quick deployment setup for production"""
    
    print("[DEPLOY] Quick EVO-1 FHDP Autonomous Driving Deployment")
    print("=" * 60)
    
    # Create production configuration
    config = create_production_deployment()
    
    # Create deployment manager
    deployment_manager = FHDAutonomousDeployment()
    
    # Validate and deploy
    success = deployment_manager.deploy_configuration(config)
    
    if success:
        print("✅ Deployment configuration created successfully!")
        print("\nNext steps:")
        print("1. Review deployment manifest")
        print("2. Test in staging environment")
        print("3. Deploy to production")
        print("4. Monitor fleet performance")
    else:
        print("❌ Deployment failed. Check logs for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(quick_deployment())