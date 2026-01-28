"""
Edge Server Implementation for FHDP System

Coordinates all edge server components and provides unified interface.
"""
import time
import threading
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .mobility_predictor import MobilityPredictor
from .template_manager import TemplateManager
from .aggregation_engine import AsynchronousAggregator
from .resource_classifier import ResourceClassifier
from core.types import VehicleInfo, ModelUpdate, AggregationResult, FairnessMetrics
from core.constants import ASYNC_AGGREGATION_INTERVAL

class EdgeServer:
    """Main Edge Server implementation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.mobility_predictor = MobilityPredictor()
        self.template_manager = TemplateManager()
        self.aggregator = AsynchronousAggregator()
        self.resource_classifier = ResourceClassifier()
        
        # Server state
        self.server_active = False
        self.registered_vehicles: Dict[str, VehicleInfo] = {}
        self.coverage_area = (1000.0, 1000.0)  # meters x meters
        
        # Statistics
        self.server_stats = {
            'vehicles_served': 0,
            'templates_generated': 0,
            'aggregations_performed': 0,
            'predictions_made': 0,
            'server_uptime': 0.0
        }
        
        # Threading
        self.monitoring_thread = None
        self.stop_event = threading.Event()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception as e:
                print(f"Failed to load config from {config_path}: {e}, using defaults")
        
        # Return default configuration
        return {
            'edge_server': {
                'mobility_prediction': {
                    'update_interval': 1.0,
                    'prediction_horizon': 10.0
                },
                'template_management': {
                    'generation_interval': 60.0,
                    'cache_size': 1000
                },
                'aggregation': {
                    'min_participants': 3,
                    'interval': ASYNC_AGGREGATION_INTERVAL
                },
                'resource_classification': {
                    'high_resource_cpu': 0.8,
                    'medium_resource_cpu': 0.5
                }
            }
        }
    
    def start_server(self):
        """Start edge server services"""
        if self.server_active:
            return
        
        print("Starting FHDP Edge Server...")
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.server_active = True
        self.start_time = time.time()
        
        print("Edge Server started successfully")
    
    def stop_server(self):
        """Stop edge server services"""
        if not self.server_active:
            return
        
        print("Stopping FHDP Edge Server...")
        
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop aggregator
        self.aggregator.shutdown()
        
        self.server_active = False
        self.server_stats['server_uptime'] = time.time() - self.start_time
        
        print("Edge Server stopped")
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while not self.stop_event.is_set():
            try:
                # Periodic tasks can be added here
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Monitoring worker error: {e}")
                time.sleep(5.0)
    
    def register_vehicle(self, vehicle_info: VehicleInfo) -> bool:
        """Register vehicle with edge server"""
        try:
            # Validate vehicle position within coverage area
            if not self._is_vehicle_in_coverage(vehicle_info):
                print(f"Vehicle {vehicle_info.vehicle_id} outside coverage area")
                return False
            
            # Register vehicle
            self.registered_vehicles[vehicle_info.vehicle_id] = vehicle_info
            
            # Update mobility prediction
            self.mobility_predictor.update_vehicle_mobility(vehicle_info)
            
            # Classify vehicle resources
            self.resource_classifier.create_resource_profile(vehicle_info)
            
            # Update statistics
            self.server_stats['vehicles_served'] += 1
            
            print(f"Vehicle {vehicle_info.vehicle_id} registered successfully")
            return True
            
        except Exception as e:
            print(f"Failed to register vehicle {vehicle_info.vehicle_id}: {e}")
            return False
    
    def unregister_vehicle(self, vehicle_id: str):
        """Unregister vehicle from edge server"""
        if vehicle_id in self.registered_vehicles:
            del self.registered_vehicles[vehicle_id]
            print(f"Vehicle {vehicle_id} unregistered")
    
    def _is_vehicle_in_coverage(self, vehicle_info: VehicleInfo) -> bool:
        """Check if vehicle is within server coverage area"""
        x, y = vehicle_info.position
        width, height = self.coverage_area
        
        # Coverage area is centered at origin, spanning from -width/2 to width/2
        min_x, max_x = -width/2, width/2
        min_y, max_y = -height/2, height/2
        
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def update_vehicle_position(self, vehicle_id: str, position: tuple, velocity: float, direction: float):
        """Update vehicle position and mobility data"""
        if vehicle_id not in self.registered_vehicles:
            return
        
        vehicle_info = self.registered_vehicles[vehicle_id]
        vehicle_info.position = position
        vehicle_info.velocity = velocity
        vehicle_info.direction = direction
        vehicle_info.last_seen = time.time()
        
        # Update mobility prediction
        self.mobility_predictor.update_vehicle_mobility(vehicle_info)
        
        self.server_stats['predictions_made'] += 1
    
    def get_mobility_prediction(self, vehicle_id: str, horizon: float = 10.0):
        """Get mobility prediction for vehicle"""
        return self.mobility_predictor.predict_mobility(vehicle_id, horizon)
    
    def find_pipeline_template(self, vehicle_ids: List[str]):
        """Find suitable pipeline template for vehicles"""
        vehicle_infos = [self.registered_vehicles.get(vid) for vid in vehicle_ids]
        vehicle_infos = [v for v in vehicle_infos if v is not None]
        
        if len(vehicle_infos) >= 2:
            template = self.template_manager.find_template_for_vehicles(vehicle_infos)
            return template
        
        return None
    
    def submit_model_update(self, update: ModelUpdate, fairness_metrics: Optional[FairnessMetrics] = None):
        """Submit model update for aggregation"""
        self.aggregator.submit_update(update, fairness_metrics)
        self.server_stats['aggregations_performed'] += 1
    
    def get_global_model(self):
        """Get current global model"""
        return self.aggregator.get_global_model()
    
    def classify_vehicle_resources(self, vehicle_info: VehicleInfo):
        """Classify vehicle resources"""
        return self.resource_classifier.classify_vehicle(vehicle_info)
    
    def get_fairness_metrics(self, vehicle_id: str):
        """Get fairness metrics for vehicle"""
        return self.resource_classifier.get_fairness_metrics(vehicle_id)
    
    def predict_pipeline_stability(self, vehicle_ids: List[str], duration: float) -> float:
        """Predict pipeline stability over duration"""
        return self.mobility_predictor.predict_pipeline_stability(vehicle_ids, duration)
    
    def register_pipeline_success(self, pipeline_id: str, success: bool, duration: float):
        """Register pipeline execution for template learning"""
        # Create mock pipeline object for template generation
        # In real implementation, this would receive actual pipeline data
        from ..core.types import Pipeline
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            template_id="temp_" + pipeline_id,
            vehicles=[],  # Would be populated with actual data
            stages=[],
            start_time=time.time() - duration,
            expected_completion=time.time()
        )
        
        self.template_manager.register_successful_pipeline(pipeline, success, duration)
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time if hasattr(self, 'start_time') else 0
        
        return {
            'server_active': self.server_active,
            'uptime_seconds': uptime,
            'registered_vehicles': len(self.registered_vehicles),
            'coverage_area': self.coverage_area,
            **self.server_stats,
            'mobility_stats': {
                'predictions_made': self.server_stats['predictions_made']
            },
            'template_stats': self.template_manager.get_template_statistics(),
            'aggregation_stats': self.aggregator.get_aggregation_statistics(),
            'classification_stats': self.resource_classifier.get_classification_statistics()
        }
    
    def get_vehicle_list(self) -> List[Dict[str, Any]]:
        """Get list of registered vehicles with their status"""
        vehicles = []
        
        for vehicle_id, vehicle_info in self.registered_vehicles.items():
            # Get resource classification
            resource_class = self.resource_classifier.classify_vehicle(vehicle_info)
            
            # Get fairness metrics
            fairness_metrics = self.resource_classifier.get_fairness_metrics(vehicle_id)
            
            vehicles.append({
                'vehicle_id': vehicle_id,
                'position': vehicle_info.position,
                'velocity': vehicle_info.velocity,
                'direction': vehicle_info.direction,
                'state': vehicle_info.state.value,
                'resource_class': resource_class.value,
                'training_capability': vehicle_info.training_capability,
                'last_seen': vehicle_info.last_seen,
                'priority_weight': fairness_metrics.priority_weight,
                'contribution_score': fairness_metrics.contribution_score
            })
        
        return vehicles
    
    def set_coverage_area(self, width: float, height: float):
        """Set server coverage area"""
        self.coverage_area = (width, height)
        print(f"Coverage area set to {width}m x {height}m")
    
    def perform_system_maintenance(self):
        """Perform periodic system maintenance"""
        current_time = time.time()
        
        # Remove inactive vehicles
        inactive_vehicles = []
        for vehicle_id, vehicle_info in self.registered_vehicles.items():
            if current_time - vehicle_info.last_seen > 300:  # 5 minutes
                inactive_vehicles.append(vehicle_id)
        
        for vehicle_id in inactive_vehicles:
            self.unregister_vehicle(vehicle_id)
        
        if inactive_vehicles:
            print(f"Removed {len(inactive_vehicles)} inactive vehicles")
        
        # Generate new templates if needed
        self.template_manager._generate_new_templates()
        
        # Clean up old data
        # (Additional cleanup tasks can be added here)