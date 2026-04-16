"""
FHDP Pipeline Coordinator for EVO-1

This module coordinates EVO-1 pipeline training using FHDP's native
coordination system and aggregation capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from core.fhdp_system import FHDPSystem, SystemConfiguration
from core.types import VehicleInfo, VehicleState, TrainingMode, Pipeline, ModelUpdate
from core.hardware_adapter import HardwareAdapter
import logging
from typing import Dict, List, Optional, Any
import time


class FHDPipelineCoordinator:
    """
    Pipeline coordinator for EVO-1 using FHDP's native coordination system
    
    This class leverages FHDP's existing coordination capabilities while
    providing EVO-1 specific pipeline management.
    """
    
    def __init__(self, system_config: Optional[Dict] = None):
        # Initialize FHDP system with EVO-1 specific configuration
        self.fhdp_system = self._initialize_fhdp_system(system_config)
        
        # EVO-1 specific pipeline configuration
        self.evo1_pipeline_config = {
            'encoder_training': True,
            'vlm_backbone_frozen': True,
            'action_head_training': True,
            'resource_aware_selection': True
        }
        
        # Coordination state
        self.active_vehicles = {}
        self.training_pipelines = {}
        self.aggregation_history = []
        
        logging.info("FHDP Pipeline Coordinator initialized for EVO-1")
    
    def _initialize_fhdp_system(self, config: Optional[Dict]) -> FHDPSystem:
        """Initialize FHDP system with EVO-1 specific settings"""
        if config is None:
            config = {}
        
        # Create FHDP system configuration
        system_config = SystemConfiguration(
            max_vehicles_per_region=config.get('max_vehicles', 50),
            pipeline_formation_interval=config.get('pipeline_interval', 5.0),
            model_broadcast_interval=config.get('broadcast_interval', 10.0),
            participation_timeout=config.get('timeout', 30.0),
            aggregation_interval=config.get('aggregation_interval', 15.0),
            enable_pipeline_training=config.get('enable_pipeline', True),
            enable_individual_training=config.get('enable_individual', True),
            fairness_enabled=config.get('fairness_enabled', True)
        )
        
        return FHDPSystem(system_config)
    
    def register_vehicle(self, vehicle_id: str, vehicle_info: Dict[str, Any]) -> bool:
        """Register vehicle using FHDP's native vehicle registration"""
        try:
            # Convert to FHDP VehicleInfo format
            fhdp_vehicle_info = VehicleInfo(
                vehicle_id=vehicle_id,
                position=vehicle_info.get('position', (0.0, 0.0)),
                velocity=vehicle_info.get('velocity', 0.0),
                direction=vehicle_info.get('direction', 0.0),
                resources=self._convert_to_fhdp_resources(vehicle_info.get('resources', {})),
                state=VehicleState.IDLE
            )
            
            # Register with FHDP system
            success = self.fhdp_system.register_vehicle(fhdp_vehicle_info)
            
            if success:
                self.active_vehicles[vehicle_id] = {
                    'fhdp_info': fhdp_vehicle_info,
                    'evo1_config': vehicle_info,
                    'registration_time': time.time()
                }
                logging.info(f"Vehicle registered: {vehicle_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to register vehicle {vehicle_id}: {e}")
            return False
    
    def _convert_to_fhdp_resources(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Convert EVO-1 resources to FHDP format"""
        return {
            'cpu_cores': resources.get('cpu_cores', 4),
            'memory_gb': resources.get('memory_gb', 8),
            'gpu_available': resources.get('gpu_available', False),
            'gpu_memory_gb': resources.get('gpu_memory_gb', 0),
            'storage_gb': resources.get('storage_gb', 100),
            'network_bandwidth': resources.get('network_bandwidth', 100),
            'battery_level': resources.get('battery_level', 100),
            'thermal_state': resources.get('thermal_state', 'normal')
        }
    
    def _is_high_resource_device(self, vehicle_info: VehicleInfo) -> bool:
        """Check if vehicle is a high-resource device (Jetson AGX or Ghost node)"""
        resources = vehicle_info.resources
        
        # Check for Jetson AGX or Ghost node characteristics
        if resources.get('gpu_available', False):
            gpu_memory = resources.get('gpu_memory_gb', 0)
            compute_score = resources.get('compute_score', 0)
            
            # Jetson AGX has >= 16GB GPU memory
            if gpu_memory >= 16:
                return True
            
            # Ghost node has high compute score
            if compute_score >= 0.9:
                return True
        
        return False
    
    def _map_stages_to_vehicles(self, vehicles: List[VehicleInfo]) -> Dict[int, str]:
        """Map pipeline stages to vehicles, with Stage 1 on high-resource device"""
        stage_mapping = {}
        
        # Separate high and low resource vehicles
        high_resource_vehicles = [v for v in vehicles if self._is_high_resource_device(v)]
        other_vehicles = [v for v in vehicles if not self._is_high_resource_device(v)]
        
        # Ensure we have at least one vehicle for Stage 1
        if not high_resource_vehicles:
            # If no high resource vehicles, use the vehicle with highest compute score
            high_resource_vehicles = sorted(vehicles, 
                                         key=lambda v: v.resources.get('compute_score', 0), 
                                         reverse=True)[:1]
        
        # Map Stage 1 to high resource device
        stage_mapping[1] = high_resource_vehicles[0].vehicle_id
        
        # Map remaining stages to other vehicles
        available_vehicles = other_vehicles + high_resource_vehicles[1:]
        for stage in range(2, len(vehicles) + 1):
            if available_vehicles:
                stage_mapping[stage] = available_vehicles.pop(0).vehicle_id
            else:
                # If not enough vehicles, reuse existing ones
                stage_mapping[stage] = high_resource_vehicles[0].vehicle_id
        
        return stage_mapping
    
    def start_pipeline_training(self, vehicle_ids: List[str], 
                             pipeline_config: Optional[Dict] = None) -> bool:
        """Start pipeline training using FHDP's native pipeline formation"""
        try:
            if pipeline_config is None:
                pipeline_config = self.evo1_pipeline_config
            
            # Use FHDP's hybrid participation manager
            selected_vehicles = []
            for vehicle_id in vehicle_ids:
                if vehicle_id in self.active_vehicles:
                    selected_vehicles.append(self.active_vehicles[vehicle_id]['fhdp_info'])
            
            if not selected_vehicles:
                logging.error("No valid vehicles for pipeline training")
                return False
            
            # Map stages to vehicles with Stage 1 on high-resource device
            stage_mapping = self._map_stages_to_vehicles(selected_vehicles)
            
            # Add stage mapping to pipeline config
            pipeline_config['stage_mapping'] = stage_mapping
            
            # Create pipeline using FHDP's native pipeline formation
            pipeline_id = f"evo1_pipeline_{int(time.time())}"
            
            success = self.fhdp_system.hybrid_participation_manager.create_pipeline(
                pipeline_id=pipeline_id,
                vehicles=selected_vehicles,
                training_mode=TrainingMode.PIPELINE,
                config=pipeline_config
            )
            
            if success:
                self.training_pipelines[pipeline_id] = {
                    'vehicles': vehicle_ids,
                    'config': pipeline_config,
                    'start_time': time.time(),
                    'status': 'active',
                    'stage_mapping': stage_mapping
                }
                logging.info(f"Pipeline training started: {pipeline_id}")
                logging.info(f"Stage mapping: {stage_mapping}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to start pipeline training: {e}")
            return False
    
    def collect_model_updates(self, pipeline_id: str) -> List[ModelUpdate]:
        """Collect model updates using FHDP's native aggregation"""
        try:
            if pipeline_id not in self.training_pipelines:
                logging.error(f"Pipeline not found: {pipeline_id}")
                return []
            
            # Use FHDP's asynchronous aggregation manager
            updates = self.fhdp_system.asynchronous_aggregation_manager.collect_updates(
                pipeline_id=pipeline_id
            )
            
            # Convert to EVO-1 format if needed
            evo1_updates = []
            for update in updates:
                evo1_updates.append(self._convert_to_evo1_update(update))
            
            return evo1_updates
            
        except Exception as e:
            logging.error(f"Failed to collect model updates: {e}")
            return []
    
    def _convert_to_evo1_update(self, fhdp_update: ModelUpdate) -> ModelUpdate:
        """Convert FHDP model update to EVO-1 format"""
        # Add EVO-1 specific metadata
        evo1_metadata = fhdp_update.metadata.copy()
        evo1_metadata.update({
            'evo1_pipeline': True,
            'encoder_trained': True,
            'action_head_trained': True,
            'vlm_backbone_frozen': True
        })
        
        return ModelUpdate(
            vehicle_id=fhdp_update.vehicle_id,
            model_state=fhdp_update.model_state,
            optimizer_state=fhdp_update.optimizer_state,
            metadata=evo1_metadata
        )
    
    def aggregate_updates(self, updates: List[ModelUpdate], 
                       aggregation_method: str = 'fedavg') -> Optional[ModelUpdate]:
        """Aggregate model updates using FHDP's native aggregation"""
        try:
            # Use FHDP's aggregation engine
            if aggregation_method == 'fedavg':
                aggregated = self.fhdp_system.asynchronous_aggregation_manager.federated_average(updates)
            else:
                # Default to FedAvg
                aggregated = self.fhdp_system.asynchronous_aggregation_manager.federated_average(updates)
            
            # Add EVO-1 specific metadata
            if aggregated:
                aggregated.metadata.update({
                    'aggregation_method': aggregation_method,
                    'evo1_pipeline': True,
                    'num_contributors': len(updates),
                    'aggregation_time': time.time()
                })
            
            return aggregated
            
        except Exception as e:
            logging.error(f"Failed to aggregate updates: {e}")
            return None
    
    def broadcast_aggregated_model(self, aggregated_update: ModelUpdate, 
                                pipeline_id: str) -> bool:
        """Broadcast aggregated model using FHDP's native broadcasting"""
        try:
            if pipeline_id not in self.training_pipelines:
                logging.error(f"Pipeline not found: {pipeline_id}")
                return False
            
            # Use FHDP's model broadcasting
            success = self.fhdp_system.broadcast_model(
                model_update=aggregated_update,
                target_vehicles=self.training_pipelines[pipeline_id]['vehicles']
            )
            
            if success:
                # Record aggregation in history
                self.aggregation_history.append({
                    'pipeline_id': pipeline_id,
                    'aggregation_time': time.time(),
                    'num_vehicles': len(self.training_pipelines[pipeline_id]['vehicles']),
                    'aggregation_method': aggregated_update.metadata.get('aggregation_method', 'fedavg')
                })
                logging.info(f"Model broadcast successful for pipeline: {pipeline_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to broadcast aggregated model: {e}")
            return False
    
    def get_fairness_metrics(self) -> Dict[str, Any]:
        """Get fairness metrics using FHDP's native fairness management"""
        try:
            # Use FHDP's fairness management system
            fairness_metrics = self.fhdp_system.fairness_manager.get_fairness_metrics()
            
            # Add EVO-1 specific fairness metrics
            evo1_fairness = fairness_metrics.copy()
            evo1_fairness.update({
                'encoder_training_fairness': self._calculate_encoder_fairness(),
                'resource_utilization_fairness': self._calculate_resource_fairness(),
                'pipeline_participation_fairness': self._calculate_pipeline_fairness()
            })
            
            return evo1_fairness
            
        except Exception as e:
            logging.error(f"Failed to get fairness metrics: {e}")
            return {}
    
    def _calculate_encoder_fairness(self) -> float:
        """Calculate fairness of encoder training across vehicles"""
        # Simple fairness calculation based on participation
        total_participation = sum(
            1 for pipeline in self.training_pipelines.values()
            for _ in pipeline['vehicles']
        )
        
        if not self.active_vehicles:
            return 1.0
        
        expected_participation = len(self.active_vehicles)
        fairness = min(total_participation / expected_participation, 1.0)
        
        return fairness
    
    def _calculate_resource_fairness(self) -> float:
        """Calculate resource utilization fairness"""
        if not self.active_vehicles:
            return 1.0
        
        # Get resource utilization from FHDP hardware adapter
        resource_utilizations = []
        for vehicle_data in self.active_vehicles.values():
            resources = vehicle_data['evo1_config'].get('resources', {})
            utilization = resources.get('utilization', 0.5)  # Default
            resource_utilizations.append(utilization)
        
        if not resource_utilizations:
            return 1.0
        
        # Simple fairness metric based on variance
        import numpy as np
        mean_util = np.mean(resource_utilizations)
        std_util = np.std(resource_utilizations)
        
        fairness = max(0.0, 1.0 - (std_util / mean_util) if mean_util > 0 else 1.0)
        return fairness
    
    def _calculate_pipeline_fairness(self) -> float:
        """Calculate pipeline participation fairness"""
        if not self.active_vehicles:
            return 1.0
        
        # Count participation in pipelines
        participation_counts = {vid: 0 for vid in self.active_vehicles.keys()}
        
        for pipeline in self.training_pipelines.values():
            for vehicle_id in pipeline['vehicles']:
                if vehicle_id in participation_counts:
                    participation_counts[vehicle_id] += 1
        
        # Calculate fairness based on participation distribution
        import numpy as np
        counts = list(participation_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        fairness = max(0.0, 1.0 - (std_count / mean_count) if mean_count > 0 else 1.0)
        return fairness
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status using FHDP's native status reporting"""
        fhdp_status = self.fhdp_system.get_system_status()
        
        return {
            'fhdp_system_status': fhdp_status,
            'active_vehicles': len(self.active_vehicles),
            'active_pipelines': len(self.training_pipelines),
            'aggregation_history_count': len(self.aggregation_history),
            'evo1_pipeline_config': self.evo1_pipeline_config,
            'fairness_metrics': self.get_fairness_metrics()
        }
    
    def shutdown_coordinator(self):
        """Shutdown coordinator using FHDP's native shutdown"""
        try:
            # Shutdown all active pipelines
            for pipeline_id in list(self.training_pipelines.keys()):
                self.fhdp_system.hybrid_participation_manager.shutdown_pipeline(pipeline_id)
                del self.training_pipelines[pipeline_id]
            
            # Shutdown FHDP system
            self.fhdp_system.shutdown()
            
            logging.info("FHDP Pipeline Coordinator shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during coordinator shutdown: {e}")