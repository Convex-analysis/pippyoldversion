"""
FHDP System Core Implementation

Implements the core FHDP system with hybrid participation model,
asynchronous aggregation, and coordination between edge server and vehicles.
"""
import time
import threading
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass
import numpy as np
import torch

from .types import (
    VehicleInfo, VehicleState, TrainingMode, Pipeline, PipelineTemplate,
    ModelUpdate, AggregationResult, FairnessMetrics, ResourceClass
)
from .constants import (
    ASYNC_AGGREGATION_INTERVAL, MIN_AGGREGATION_PARTICIPANTS,
    PIPELINE_TIMEOUT, TRAINING_EPOCHS_SHORT
)
from .logging_config import get_logger

# Defer imports to avoid circular dependencies

# Setup logger
logger = get_logger(__name__)

@dataclass
class SystemConfiguration:
    """FHDP system configuration"""
    max_vehicles_per_region: int = 50
    pipeline_formation_interval: float = 5.0  # seconds
    model_broadcast_interval: float = 10.0  # seconds
    participation_timeout: float = 30.0  # seconds
    aggregation_interval: float = ASYNC_AGGREGATION_INTERVAL
    enable_pipeline_training: bool = True
    enable_individual_training: bool = True
    fairness_enabled: bool = True
    default_protocol: str = "dsrc"

class HybridParticipationManager:
    """Manages hybrid participation model (individual + pipeline training)"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.active_vehicles: Dict[str, VehicleInfo] = {}
        self.pipeline_participants: Dict[str, str] = {}  # vehicle_id -> pipeline_id
        self.individual_participants: Set[str] = set()
        
        # Participation decisions
        self.participation_decisions: Dict[str, TrainingMode] = {}
        self.decision_timestamps: Dict[str, float] = {}
        
        # Statistics
        self.participation_stats = {
            'total_participations': 0,
            'individual_participations': 0,
            'pipeline_participations': 0,
            'pipeline_formations': 0,
            'avg_pipeline_length': 0.0
        }
    
    def register_vehicle(self, vehicle_info: VehicleInfo):
        """Register vehicle in the system"""
        self.active_vehicles[vehicle_info.vehicle_id] = vehicle_info
        
        # Clear old participation decisions
        if vehicle_info.vehicle_id in self.participation_decisions:
            age = time.time() - self.decision_timestamps[vehicle_info.vehicle_id]
            if age > self.config.participation_timeout:
                del self.participation_decisions[vehicle_info.vehicle_id]
                del self.decision_timestamps[vehicle_info.vehicle_id]
    
    def unregister_vehicle(self, vehicle_id: str):
        """Unregister vehicle from the system"""
        if vehicle_id in self.active_vehicles:
            del self.active_vehicles[vehicle_id]
        
        if vehicle_id in self.pipeline_participants:
            pipeline_id = self.pipeline_participants[vehicle_id]
            del self.pipeline_participants[vehicle_id]
        
        self.individual_participants.discard(vehicle_id)
        
        if vehicle_id in self.participation_decisions:
            del self.participation_decisions[vehicle_id]
            del self.decision_timestamps[vehicle_id]
    
    def make_participation_decision(self, vehicle_id: str, 
                                   resource_classifier: Any,  # Defer type annotation
                                   template_manager: Any,    # Defer type annotation
                                   neighbors: Dict[str, Any] = None) -> TrainingMode:
        """Make participation decision for vehicle"""
        # Check if already in a pipeline
        if vehicle_id in self.pipeline_participants:
            return TrainingMode.PIPELINE
        
        # Check if already training individually
        if vehicle_id in self.individual_participants:
            return TrainingMode.INDIVIDUAL
        
        # Get vehicle info
        vehicle_info = self.active_vehicles.get(vehicle_id)
        if not vehicle_info:
            return TrainingMode.INDIVIDUAL  # Default to individual
        
        # Classify vehicle resources
        resource_class = resource_classifier.classify_vehicle(vehicle_info)
        
        # Get fairness metrics
        fairness_metrics = resource_classifier.get_fairness_metrics(vehicle_id)
        
        # Decision logic
        decision = self._evaluate_participation_mode(
            vehicle_info, resource_class, fairness_metrics, neighbors
        )
        
        # Record decision
        self.participation_decisions[vehicle_id] = decision
        self.decision_timestamps[vehicle_id] = time.time()
        
        return decision
    
    def _evaluate_participation_mode(self, vehicle_info: VehicleInfo,
                                   resource_class: ResourceClass,
                                   fairness_metrics: FairnessMetrics,
                                   neighbors: Dict[str, Any] = None) -> TrainingMode:
        """Evaluate optimal participation mode"""
        # High-resource vehicles with good connectivity are good pipeline candidates
        if (resource_class == ResourceClass.HIGH and 
            fairness_metrics.priority_weight > 1.5 and
            neighbors and len(neighbors) >= 2):
            return TrainingMode.PIPELINE
        
        # Medium-resource vehicles can participate in pipelines if fairness demands it
        if (resource_class == ResourceClass.MEDIUM and
            fairness_metrics.priority_weight > 2.0 and
            neighbors and len(neighbors) >= 3):
            return TrainingMode.PIPELINE
        
        # Default to individual training
        return TrainingMode.INDIVIDUAL
    
    def assign_pipeline_participation(self, vehicle_id: str, pipeline_id: str):
        """Assign vehicle to pipeline training"""
        self.pipeline_participants[vehicle_id] = pipeline_id
        self.participation_stats['pipeline_participations'] += 1
    
    def assign_individual_participation(self, vehicle_id: str):
        """Assign vehicle to individual training"""
        self.individual_participants.add(vehicle_id)
        self.participation_stats['individual_participations'] += 1
    
    def complete_participation(self, vehicle_id: str, training_mode: TrainingMode):
        """Mark participation as completed"""
        self.participation_stats['total_participations'] += 1
        
        if training_mode == TrainingMode.PIPELINE:
            if vehicle_id in self.pipeline_participants:
                del self.pipeline_participants[vehicle_id]
        else:
            self.individual_participants.discard(vehicle_id)
        
        # Clear participation decision
        if vehicle_id in self.participation_decisions:
            del self.participation_decisions[vehicle_id]
            del self.decision_timestamps[vehicle_id]
    
    def get_pipeline_candidates(self, count: int = 5) -> List[str]:
        """Get candidates for pipeline formation"""
        candidates = []
        
        for vehicle_id, vehicle_info in self.active_vehicles.items():
            # Skip vehicles already participating
            if (vehicle_id in self.pipeline_participants or 
                vehicle_id in self.individual_participants):
                continue
            
            # Get participation decision
            decision = self.participation_decisions.get(vehicle_id)
            if decision == TrainingMode.PIPELINE:
                candidates.append(vehicle_id)
        
        return candidates[:count]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get participation statistics"""
        current_individual = len(self.individual_participants)
        current_pipeline = len(self.pipeline_participants)
        
        return {
            **self.participation_stats,
            'current_individual_participants': current_individual,
            'current_pipeline_participants': current_pipeline,
            'total_active_vehicles': len(self.active_vehicles),
            'pipeline_formation_rate': (
                self.participation_stats['pipeline_formations'] / 
                max(1, self.participation_stats['total_participations'])
            ) * 100
        }

class AsynchronousCoordinationManager:
    """Manages asynchronous coordination between edge server and vehicles"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.coordination_active = False
        self.coordination_thread = None
        
        # Coordination state
        self.last_aggregation = time.time()
        self.last_broadcast = time.time()
        self.pending_updates: List[ModelUpdate] = []
        
        # Callbacks
        self.aggregation_callbacks: List[Callable[[AggregationResult], None]] = []
        self.pipeline_callbacks: List[Callable[[str, List[str]], None]] = []
    
    def start_coordination(self):
        """Start asynchronous coordination"""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        self.coordination_thread = threading.Thread(target=self._coordination_worker, daemon=True)
        self.coordination_thread.start()
    
    def stop_coordination(self):
        """Stop asynchronous coordination"""
        self.coordination_active = False
        
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
    
    def _coordination_worker(self):
        """Coordination worker thread"""
        while self.coordination_active:
            try:
                current_time = time.time()
                
                # Check if aggregation should be triggered
                if (current_time - self.last_aggregation >= self.config.aggregation_interval and
                    len(self.pending_updates) >= MIN_AGGREGATION_PARTICIPANTS):
                    self._trigger_aggregation()
                
                # Check if model broadcast should be triggered
                if current_time - self.last_broadcast >= self.config.model_broadcast_interval:
                    self._trigger_model_broadcast()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Coordination error: {e}")
                time.sleep(1.0)
    
    def submit_update(self, update: ModelUpdate):
        """Submit model update for aggregation"""
        self.pending_updates.append(update)
    
    def _trigger_aggregation(self):
        """Trigger asynchronous aggregation"""
        if not self.pending_updates:
            return
        
        # In real implementation, this would coordinate with the edge server
        # For now, simulate aggregation
        updates_to_aggregate = self.pending_updates.copy()
        self.pending_updates.clear()
        
        # Create mock aggregation result
        aggregated_model = self._mock_aggregate_updates(updates_to_aggregate)
        
        result = AggregationResult(
            global_model=aggregated_model,
            participating_sources=[u.source_id for u in updates_to_aggregate],
            aggregation_weight={u.source_id: 1.0/len(updates_to_aggregate) for u in updates_to_aggregate},
            timestamp=time.time()
        )
        
        # Notify callbacks
        for callback in self.aggregation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Aggregation callback error: {e}")
        
        self.last_aggregation = time.time()
    
    def _mock_aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Mock aggregation (in real implementation would use actual aggregation logic)"""
        if not updates:
            return {"mock_param": torch.zeros(100)}  # Return a dictionary instead of tensor
        
        # Simple averaging
        first_update = updates[0].update_data
        if isinstance(first_update, torch.Tensor):
            # Mock model with multiple parameters
            aggregated = {}
            # Simulate aggregation of multiple model parameters
            aggregated["layer1_weight"] = torch.zeros_like(first_update)
            aggregated["layer1_bias"] = torch.zeros(10)
            aggregated["layer2_weight"] = torch.zeros_like(first_update)
            aggregated["layer2_bias"] = torch.zeros(10)
            return aggregated
        elif isinstance(first_update, dict):
            # If first update is already a dictionary, average each parameter
            aggregated = {}
            for key in first_update.keys():
                # Collect all values for this key from all updates
                param_values = []
                for update in updates:
                    if isinstance(update.update_data, dict) and key in update.update_data:
                        param_values.append(update.update_data[key])
                
                if param_values:
                    # Average the parameter values
                    aggregated[key] = sum(param_values) / len(param_values)
            return aggregated
        else:
            return {"mock_param": torch.zeros(100)}  # Fallback to mock dictionary
    
    def _trigger_model_broadcast(self):
        """Trigger global model broadcast"""
        # In real implementation, this would broadcast the latest global model
        # to all vehicles in the coverage area
        self.last_broadcast = time.time()
    
    def request_pipeline_formation(self, candidates: List[str]):
        """Request pipeline formation"""
        for callback in self.pipeline_callbacks:
            try:
                callback("formation_request", candidates)
            except Exception as e:
                logger.error(f"Pipeline callback error: {e}")
    
    def add_aggregation_callback(self, callback: Callable[[AggregationResult], None]):
        """Add aggregation callback"""
        self.aggregation_callbacks.append(callback)
    
    def add_pipeline_callback(self, callback: Callable[[str, List[str]], None]):
        """Add pipeline callback"""
        self.pipeline_callbacks.append(callback)

class FHDPSystem:
    """Main FHDP system implementation"""
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # Import edge server components
        from ..edge_server import (
            MobilityPredictor, TemplateManager, AsynchronousAggregator, ResourceClassifier
        )
        
        # Core components
        self.mobility_predictor = MobilityPredictor()
        self.template_manager = TemplateManager()
        self.aggregator = AsynchronousAggregator()
        self.resource_classifier = ResourceClassifier()
        
        # Management layers
        self.participation_manager = HybridParticipationManager(self.config)
        self.coordination_manager = AsynchronousCoordinationManager(self.config)
        
        # Active entities
        self.registered_vehicles: Dict[str, VehicleInfo] = {}
        self.active_pipelines: Dict[str, Pipeline] = {}
        self.vehicle_managers: Dict[str, Dict[str, Any]] = {}  # vehicle_id -> manager components
        
        # System state
        self.system_active = False
        self.global_model: Optional[torch.Tensor] = None
        self.round_number = 0
        
        # Statistics
        self.system_stats = {
            'total_rounds': 0,
            'total_aggregations': 0,
            'total_pipelines_formed': 0,
            'total_vehicles_served': 0,
            'system_uptime': 0.0
        }
    
    def start_system(self):
        """Start FHDP system"""
        if self.system_active:
            return
        
        # Start core components
        self.coordination_manager.start_coordination()
        
        # Set up coordination callbacks
        self.coordination_manager.add_aggregation_callback(self._handle_aggregation_result)
        self.coordination_manager.add_pipeline_callback(self._handle_pipeline_request)
        
        self.system_active = True
        self.start_time = time.time()
        
        logger.info("FHDP system started")
    
    def stop_system(self):
        """Stop FHDP system"""
        if not self.system_active:
            return
        
        self.coordination_manager.stop_coordination()
        
        # Stop vehicle managers
        for vehicle_components in self.vehicle_managers.values():
            if 'communication' in vehicle_components:
                vehicle_components['communication'].shutdown()
            if 'training' in vehicle_components:
                vehicle_components['training'].stop_execution_service()
            if 'monitor' in vehicle_components:
                vehicle_components['monitor'].stop_monitoring()
        
        self.system_active = False
        self.system_stats['system_uptime'] = time.time() - self.start_time
        
        logger.info("FHDP system stopped")
    
    def register_vehicle(self, vehicle_info: VehicleInfo) -> bool:
        """Register vehicle with FHDP system"""
        try:
            # Register with participation manager
            self.participation_manager.register_vehicle(vehicle_info)
            
            # Create vehicle components
            self._create_vehicle_manager(vehicle_info)
            
            self.registered_vehicles[vehicle_info.vehicle_id] = vehicle_info
            self.system_stats['total_vehicles_served'] += 1
            
            return True
            
        except Exception as e:
            print(f"Failed to register vehicle {vehicle_info.vehicle_id}: {e}")
            return False
    
    def unregister_vehicle(self, vehicle_id: str):
        """Unregister vehicle from FHDP system"""
        if vehicle_id not in self.registered_vehicles:
            return
        
        # Clean up vehicle components
        vehicle_components = self.vehicle_managers.get(vehicle_id)
        if vehicle_components:
            if 'communication' in vehicle_components:
                vehicle_components['communication'].shutdown()
            if 'training' in vehicle_components:
                vehicle_components['training'].stop_execution_service()
            if 'monitor' in vehicle_components:
                vehicle_components['monitor'].stop_monitoring()
            
            del self.vehicle_managers[vehicle_id]
        
        # Remove from participation manager
        self.participation_manager.unregister_vehicle(vehicle_id)
        
        del self.registered_vehicles[vehicle_id]
    
    def _create_vehicle_manager(self, vehicle_info: VehicleInfo):
        """Create management components for vehicle"""
        from ..vehicle_layer import (
            V2VCommunicationManager, PipelineFormation, TrainingExecutor, VehicleMonitor
        )
        
        # Initialize communication manager
        communication = V2VCommunicationManager(vehicle_info)
        communication.initialize([self.config.default_protocol])
        
        # Initialize pipeline formation
        pipeline_formation = PipelineFormation(vehicle_info)
        
        # Initialize training executor
        training = TrainingExecutor(vehicle_info)
        training.start_execution_service()
        
        # Initialize vehicle monitor
        monitor = VehicleMonitor(vehicle_info)
        monitor.start_monitoring()
        
        self.vehicle_managers[vehicle_info.vehicle_id] = {
            'communication': communication,
            'pipeline_formation': pipeline_formation,
            'training': training,
            'monitor': monitor
        }
    
    def _handle_aggregation_result(self, result: AggregationResult):
        """Handle aggregation result from coordination manager"""
        self.global_model = result.global_model
        self.system_stats['total_aggregations'] += 1
        
        # Broadcast new model to vehicles
        self._broadcast_model_to_vehicles(result)
        
        # Update round number
        self.round_number += 1
    
    def _handle_pipeline_request(self, request_type: str, candidates: List[str]):
        """Handle pipeline formation request"""
        if request_type == "formation_request":
            self._initiate_pipeline_formation(candidates)
    
    def _initiate_pipeline_formation(self, candidates: List[str]):
        """Initiate pipeline formation with candidates"""
        # Get best candidates based on resources and fairness
        selected_candidates = self._select_pipeline_candidates(candidates)
        
        if len(selected_candidates) >= MIN_AGGREGATION_PARTICIPANTS:
            # Find suitable template
            candidate_vehicles = [self.registered_vehicles[vid] for vid in selected_candidates]
            template = self.template_manager.find_template_for_vehicles(candidate_vehicles)
            
            if template:
                # Initiate pipeline formation through candidate vehicles
                for i, vehicle_id in enumerate(selected_candidates):
                    vehicle_components = self.vehicle_managers.get(vehicle_id)
                    if vehicle_components and 'pipeline_formation' in vehicle_components:
                        # Notify vehicle about pipeline opportunity
                        # (In real implementation, this would send actual messages)
                        pass
                
                self.system_stats['total_pipelines_formed'] += 1
    
    def _select_pipeline_candidates(self, candidates: List[str]) -> List[str]:
        """Select best candidates for pipeline formation"""
        scored_candidates = []
        
        for vehicle_id in candidates:
            if vehicle_id not in self.registered_vehicles:
                continue
            
            vehicle_info = self.registered_vehicles[vehicle_id]
            
            # Calculate selection score
            resource_class = self.resource_classifier.classify_vehicle(vehicle_info)
            fairness_metrics = self.resource_classifier.get_fairness_metrics(vehicle_id)
            
            # Score factors
            resource_score = {
                ResourceClass.HIGH: 3.0,
                ResourceClass.MEDIUM: 2.0,
                ResourceClass.LOW: 1.0
            }.get(resource_class, 1.0)
            
            fairness_score = fairness_metrics.priority_weight
            mobility_score = self.mobility_predictor.get_mobility_features(vehicle_id).get('mobility_regularity', 0.5)
            
            total_score = resource_score * 0.4 + fairness_score * 0.4 + mobility_score * 0.2
            
            scored_candidates.append((vehicle_id, total_score))
        
        # Sort by score and return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [vid for vid, _ in scored_candidates[:5]]  # Top 5 candidates
    
    def _broadcast_model_to_vehicles(self, result: AggregationResult):
        """Broadcast aggregated model to vehicles"""
        for vehicle_id, vehicle_components in self.vehicle_managers.items():
            if 'communication' in vehicle_components:
                # In real implementation, this would send the actual model
                communication = vehicle_components['communication']
                # communication.broadcast_model_update(result.global_model)
    
    def submit_model_update(self, vehicle_id: str, update: ModelUpdate):
        """Submit model update from vehicle"""
        # Get fairness metrics
        fairness_metrics = self.resource_classifier.get_fairness_metrics(vehicle_id)
        
        # Submit to aggregator
        self.aggregator.submit_update(update, fairness_metrics)
        
        # Submit to coordination manager
        self.coordination_manager.submit_update(update)
    
    def process_vehicle_arrival(self, vehicle_info: VehicleInfo):
        """Process new vehicle arrival"""
        # Update mobility prediction
        self.mobility_predictor.update_vehicle_mobility(vehicle_info)
        
        # Register vehicle
        if self.register_vehicle(vehicle_info):
            # Make participation decision
            decision = self.participation_manager.make_participation_decision(
                vehicle_info.vehicle_id,
                self.resource_classifier,
                self.template_manager
            )
            
            # Notify vehicle about participation mode
            self._notify_participation_mode(vehicle_info.vehicle_id, decision)
    
    def process_vehicle_departure(self, vehicle_id: str):
        """Process vehicle departure"""
        # Unregister vehicle
        self.unregister_vehicle(vehicle_id)
        
        # Check if any active pipelines are affected
        affected_pipelines = [
            pid for pid, pipeline in self.active_pipelines.items()
            if vehicle_id in pipeline.vehicles
        ]
        
        for pipeline_id in affected_pipelines:
            self._handle_pipeline_disruption(pipeline_id, vehicle_id)
    
    def _handle_pipeline_disruption(self, pipeline_id: str, departed_vehicle: str):
        """Handle pipeline disruption due to vehicle departure"""
        pipeline = self.active_pipelines.get(pipeline_id)
        if not pipeline:
            return
        
        # Remove departed vehicle
        if departed_vehicle in pipeline.vehicles:
            pipeline.vehicles.remove(departed_vehicle)
        
        # Check if pipeline can continue
        if len(pipeline.vehicles) < MIN_AGGREGATION_PARTICIPANTS:
            # Dissolve pipeline
            del self.active_pipelines[pipeline_id]
            
            # Notify remaining vehicles
            for vehicle_id in pipeline.vehicles:
                self._notify_pipeline_dissolution(vehicle_id, pipeline_id)
        else:
            # Attempt pipeline reformation
            self._attempt_pipeline_reformation(pipeline_id)
    
    def _notify_participation_mode(self, vehicle_id: str, mode: TrainingMode):
        """Notify vehicle about participation mode"""
        vehicle_components = self.vehicle_managers.get(vehicle_id)
        if not vehicle_components:
            return
        
        # In real implementation, this would send actual messages
        print(f"Notifying {vehicle_id} to participate in {mode.value} mode")
    
    def _notify_pipeline_dissolution(self, vehicle_id: str, pipeline_id: str):
        """Notify vehicle about pipeline dissolution"""
        # In real implementation, this would send actual messages
        print(f"Notifying {vehicle_id} about pipeline {pipeline_id} dissolution")
    
    def _attempt_pipeline_reformation(self, pipeline_id: str):
        """Attempt to reform disrupted pipeline"""
        # In real implementation, this would coordinate with remaining vehicles
        # to attempt pipeline reformation with replacement vehicles
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        uptime = current_time - self.start_time if hasattr(self, 'start_time') else 0
        
        return {
            'system_active': self.system_active,
            'uptime': uptime,
            'round_number': self.round_number,
            'registered_vehicles': len(self.registered_vehicles),
            'active_pipelines': len(self.active_pipelines),
            'participation_stats': self.participation_manager.get_statistics(),
            'aggregation_stats': self.aggregator.get_aggregation_statistics(),
            'template_stats': self.template_manager.get_template_statistics(),
            'classification_stats': self.resource_classifier.get_classification_statistics(),
            **self.system_stats
        }