"""
Pipeline Formation Algorithm with Greedy Selection for FHDP System

Implements greedy pipeline formation algorithm to create efficient vehicle
pipelines for collaborative training based on resource capabilities,
mobility patterns, and communication quality.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import heapq

from ..core.types import (
    VehicleInfo, Pipeline, PipelineTemplate, ResourceClass,
    TrainingMode, MobilityPrediction, NeighborInfo
)
from ..core.constants import (
    MAX_PIPELINE_LENGTH, MIN_PIPELINE_PARTICIPANTS,
    PIPELINE_RECOMPOSITION_TIME, PIPELINE_TIMEOUT,
    MAX_NEIGHBOR_DISTANCE
)

@dataclass
class PipelineCandidate:
    """Candidate vehicle for pipeline formation"""
    vehicle_id: str
    vehicle_info: VehicleInfo
    resource_score: float  # 0.0-1.0
    mobility_score: float  # 0.0-1.0  
    communication_score: float  # 0.0-1.0
    overall_score: float  # 0.0-1.0
    expected_contribution: float
    position_in_pipeline: int = -1

@dataclass
class PipelineState:
    """State of pipeline formation process"""
    template: PipelineTemplate
    required_positions: int
    filled_positions: int = 0
    selected_candidates: List[PipelineCandidate] = field(default_factory=list)
    rejected_candidates: Set[str] = field(default_factory=set)
    formation_start_time: float = field(default_factory=time.time)
    expected_completion: float = 0.0

class GreedySelector:
    """Greedy selection algorithm for pipeline formation"""
    
    def __init__(self):
        self.selection_history: Dict[str, List[float]] = defaultdict(list)
        self.bias_correction_factor = 0.1  # To prevent bias toward frequent participants
        
    def calculate_resource_score(self, vehicle_info: VehicleInfo, 
                                required_class: ResourceClass) -> float:
        """Calculate resource matching score"""
        # Extract vehicle resources
        cpu_available = 1.0 - vehicle_info.resources.get('cpu_usage', 0.5)
        memory_available = 1.0 - vehicle_info.resources.get('memory_usage', 0.5)
        battery_level = vehicle_info.resources.get('battery', 0.7)
        
        # Calculate match score based on required class
        if required_class == ResourceClass.HIGH:
            required_cpu = 0.8
            required_memory = 0.8
            required_battery = 0.7
        elif required_class == ResourceClass.MEDIUM:
            required_cpu = 0.5
            required_memory = 0.5
            required_battery = 0.4
        else:  # LOW
            required_cpu = 0.2
            required_memory = 0.2
            required_battery = 0.2
        
        # Calculate individual scores
        cpu_score = min(1.0, cpu_available / required_cpu) if required_cpu > 0 else 1.0
        memory_score = min(1.0, memory_available / required_memory) if required_memory > 0 else 1.0
        battery_score = min(1.0, battery_level / required_battery) if required_battery > 0 else 1.0
        
        # Weighted combination
        resource_score = (cpu_score * 0.4 + memory_score * 0.3 + battery_score * 0.3)
        
        # Apply fairness correction
        participation_history = self.selection_history.get(vehicle_info.vehicle_id, [])
        if participation_history:
            recent_participation = np.mean(participation_history[-5:])  # Last 5 selections
            fairness_correction = 1.0 - (recent_participation * self.bias_correction_factor)
            resource_score *= fairness_correction
        
        return min(1.0, resource_score)
    
    def calculate_mobility_score(self, vehicle_info: VehicleInfo, 
                               mobility_predictions: Optional[List[MobilityPrediction]] = None) -> float:
        """Calculate mobility stability score"""
        base_mobility_score = 0.7  # Base score
        
        # Factor in velocity (moderate velocity is better)
        velocity = vehicle_info.velocity
        if velocity < 5.0:  # Too slow, might be stuck
            velocity_factor = 0.8
        elif velocity > 30.0:  # Too fast, might leave quickly
            velocity_factor = 0.7
        else:  # Optimal range
            velocity_factor = 1.0
        
        # Factor in direction changes (fewer changes are better)
        # In real implementation, this would use historical data
        direction_stability = 0.8  # Placeholder
        
        # Factor in mobility predictions if available
        prediction_confidence = 0.0
        if mobility_predictions:
            prediction_confidence = np.mean([pred.confidence for pred in mobility_predictions])
        
        mobility_score = (base_mobility_score * 0.4 + 
                         velocity_factor * 0.3 + 
                         direction_stability * 0.2 + 
                         prediction_confidence * 0.1)
        
        return mobility_score
    
    def calculate_communication_score(self, vehicle_info: VehicleInfo,
                                    neighbor_info: Optional[NeighborInfo] = None) -> float:
        """Calculate communication quality score"""
        if neighbor_info:
            # Use actual neighbor communication quality
            return neighbor_info.connection_quality
        
        # Estimate based on network resources
        network_quality = vehicle_info.resources.get('network_quality', 0.8)
        signal_strength = vehicle_info.resources.get('signal_strength', -70.0)
        
        # Convert signal strength to score (higher is better)
        signal_score = max(0.0, (signal_strength + 100.0) / 30.0)  # -70 to -100 dBm range
        
        communication_score = (network_quality * 0.6 + signal_score * 0.4)
        return communication_score
    
    def select_best_candidate(self, candidates: List[PipelineCandidate], 
                             position_index: int, template: PipelineTemplate) -> Optional[PipelineCandidate]:
        """Select best candidate for specific position using greedy selection"""
        if not candidates:
            return None
        
        # Calculate position-specific weights
        position_weights = self._get_position_weights(position_index, len(template.resource_requirements))
        
        # Re-score candidates based on position requirements
        scored_candidates = []
        for candidate in candidates:
            # Adjust scores based on position importance
            adjusted_score = (
                candidate.resource_score * position_weights['resource'] +
                candidate.mobility_score * position_weights['mobility'] +
                candidate.communication_score * position_weights['communication']
            )
            
            # Create new candidate with adjusted score
            adjusted_candidate = PipelineCandidate(
                vehicle_id=candidate.vehicle_id,
                vehicle_info=candidate.vehicle_info,
                resource_score=candidate.resource_score,
                mobility_score=candidate.mobility_score,
                communication_score=candidate.communication_score,
                overall_score=adjusted_score,
                expected_contribution=candidate.expected_contribution,
                position_in_pipeline=position_index
            )
            
            scored_candidates.append(adjusted_candidate)
        
        # Select highest scoring candidate (greedy)
        best_candidate = max(scored_candidates, key=lambda x: x.overall_score)
        
        # Update selection history
        self.selection_history[best_candidate.vehicle_id].append(1.0)
        if len(self.selection_history[best_candidate.vehicle_id]) > 100:
            self.selection_history[best_candidate.vehicle_id].pop(0)
        
        return best_candidate
    
    def _get_position_weights(self, position: int, total_positions: int) -> Dict[str, float]:
        """Get importance weights for pipeline position"""
        # First and last positions are more critical
        if position == 0 or position == total_positions - 1:
            return {
                'resource': 0.5,
                'mobility': 0.3,
                'communication': 0.2
            }
        else:
            return {
                'resource': 0.4,
                'mobility': 0.4,
                'communication': 0.2
            }

class PipelineFormation:
    """Main pipeline formation orchestrator"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.greedy_selector = GreedySelector()
        
        # Pipeline state management
        self.active_formations: Dict[str, PipelineState] = {}  # pipeline_id -> PipelineState
        self.active_pipelines: Dict[str, Pipeline] = {}  # pipeline_id -> Pipeline
        
        # Formation statistics
        self.formation_stats = {
            'successful_formations': 0,
            'failed_formations': 0,
            'avg_formation_time': 0.0,
            'avg_pipeline_length': 0.0
        }
        
    def initiate_pipeline_formation(self, template: PipelineTemplate, 
                                  candidate_vehicles: List[VehicleInfo],
                                  mobility_predictions: Dict[str, List[MobilityPrediction]] = None,
                                  neighbor_info: Dict[str, NeighborInfo] = None) -> Optional[str]:
        """Initiate pipeline formation process"""
        pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        
        # Create pipeline state
        pipeline_state = PipelineState(
            template=template,
            required_positions=len(template.resource_requirements),
            expected_completion=time.time() + template.expected_duration
        )
        
        # Create candidate list
        candidates = self._create_candidates(candidate_vehicles, template, 
                                            mobility_predictions, neighbor_info)
        
        # Perform greedy selection
        selected_candidates = self._perform_greedy_selection(candidates, template)
        
        if len(selected_candidates) == template.resource_requirements:
            # Successfully formed pipeline
            pipeline_state.selected_candidates = selected_candidates
            pipeline_state.filled_positions = len(selected_candidates)
            
            # Create pipeline object
            pipeline = Pipeline(
                pipeline_id=pipeline_id,
                template_id=template.template_id,
                vehicles=[c.vehicle_id for c in selected_candidates],
                stages=[f"stage_{i}" for i in range(len(selected_candidates))],
                current_stage=0,
                start_time=time.time(),
                expected_completion=pipeline_state.expected_completion,
                communication_overhead=0
            )
            
            self.active_formations[pipeline_id] = pipeline_state
            self.active_pipelines[pipeline_id] = pipeline
            
            # Update statistics
            self._update_formation_stats(True, time.time() - pipeline_state.formation_start_time)
            
            return pipeline_id
        else:
            # Failed to form pipeline
            self._update_formation_stats(False, time.time() - pipeline_state.formation_start_time)
            return None
    
    def _create_candidates(self, candidate_vehicles: List[VehicleInfo], 
                          template: PipelineTemplate,
                          mobility_predictions: Dict[str, List[MobilityPrediction]] = None,
                          neighbor_info: Dict[str, NeighborInfo] = None) -> List[PipelineCandidate]:
        """Create candidate pool with scores"""
        candidates = []
        
        for i, vehicle in enumerate(candidate_vehicles):
            if vehicle.vehicle_id == self.vehicle_info.vehicle_id:
                continue  # Skip self
            
            # Calculate resource score based on template requirements
            required_class = template.resource_requirements[i] if i < len(template.resource_requirements) else ResourceClass.MEDIUM
            resource_score = self.greedy_selector.calculate_resource_score(vehicle, required_class)
            
            # Calculate mobility score
            vehicle_predictions = mobility_predictions.get(vehicle.vehicle_id, []) if mobility_predictions else []
            mobility_score = self.greedy_selector.calculate_mobility_score(vehicle, vehicle_predictions)
            
            # Calculate communication score
            vehicle_neighbor_info = neighbor_info.get(vehicle.vehicle_id) if neighbor_info else None
            communication_score = self.greedy_selector.calculate_communication_score(vehicle, vehicle_neighbor_info)
            
            # Calculate overall score
            overall_score = (resource_score * 0.4 + mobility_score * 0.3 + communication_score * 0.3)
            
            # Expected contribution based on capabilities
            expected_contribution = overall_score * vehicle.training_capability
            
            candidate = PipelineCandidate(
                vehicle_id=vehicle.vehicle_id,
                vehicle_info=vehicle,
                resource_score=resource_score,
                mobility_score=mobility_score,
                communication_score=communication_score,
                overall_score=overall_score,
                expected_contribution=expected_contribution
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _perform_greedy_selection(self, candidates: List[PipelineCandidate], 
                                template: PipelineTemplate) -> List[PipelineCandidate]:
        """Perform greedy selection of candidates"""
        selected = []
        remaining_candidates = candidates.copy()
        
        for position in range(len(template.resource_requirements)):
            # Select best candidate for this position
            best_candidate = self.greedy_selector.select_best_candidate(
                remaining_candidates, position, template
            )
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                # No suitable candidate found
                break
        
        return selected
    
    def respond_to_pipeline_invitation(self, pipeline_id: str, template: PipelineTemplate,
                                      position_in_pipeline: int, accept: bool) -> bool:
        """Respond to pipeline formation invitation"""
        if not accept:
            return False
        
        # In a real implementation, this would involve coordination with other vehicles
        # For now, we assume successful acceptance
        
        pipeline_state = PipelineState(
            template=template,
            required_positions=len(template.resource_requirements),
            filled_positions=1,  # This vehicle
            selected_candidates=[],  # Would be populated by coordinator
            formation_start_time=time.time(),
            expected_completion=time.time() + template.expected_duration
        )
        
        self.active_formations[pipeline_id] = pipeline_state
        
        return True
    
    def check_pipeline_integrity(self, pipeline_id: str) -> bool:
        """Check if pipeline is still intact"""
        if pipeline_id not in self.active_pipelines:
            return False
        
        pipeline = self.active_pipelines[pipeline_id]
        current_time = time.time()
        
        # Check timeout
        if current_time - pipeline.start_time > PIPELINE_TIMEOUT:
            self._dissolve_pipeline(pipeline_id)
            return False
        
        # In a real implementation, this would check if all vehicles are still reachable
        return True
    
    def reorganize_pipeline(self, pipeline_id: str, available_vehicles: List[VehicleInfo],
                          mobility_predictions: Dict[str, List[MobilityPrediction]] = None,
                          neighbor_info: Dict[str, NeighborInfo] = None) -> Optional[str]:
        """Reorganize pipeline with new vehicles if needed"""
        if pipeline_id not in self.active_pipelines:
            return None
        
        old_pipeline = self.active_pipelines[pipeline_id]
        template = self._get_template_by_id(old_pipeline.template_id)
        
        if not template:
            return None
        
        # Attempt to form new pipeline
        new_pipeline_id = self.initiate_pipeline_formation(
            template, available_vehicles, mobility_predictions, neighbor_info
        )
        
        if new_pipeline_id:
            # Dissolve old pipeline
            self._dissolve_pipeline(pipeline_id)
            return new_pipeline_id
        
        return None
    
    def _get_template_by_id(self, template_id: str) -> Optional[PipelineTemplate]:
        """Get template by ID (placeholder - would interface with edge server)"""
        # In real implementation, this would fetch from edge server
        return None
    
    def _dissolve_pipeline(self, pipeline_id: str):
        """Dissolve pipeline and clean up resources"""
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]
        
        if pipeline_id in self.active_formations:
            del self.active_formations[pipeline_id]
    
    def _update_formation_stats(self, success: bool, formation_time: float):
        """Update formation statistics"""
        if success:
            self.formation_stats['successful_formations'] += 1
        else:
            self.formation_stats['failed_formations'] += 1
        
        total_formations = self.formation_stats['successful_formations'] + self.formation_stats['failed_formations']
        
        # Update average formation time
        old_avg = self.formation_stats['avg_formation_time']
        self.formation_stats['avg_formation_time'] = (
            (old_avg * (total_formations - 1) + formation_time) / total_formations
        )
    
    def get_active_pipelines(self) -> Dict[str, Pipeline]:
        """Get all active pipelines"""
        return self.active_pipelines.copy()
    
    def get_pipeline_statistics(self) -> Dict[str, any]:
        """Get pipeline formation statistics"""
        total_formations = (self.formation_stats['successful_formations'] + 
                          self.formation_stats['failed_formations'])
        
        success_rate = (self.formation_stats['successful_formations'] / max(1, total_formations)) * 100
        
        avg_length = 0.0
        if self.active_pipelines:
            avg_length = np.mean([len(p.vehicles) for p in self.active_pipelines.values()])
        
        return {
            **self.formation_stats,
            'success_rate_percent': success_rate,
            'active_pipelines': len(self.active_pipelines),
            'avg_active_pipeline_length': avg_length
        }
    
    def should_reorganize_pipeline(self, pipeline_id: str) -> bool:
        """Determine if pipeline should be reorganized"""
        if pipeline_id not in self.active_formations:
            return False
        
        pipeline_state = self.active_formations[pipeline_id]
        current_time = time.time()
        
        # Check if formation is taking too long
        if current_time - pipeline_state.formation_start_time > PIPELINE_RECOMPOSITION_TIME:
            return True
        
        # Check if pipeline is near completion but has missing participants
        pipeline = self.active_pipelines.get(pipeline_id)
        if pipeline and len(pipeline.vehicles) < len(pipeline_state.template.resource_requirements):
            return True
        
        return False