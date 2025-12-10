"""
Resource Classification Service for FHDP System

Classifies vehicles based on their computational and communication resources
to optimize pipeline formation and task assignment.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import threading

from ..core.types import (
    VehicleInfo, ResourceClass, ResourceMetrics, FairnessMetrics,
    VehicleState, TrainingMode
)
from ..core.constants import (
    HIGH_RESOURCE_CPU, HIGH_RESOURCE_MEMORY, HIGH_RESOURCE_BATTERY,
    MEDIUM_RESOURCE_CPU, MEDIUM_RESOURCE_MEMORY, MEDIUM_RESOURCE_BATTERY,
    FAIRNESS_WINDOW_SIZE, FAIRNESS_DECAY_FACTOR, 
    MIN_PARTICIPATION_INTERVAL, MAX_PRIORITY_WEIGHT
)

@dataclass
class ResourceProfile:
    """Detailed resource profile for a vehicle"""
    vehicle_id: str
    cpu_capacity: float  # GHz
    memory_capacity: float  # GB
    battery_capacity: float  # Wh
    network_bandwidth: float  # Mbps
    compute_score: float  # 0.0-1.0
    communication_score: float  # 0.0-1.0
    reliability_score: float  # 0.0-1.0
    last_updated: float
    
class ResourceMonitor:
    """Monitors and tracks vehicle resource utilization"""
    
    def __init__(self):
        self.resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.baseline_resources: Dict[str, ResourceMetrics] = {}
        self.trend_analyzer = ResourceTrendAnalyzer()
        
    def update_vehicle_resources(self, vehicle_id: str, metrics: ResourceMetrics):
        """Update resource metrics for a vehicle"""
        timestamp = time.time()
        
        # Store in history
        self.resource_history[vehicle_id].append((timestamp, metrics))
        
        # Update baseline if not exists
        if vehicle_id not in self.baseline_resources:
            self.baseline_resources[vehicle_id] = metrics
        
        # Analyze trends
        self.trend_analyzer.analyze_trends(vehicle_id, self.resource_history[vehicle_id])
    
    def get_current_resources(self, vehicle_id: str) -> Optional[ResourceMetrics]:
        """Get current resource metrics for a vehicle"""
        if vehicle_id not in self.resource_history or not self.resource_history[vehicle_id]:
            return None
        
        return self.resource_history[vehicle_id][-1][1]
    
    def get_resource_stability(self, vehicle_id: str) -> float:
        """Calculate resource stability (0.0-1.0)"""
        if vehicle_id not in self.resource_history:
            return 0.5  # Unknown stability
        
        history = list(self.resource_history[vehicle_id])
        if len(history) < 5:
            return 0.5
        
        # Calculate variance in resource usage
        cpu_values = [metrics.cpu_usage for _, metrics in history]
        memory_values = [metrics.memory_usage for _, metrics in history]
        
        cpu_variance = np.var(cpu_values)
        memory_variance = np.var(memory_values)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + cpu_variance + memory_variance)
        return min(1.0, stability)
    
    def predict_resources(self, vehicle_id: str, horizon: float = 10.0) -> Optional[ResourceMetrics]:
        """Predict future resource availability"""
        if vehicle_id not in self.resource_history:
            return None
        
        history = list(self.resource_history[vehicle_id])
        if len(history) < 3:
            return history[-1][1] if history else None
        
        # Simple linear trend prediction
        recent_metrics = history[-3:]
        
        # Extract trends
        cpu_trend = np.polyfit(range(len(recent_metrics)), 
                              [m.cpu_usage for _, m in recent_metrics], 1)[0]
        memory_trend = np.polyfit(range(len(recent_metrics)), 
                                 [m.memory_usage for _, m in recent_metrics], 1)[0]
        battery_trend = np.polyfit(range(len(recent_metrics)), 
                                  [m.battery_level for _, m in recent_metrics], 1)[0]
        
        # Predict future values
        steps_ahead = int(horizon / 1.0)  # Assuming 1-second intervals
        current = recent_metrics[-1][1]
        
        predicted_cpu = max(0.0, min(1.0, current.cpu_usage + cpu_trend * steps_ahead))
        predicted_memory = max(0.0, min(1.0, current.memory_usage + memory_trend * steps_ahead))
        predicted_battery = max(0.0, min(1.0, current.battery_level + battery_trend * steps_ahead))
        
        # Network quality is harder to predict, use current
        predicted_network = current.network_quality
        predicted_thermal = current.thermal_state
        
        return ResourceMetrics(
            cpu_usage=predicted_cpu,
            memory_usage=predicted_memory,
            battery_level=predicted_battery,
            network_quality=predicted_network,
            thermal_state=predicted_thermal
        )

class ResourceTrendAnalyzer:
    """Analyzes resource usage trends for prediction"""
    
    def __init__(self):
        self.trends: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def analyze_trends(self, vehicle_id: str, history: deque):
        """Analyze resource usage trends"""
        if len(history) < 5:
            return
        
        # Convert to list for easier manipulation
        metrics_list = [(t, m) for t, m in history]
        timestamps = [t for t, _ in metrics_list]
        
        # Normalize timestamps
        if len(timestamps) > 1:
            start_time = timestamps[0]
            normalized_times = [(t - start_time) for t in timestamps]
        else:
            normalized_times = [0]
        
        # Extract metric values
        cpu_values = [m.cpu_usage for _, m in metrics_list]
        memory_values = [m.memory_usage for _, m in metrics_list]
        battery_values = [m.battery_level for _, m in metrics_list]
        
        # Calculate trends (slopes)
        if len(normalized_times) > 1 and len(set(normalized_times)) > 1:
            cpu_trend = np.polyfit(normalized_times, cpu_values, 1)[0]
            memory_trend = np.polyfit(normalized_times, memory_values, 1)[0]
            battery_trend = np.polyfit(normalized_times, battery_values, 1)[0]
        else:
            cpu_trend = memory_trend = battery_trend = 0.0
        
        self.trends[vehicle_id] = {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'battery_trend': battery_trend,
            'last_analysis': time.time()
        }
    
    def get_trend(self, vehicle_id: str, resource_type: str) -> float:
        """Get trend for specific resource type"""
        return self.trends.get(vehicle_id, {}).get(f'{resource_type}_trend', 0.0)

class FairnessManager:
    """Manages participation fairness across vehicles"""
    
    def __init__(self):
        self.participation_history: Dict[str, List[float]] = defaultdict(list)
        self.contribution_scores: Dict[str, float] = defaultdict(float)
        self.priority_weights: Dict[str, float] = defaultdict(float)
        self.last_participation: Dict[str, float] = {}
        
    def record_participation(self, vehicle_id: str, contribution_score: float = 1.0):
        """Record vehicle participation and update fairness metrics"""
        current_time = time.time()
        
        # Update participation history
        self.participation_history[vehicle_id].append(current_time)
        
        # Maintain window size
        if len(self.participation_history[vehicle_id]) > FAIRNESS_WINDOW_SIZE:
            self.participation_history[vehicle_id].pop(0)
        
        # Update contribution score with exponential moving average
        if vehicle_id not in self.contribution_scores:
            self.contribution_scores[vehicle_id] = contribution_score
        else:
            alpha = 0.1
            self.contribution_scores[vehicle_id] = (
                (1 - alpha) * self.contribution_scores[vehicle_id] + alpha * contribution_score
            )
        
        # Update last participation time
        self.last_participation[vehicle_id] = current_time
        
        # Recalculate priority weight
        self._update_priority_weight(vehicle_id)
    
    def _update_priority_weight(self, vehicle_id: str):
        """Update priority weight for a vehicle"""
        current_time = time.time()
        
        # Time since last participation
        time_since_last = current_time - self.last_participation.get(vehicle_id, 0)
        recency_factor = min(3.0, 1.0 + time_since_last / MIN_PARTICIPATION_INTERVAL)
        
        # Participation frequency
        participation_times = self.participation_history.get(vehicle_id, [])
        if len(participation_times) > 1:
            time_span = participation_times[-1] - participation_times[0]
            frequency = len(participation_times) / max(1.0, time_span)
            frequency_factor = max(0.1, 1.0 - frequency / 10.0)  # Penalize over-participation
        else:
            frequency_factor = 1.0
        
        # Contribution quality
        contribution_factor = self.contribution_scores.get(vehicle_id, 1.0)
        
        # Calculate final priority weight
        priority_weight = recency_factor * frequency_factor * contribution_factor
        self.priority_weights[vehicle_id] = min(MAX_PRIORITY_WEIGHT, priority_weight)
    
    def get_fairness_metrics(self, vehicle_id: str) -> FairnessMetrics:
        """Get fairness metrics for a vehicle"""
        participation_times = self.participation_history.get(vehicle_id, [])
        
        return FairnessMetrics(
            vehicle_id=vehicle_id,
            participation_count=len(participation_times),
            last_participation=self.last_participation.get(vehicle_id, 0.0),
            contribution_score=self.contribution_scores.get(vehicle_id, 1.0),
            priority_weight=self.priority_weights.get(vehicle_id, 1.0)
        )
    
    def should_participate(self, vehicle_id: str, min_interval: float = MIN_PARTICIPATION_INTERVAL) -> bool:
        """Determine if vehicle should participate based on fairness"""
        current_time = time.time()
        last_time = self.last_participation.get(vehicle_id, 0.0)
        
        return (current_time - last_time) >= min_interval

class ResourceClassifier:
    """Main resource classification service"""
    
    def __init__(self):
        self.monitor = ResourceMonitor()
        self.fairness_manager = FairnessManager()
        self.profiles: Dict[str, ResourceProfile] = {}
        self.classification_cache: Dict[str, Tuple[ResourceClass, float]] = {}
        self.cache_timeout = 5.0  # seconds
        
    def classify_vehicle(self, vehicle_info: VehicleInfo) -> ResourceClass:
        """Classify vehicle based on current resources"""
        current_metrics = self.monitor.get_current_resources(vehicle_info.vehicle_id)
        
        if not current_metrics:
            # Use vehicle resources if no monitoring data
            current_metrics = ResourceMetrics(
                cpu_usage=1.0 - vehicle_info.resources.get('cpu', 0.5),
                memory_usage=1.0 - vehicle_info.resources.get('memory', 0.5),
                battery_level=vehicle_info.resources.get('battery', 0.7),
                network_quality=vehicle_info.resources.get('network', 0.8),
                thermal_state=vehicle_info.resources.get('thermal', 0.5)
            )
        
        # Classification logic
        cpu_available = 1.0 - current_metrics.cpu_usage
        memory_available = 1.0 - current_metrics.memory_usage
        battery_available = current_metrics.battery_level
        
        # Add stability factor
        stability = self.monitor.get_resource_stability(vehicle_info.vehicle_id)
        
        # Adjust thresholds based on stability
        stability_bonus = stability * 0.1
        
        if (cpu_available >= HIGH_RESOURCE_CPU - stability_bonus and 
            memory_available >= HIGH_RESOURCE_MEMORY - stability_bonus and 
            battery_available >= HIGH_RESOURCE_BATTERY):
            return ResourceClass.HIGH
        elif (cpu_available >= MEDIUM_RESOURCE_CPU - stability_bonus and 
              memory_available >= MEDIUM_RESOURCE_MEMORY - stability_bonus and 
              battery_available >= MEDIUM_RESOURCE_BATTERY):
            return ResourceClass.MEDIUM
        else:
            return ResourceClass.LOW
    
    def update_vehicle_resources(self, vehicle_id: str, metrics: ResourceMetrics):
        """Update resource monitoring for a vehicle"""
        self.monitor.update_vehicle_resources(vehicle_id, metrics)
        
        # Invalidate cache
        if vehicle_id in self.classification_cache:
            del self.classification_cache[vehicle_id]
    
    def create_resource_profile(self, vehicle_info: VehicleInfo) -> ResourceProfile:
        """Create comprehensive resource profile for a vehicle"""
        current_metrics = self.monitor.get_current_resources(vehicle_info.vehicle_id)
        
        if not current_metrics:
            current_metrics = ResourceMetrics(
                cpu_usage=1.0 - vehicle_info.resources.get('cpu', 0.5),
                memory_usage=1.0 - vehicle_info.resources.get('memory', 0.5),
                battery_level=vehicle_info.resources.get('battery', 0.7),
                network_quality=vehicle_info.resources.get('network', 0.8),
                thermal_state=vehicle_info.resources.get('thermal', 0.5)
            )
        
        # Calculate scores
        compute_score = self._calculate_compute_score(current_metrics)
        communication_score = self._calculate_communication_score(current_metrics)
        reliability_score = self._calculate_reliability_score(vehicle_info.vehicle_id)
        
        profile = ResourceProfile(
            vehicle_id=vehicle_info.vehicle_id,
            cpu_capacity=vehicle_info.resources.get('cpu_capacity', 2.0),
            memory_capacity=vehicle_info.resources.get('memory_capacity', 8.0),
            battery_capacity=vehicle_info.resources.get('battery_capacity', 50.0),
            network_bandwidth=vehicle_info.resources.get('bandwidth', 100.0),
            compute_score=compute_score,
            communication_score=communication_score,
            reliability_score=reliability_score,
            last_updated=time.time()
        )
        
        self.profiles[vehicle_info.vehicle_id] = profile
        return profile
    
    def _calculate_compute_score(self, metrics: ResourceMetrics) -> float:
        """Calculate compute capability score"""
        cpu_score = 1.0 - metrics.cpu_usage
        memory_score = 1.0 - metrics.memory_usage
        thermal_score = 1.0 - metrics.thermal_state
        
        # Weighted combination
        return (cpu_score * 0.5 + memory_score * 0.3 + thermal_score * 0.2)
    
    def _calculate_communication_score(self, metrics: ResourceMetrics) -> float:
        """Calculate communication capability score"""
        network_score = metrics.network_quality
        battery_score = metrics.battery_level
        
        return (network_score * 0.7 + battery_score * 0.3)
    
    def _calculate_reliability_score(self, vehicle_id: str) -> float:
        """Calculate vehicle reliability score"""
        stability = self.monitor.get_resource_stability(vehicle_id)
        
        # Factor in participation history
        fairness_metrics = self.fairness_manager.get_fairness_metrics(vehicle_id)
        contribution_factor = fairness_metrics.contribution_score
        
        return (stability * 0.6 + contribution_factor * 0.4)
    
    def record_training_participation(self, vehicle_id: str, training_mode: TrainingMode, 
                                    success: bool, contribution_score: float = 1.0):
        """Record training participation for fairness management"""
        # Adjust contribution score based on training mode and success
        adjusted_score = contribution_score
        if training_mode == TrainingMode.PIPELINE:
            adjusted_score *= 1.1  # Bonus for pipeline participation
        
        if success:
            adjusted_score *= 1.0
        else:
            adjusted_score *= 0.7  # Penalty for failure
        
        self.fairness_manager.record_participation(vehicle_id, adjusted_score)
    
    def predict_vehicle_suitability(self, vehicle_id: str, task_requirements: Dict[str, float],
                                  horizon: float = 10.0) -> float:
        """Predict vehicle suitability for a task"""
        profile = self.profiles.get(vehicle_id)
        if not profile:
            return 0.0
        
        # Predict future resources
        predicted_metrics = self.monitor.predict_resources(vehicle_id, horizon)
        if not predicted_metrics:
            return profile.compute_score * profile.communication_score * profile.reliability_score
        
        # Calculate future scores
        future_compute = self._calculate_compute_score(predicted_metrics)
        future_communication = self._calculate_communication_score(predicted_metrics)
        
        # Factor in task requirements
        compute_match = min(1.0, future_compute / task_requirements.get('compute', 0.5))
        comm_match = min(1.0, future_communication / task_requirements.get('communication', 0.5))
        
        return (compute_match * 0.4 + comm_match * 0.3 + profile.reliability_score * 0.3)
    
    def get_fairness_metrics(self, vehicle_id: str) -> FairnessMetrics:
        """Get fairness metrics for a vehicle"""
        return self.fairness_manager.get_fairness_metrics(vehicle_id)
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification system statistics"""
        classifications = [self.classify_vehicle(
            VehicleInfo(vid, (0, 0), 0, 0, {})
        ) for vid in self.profiles.keys()]
        
        return {
            'total_vehicles': len(self.profiles),
            'high_resource': classifications.count(ResourceClass.HIGH),
            'medium_resource': classifications.count(ResourceClass.MEDIUM),
            'low_resource': classifications.count(ResourceClass.LOW),
            'avg_reliability': np.mean([p.reliability_score for p in self.profiles.values()]) if self.profiles else 0.0,
            'cache_size': len(self.classification_cache)
        }