"""
Resource Monitoring and Participation Tracking for FHDP System

Monitors vehicle resources, tracks participation history, and provides
resource-aware decision making for optimal training participation.
"""
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics

from ..core.types import (
    VehicleInfo, VehicleState, TrainingMode, ResourceMetrics,
    FairnessMetrics
)
from ..core.fairness_error import ParticipationRecord
from ..core.constants import (
    MONITORING_INTERVAL, MAX_VEHICLE_MEMORY_USAGE,
    FAIRNESS_WINDOW_SIZE, FAIRNESS_DECAY_FACTOR,
    MIN_PARTICIPATION_INTERVAL
)

@dataclass
class ResourceSnapshot:
    """Snapshot of vehicle resources at a specific time"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    battery_level: float
    network_quality: float
    thermal_state: float
    available_storage: float  # GB
    
@dataclass  
class ParticipationRecord:
    """Record of training participation"""
    timestamp: float
    training_mode: TrainingMode
    task_id: str
    pipeline_id: Optional[str]
    duration: float
    success: bool
    contribution_score: float
    resources_used: ResourceMetrics

class ResourceMonitor:
    """Monitors vehicle resources in real-time"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.resource_history: deque = deque(maxlen=1000)
        self.baseline_resources: Optional[ResourceSnapshot] = None
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.monitoring_active = False
        
        # Resource thresholds
        self.thresholds = {
            'cpu_critical': 0.9,
            'memory_critical': 0.9,
            'battery_critical': 0.15,
            'thermal_critical': 0.85,
            'storage_critical': 0.1  # 10% free storage
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, ResourceSnapshot], None]] = []
        
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.stop_event.clear()
        self.monitoring_active = True
        
        # Establish baseline
        self._establish_baseline()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        if not self.monitoring_active:
            return
        
        self.stop_event.set()
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _establish_baseline(self):
        """Establish baseline resource measurements"""
        snapshot = self._take_resource_snapshot()
        self.baseline_resources = snapshot
    
    def _monitoring_worker(self):
        """Resource monitoring worker thread"""
        while not self.stop_event.is_set():
            try:
                # Take resource snapshot
                snapshot = self._take_resource_snapshot()
                self.resource_history.append(snapshot)
                
                # Check for critical conditions
                self._check_alert_conditions(snapshot)
                
                time.sleep(MONITORING_INTERVAL)
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(MONITORING_INTERVAL)
    
    def _take_resource_snapshot(self) -> ResourceSnapshot:
        """Take current resource snapshot"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.used / memory.total
            
            # Battery level (if available)
            try:
                battery = psutil.sensors_battery()
                battery_level = battery.percent / 100.0 if battery else self.vehicle_info.resources.get('battery', 0.8)
            except:
                battery_level = self.vehicle_info.resources.get('battery', 0.8)
            
            # Network quality (estimate based on system metrics)
            network_quality = self._estimate_network_quality()
            
            # Thermal state (estimate)
            thermal_state = self._estimate_thermal_state()
            
            # Available storage
            storage = psutil.disk_usage('/')
            available_storage = storage.free / (1024**3)  # GB
            
            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                battery_level=battery_level,
                network_quality=network_quality,
                thermal_state=thermal_state,
                available_storage=available_storage
            )
            
        except Exception as e:
            print(f"Error taking resource snapshot: {e}")
            # Return fallback values
            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_usage=0.5,
                memory_usage=0.5,
                battery_level=self.vehicle_info.resources.get('battery', 0.8),
                network_quality=0.8,
                thermal_state=0.3,
                available_storage=10.0
            )
    
    def _estimate_network_quality(self) -> float:
        """Estimate network quality (placeholder implementation)"""
        # In real implementation, this would measure actual network conditions
        # For now, use vehicle's network resource
        return self.vehicle_info.resources.get('network_quality', 0.8)
    
    def _estimate_thermal_state(self) -> float:
        """Estimate thermal state (placeholder implementation)"""
        # In real implementation, this would read temperature sensors
        # For now, estimate based on CPU usage
        if self.resource_history:
            recent_cpu = [s.cpu_usage for s in list(self.resource_history)[-5:]]
            avg_cpu = statistics.mean(recent_cpu) if recent_cpu else 0.5
            return min(1.0, avg_cpu * 1.2)  # Simple thermal model
        return 0.3
    
    def _check_alert_conditions(self, snapshot: ResourceSnapshot):
        """Check for critical resource conditions"""
        alerts = []
        
        if snapshot.cpu_usage > self.thresholds['cpu_critical']:
            alerts.append('cpu_critical')
        
        if snapshot.memory_usage > self.thresholds['memory_critical']:
            alerts.append('memory_critical')
        
        if snapshot.battery_level < self.thresholds['battery_critical']:
            alerts.append('battery_critical')
        
        if snapshot.thermal_state > self.thresholds['thermal_critical']:
            alerts.append('thermal_critical')
        
        if snapshot.available_storage < self.thresholds['storage_critical']:
            alerts.append('storage_critical')
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, snapshot)
                except Exception as e:
                    print(f"Alert callback error: {e}")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        if not self.resource_history:
            return ResourceMetrics(0.5, 0.5, 0.8, 0.8, 0.3)
        
        latest = self.resource_history[-1]
        return ResourceMetrics(
            cpu_usage=latest.cpu_usage,
            memory_usage=latest.memory_usage,
            battery_level=latest.battery_level,
            network_quality=latest.network_quality,
            thermal_state=latest.thermal_state
        )
    
    def predict_resources(self, horizon: float = 10.0) -> ResourceMetrics:
        """Predict future resource availability"""
        if len(self.resource_history) < 5:
            return self.get_current_metrics()
        
        # Use linear trend prediction
        recent_snapshots = list(self.resource_history)[-5:]
        
        # Calculate trends
        cpu_trend = self._calculate_trend([s.cpu_usage for s in recent_snapshots])
        memory_trend = self._calculate_trend([s.memory_usage for s in recent_snapshots])
        battery_trend = self._calculate_trend([s.battery_level for s in recent_snapshots])
        thermal_trend = self._calculate_trend([s.thermal_state for s in recent_snapshots])
        
        # Predict future values
        steps_ahead = int(horizon / MONITORING_INTERVAL)
        current = self.get_current_metrics()
        
        predicted_cpu = max(0.0, min(1.0, current.cpu_usage + cpu_trend * steps_ahead))
        predicted_memory = max(0.0, min(1.0, current.memory_usage + memory_trend * steps_ahead))
        predicted_battery = max(0.0, min(1.0, current.battery_level + battery_trend * steps_ahead))
        predicted_thermal = max(0.0, min(1.0, current.thermal_state + thermal_trend * steps_ahead))
        
        return ResourceMetrics(
            cpu_usage=predicted_cpu,
            memory_usage=predicted_memory,
            battery_level=predicted_battery,
            network_quality=current.network_quality,  # Network is harder to predict
            thermal_state=predicted_thermal
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend for a list of values"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def can_participate(self, task_requirements: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """Determine if vehicle can participate in training"""
        current = self.get_current_metrics()
        
        # Basic resource checks
        if current.cpu_usage > self.thresholds['cpu_critical']:
            return False, "CPU usage too high"
        
        if current.memory_usage > MAX_VEHICLE_MEMORY_USAGE:
            return False, "Memory usage too high"
        
        if current.battery_level < self.thresholds['battery_critical']:
            return False, "Battery level too low"
        
        if current.thermal_state > self.thresholds['thermal_critical']:
            return False, "Thermal state too high"
        
        # Task-specific requirements
        if task_requirements:
            if 'min_cpu' in task_requirements:
                available_cpu = 1.0 - current.cpu_usage
                if available_cpu < task_requirements['min_cpu']:
                    return False, f"Insufficient CPU: need {task_requirements['min_cpu']}, have {available_cpu}"
            
            if 'min_memory' in task_requirements:
                available_memory = 1.0 - current.memory_usage
                if available_memory < task_requirements['min_memory']:
                    return False, f"Insufficient memory: need {task_requirements['min_memory']}, have {available_memory}"
            
            if 'min_battery' in task_requirements:
                if current.battery_level < task_requirements['min_battery']:
                    return False, f"Insufficient battery: need {task_requirements['min_battery']}, have {current.battery_level}"
        
        return True, "Can participate"
    
    def add_alert_callback(self, callback: Callable[[str, ResourceSnapshot], None]):
        """Add alert callback for resource monitoring"""
        self.alert_callbacks.append(callback)

class ParticipationTracker:
    """Tracks training participation history and fairness metrics"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.participation_history: deque = deque(maxlen=FAIRNESS_WINDOW_SIZE)
        self.contribution_history: deque = deque(maxlen=FAIRNESS_WINDOW_SIZE)
        
        # Fairness metrics
        self.fairness_metrics = FairnessMetrics(
            vehicle_id=vehicle_info.vehicle_id,
            participation_count=0,
            last_participation=0.0,
            contribution_score=1.0,
            priority_weight=1.0
        )
        
        # Participation statistics
        self.stats = {
            'total_participations': 0,
            'successful_participations': 0,
            'failed_participations': 0,
            'individual_participations': 0,
            'pipeline_participations': 0,
            'avg_participation_duration': 0.0,
            'total_training_time': 0.0
        }
        
    def record_participation(self, training_mode: TrainingMode, task_id: str,
                          pipeline_id: Optional[str], duration: float,
                          success: bool, contribution_score: float = 1.0):
        """Record training participation"""
        record = ParticipationRecord(
            timestamp=time.time(),
            training_mode=training_mode,
            task_id=task_id,
            pipeline_id=pipeline_id,
            duration=duration,
            success=success,
            contribution_score=contribution_score,
            resources_used=ResourceMetrics(0, 0, 0, 0, 0)  # Would be filled in real implementation
        )
        
        self.participation_history.append(record)
        
        # Update statistics
        self.stats['total_participations'] += 1
        self.stats['total_training_time'] += duration
        
        if success:
            self.stats['successful_participations'] += 1
        else:
            self.stats['failed_participations'] += 1
        
        if training_mode == TrainingMode.INDIVIDUAL:
            self.stats['individual_participations'] += 1
        else:
            self.stats['pipeline_participations'] += 1
        
        # Update average duration
        total_successful = self.stats['successful_participations']
        old_avg = self.stats['avg_participation_duration']
        self.stats['avg_participation_duration'] = (
            (old_avg * (total_successful - 1) + duration) / total_successful
        )
        
        # Update fairness metrics
        self._update_fairness_metrics(record)
    
    def _update_fairness_metrics(self, record: ParticipationRecord):
        """Update fairness metrics after participation"""
        current_time = time.time()
        
        # Update participation count
        self.fairness_metrics.participation_count = len(self.participation_history)
        self.fairness_metrics.last_participation = current_time
        
        # Update contribution score with exponential moving average
        alpha = FAIRNESS_DECAY_FACTOR
        if self.fairness_metrics.contribution_score == 1.0:  # First participation
            self.fairness_metrics.contribution_score = record.contribution_score
        else:
            self.fairness_metrics.contribution_score = (
                (1 - alpha) * self.fairness_metrics.contribution_score + 
                alpha * record.contribution_score
            )
        
        # Calculate priority weight
        self._calculate_priority_weight()
    
    def _calculate_priority_weight(self):
        """Calculate priority weight for fair participation"""
        current_time = time.time()
        
        # Time since last participation
        time_since_last = current_time - self.fairness_metrics.last_participation
        recency_factor = min(3.0, 1.0 + time_since_last / MIN_PARTICIPATION_INTERVAL)
        
        # Participation frequency (avoid over-participation)
        if self.participation_history:
            time_span = self.participation_history[-1].timestamp - self.participation_history[0].timestamp
            frequency = len(self.participation_history) / max(1.0, time_span)  # participations per second
            frequency_factor = max(0.1, 1.0 - frequency / 0.01)  # Penalize high frequency
        else:
            frequency_factor = 1.0
        
        # Success rate factor
        if self.stats['total_participations'] > 0:
            success_rate = self.stats['successful_participations'] / self.stats['total_participations']
            success_factor = 0.5 + 0.5 * success_rate  # Range: 0.5-1.0
        else:
            success_factor = 0.5
        
        # Calculate final priority weight
        self.fairness_metrics.priority_weight = (
            recency_factor * frequency_factor * success_factor * 
            self.fairness_metrics.contribution_score
        )
    
    def should_participate(self, min_interval: float = MIN_PARTICIPATION_INTERVAL) -> bool:
        """Determine if vehicle should participate based on fairness"""
        current_time = time.time()
        
        # Check minimum interval
        if self.fairness_metrics.last_participation > 0:
            time_since_last = current_time - self.fairness_metrics.last_participation
            if time_since_last < min_interval:
                return False
        
        # Check resource availability
        # (This would integrate with ResourceMonitor in real implementation)
        return True
    
    def get_participation_statistics(self) -> Dict[str, any]:
        """Get participation statistics"""
        current_time = time.time()
        
        # Calculate recent participation rate
        recent_participations = [
            p for p in self.participation_history 
            if current_time - p.timestamp < 3600  # Last hour
        ]
        recent_rate = len(recent_participations) / max(1.0, 3600.0)  # per second
        
        return {
            **self.stats,
            'fairness_metrics': {
                'participation_count': self.fairness_metrics.participation_count,
                'last_participation': self.fairness_metrics.last_participation,
                'contribution_score': self.fairness_metrics.contribution_score,
                'priority_weight': self.fairness_metrics.priority_weight
            },
            'recent_participation_rate': recent_rate,
            'success_rate': (
                self.stats['successful_participations'] / 
                max(1, self.stats['total_participations'])
            ) * 100,
            'pipeline_ratio': (
                self.stats['pipeline_participations'] / 
                max(1, self.stats['total_participations'])
            ) * 100
        }

class VehicleMonitor:
    """Combined monitoring system for vehicle resources and participation"""
    
    def __init__(self, vehicle_info: VehicleInfo):
        self.vehicle_info = vehicle_info
        self.resource_monitor = ResourceMonitor(vehicle_info)
        self.participation_tracker = ParticipationTracker(vehicle_info)
        
        # Monitoring state
        self.monitoring_active = False
        
        # Alert handlers
        self.alert_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.monitoring_active:
            return
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Set up alert handlers
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        self.monitoring_active = True
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring"""
        if not self.monitoring_active:
            return
        
        self.resource_monitor.stop_monitoring()
        self.monitoring_active = False
    
    def _handle_resource_alert(self, alert_type: str, snapshot: ResourceSnapshot):
        """Handle resource alerts"""
        for handler in self.alert_handlers.get(alert_type, []):
            try:
                handler(alert_type, snapshot)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def add_alert_handler(self, alert_type: str, handler: Callable):
        """Add alert handler for specific alert type"""
        self.alert_handlers[alert_type].append(handler)
    
    def can_participate_in_training(self, task_requirements: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """Check if vehicle can participate in training"""
        # Check resource availability
        can_participate, reason = self.resource_monitor.can_participate(task_requirements)
        if not can_participate:
            return False, f"Resource constraint: {reason}"
        
        # Check fairness constraints
        if not self.participation_tracker.should_participate():
            return False, "Fairness constraint: Recent participation"
        
        return True, "Can participate"
    
    def record_training_participation(self, training_mode: TrainingMode, task_id: str,
                                     pipeline_id: Optional[str], duration: float,
                                     success: bool, contribution_score: float = 1.0):
        """Record training participation"""
        self.participation_tracker.record_participation(
            training_mode, task_id, pipeline_id, duration, success, contribution_score
        )
    
    def get_vehicle_status(self) -> Dict[str, any]:
        """Get comprehensive vehicle status"""
        resource_metrics = self.resource_monitor.get_current_metrics()
        predicted_metrics = self.resource_monitor.predict_resources()
        
        return {
            'vehicle_id': self.vehicle_info.vehicle_id,
            'state': self.vehicle_info.state.value,
            'current_resources': {
                'cpu_usage': resource_metrics.cpu_usage,
                'memory_usage': resource_metrics.memory_usage,
                'battery_level': resource_metrics.battery_level,
                'network_quality': resource_metrics.network_quality,
                'thermal_state': resource_metrics.thermal_state
            },
            'predicted_resources': {
                'cpu_usage': predicted_metrics.cpu_usage,
                'memory_usage': predicted_metrics.memory_usage,
                'battery_level': predicted_metrics.battery_level,
                'network_quality': predicted_metrics.network_quality,
                'thermal_state': predicted_metrics.thermal_state
            },
            'participation_stats': self.participation_tracker.get_participation_statistics(),
            'fairness_metrics': {
                'priority_weight': self.participation_tracker.fairness_metrics.priority_weight,
                'contribution_score': self.participation_tracker.fairness_metrics.contribution_score,
                'last_participation': self.participation_tracker.fairness_metrics.last_participation
            }
        }