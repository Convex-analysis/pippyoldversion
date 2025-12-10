"""
Heterogeneous Resource Management for FHDP

Manages resources across different hardware platforms including Jetson Orin Nano,
x86 PCs, and other heterogeneous computing devices with adaptive monitoring
and task allocation strategies.
"""
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
import queue
from collections import deque, defaultdict

from .hardware_adapter import (
    HardwarePlatform, HardwareCapabilities, ComputeCapability,
    HardwareDetector, ResourceAdapter
)
from .types import ResourceMetrics, ResourceClass, FairnessMetrics

class ResourceUtilizationMode(Enum):
    """Resource utilization modes"""
    CONSERVATIVE = "conservative"  # Prioritize stability
    BALANCED = "balanced"  # Equal priority
    AGGRESSIVE = "aggressive"  # Prioritize performance
    ADAPTIVE = "adaptive"  # Dynamically adjust

class TaskComplexity(Enum):
    """Task complexity levels"""
    LIGHT = "light"  # Simple inference, small models
    MEDIUM = "medium"  # Training on small datasets
    HEAVY = "heavy"  # Large model training, complex computations
    EXTREME = "extreme"  # Maximum computational tasks

@dataclass
class ComputeWorkload:
    """Compute workload specification"""
    workload_id: str
    complexity: TaskComplexity
    cpu_requirement: float  # 0.0-1.0 of available CPU
    memory_requirement: float  # 0.0-1.0 of available memory
    gpu_requirement: float  # 0.0-1.0 of available GPU
    duration_estimate: float  # seconds
    priority: int  # Higher = more important
    platform_preferences: List[HardwarePlatform] = field(default_factory=list)
    
@dataclass
class ResourceAllocation:
    """Resource allocation result"""
    workload_id: str
    allocated_platform: HardwarePlatform
    allocated_resources: Dict[str, float]
    estimated_completion: float
    confidence_score: float  # 0.0-1.0
    allocation_strategy: str

@dataclass
class PlatformPerformance:
    """Platform performance metrics"""
    platform: HardwarePlatform
    cpu_efficiency: float  # Performance per watt
    memory_efficiency: float
    gpu_efficiency: float
    thermal_efficiency: float  # How well it handles heat
    power_efficiency: float  # Overall power efficiency
    reliability_score: float  # Historical reliability
    
class PerformanceProfiler:
    """Profiles and tracks platform performance"""
    
    def __init__(self):
        self.performance_history: Dict[HardwarePlatform, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.platform_benchmarks: Dict[HardwarePlatform, PlatformPerformance] = {}
        self.benchmark_cache: Dict[str, float] = {}
        self.profiler_active = False
        self.profiler_thread = None
        
    def start_profiling(self, capabilities: HardwareCapabilities):
        """Start performance profiling"""
        if self.profiler_active:
            return
            
        self.profiler_active = True
        self.capabilities = capabilities
        
        self.profiler_thread = threading.Thread(target=self._profiling_worker, daemon=True)
        self.profiler_thread.start()
    
    def stop_profiling(self):
        """Stop performance profiling"""
        self.profiler_active = False
        if self.profiler_thread:
            self.profiler_thread.join(timeout=2.0)
    
    def _profiling_worker(self):
        """Continuous performance profiling worker"""
        while self.profiler_active:
            try:
                performance = self._measure_current_performance()
                self.performance_history[self.capabilities.platform].append(performance)
                time.sleep(5.0)  # Profile every 5 seconds
            except Exception as e:
                print(f"Performance profiling error: {e}")
                time.sleep(5.0)
    
    def _measure_current_performance(self) -> Dict[str, float]:
        """Measure current platform performance"""
        start_time = time.time()
        
        # CPU benchmark (simple computation)
        cpu_start = time.time()
        result = sum(i * i for i in range(10000))
        cpu_time = time.time() - cpu_start
        
        # Memory benchmark
        mem_start = time.time()
        test_data = [i for i in range(100000)]
        memory_time = time.time() - mem_start
        
        # Get current resource usage
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.used / memory.total
        
        # Temperature (if available)
        temperature = self._get_temperature()
        
        return {
            'cpu_benchmark': cpu_time,
            'memory_benchmark': memory_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'temperature': temperature,
            'timestamp': start_time,
            'power_estimate': self._estimate_power_usage(cpu_usage, temperature)
        }
    
    def _get_temperature(self) -> float:
        """Get platform temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries and entries[0].current:
                        return entries[0].current
        except:
            pass
        return 45.0  # Default estimate
    
    def _estimate_power_usage(self, cpu_usage: float, temperature: float) -> float:
        """Estimate power usage based on CPU and temperature"""
        # Simple power model
        base_power = 10.0  # Base power in watts
        cpu_power = cpu_usage * 20.0  # CPU power contribution
        thermal_power = max(0, temperature - 30) * 0.5  # Cooling power
        
        return base_power + cpu_power + thermal_power
    
    def get_platform_performance(self, platform: HardwarePlatform) -> Optional[PlatformPerformance]:
        """Get platform performance summary"""
        history = list(self.performance_history[platform])
        if len(history) < 10:
            return None
        
        # Calculate metrics
        cpu_times = [h['cpu_benchmark'] for h in history]
        memory_times = [h['memory_benchmark'] for h in history]
        cpu_usages = [h['cpu_usage'] for h in history]
        memory_usages = [h['memory_usage'] for h in history]
        temperatures = [h['temperature'] for h in history]
        power_usage = [h['power_estimate'] for h in history]
        
        # CPU efficiency (inverse of benchmark time)
        cpu_efficiency = 1.0 / (statistics.mean(cpu_times) + 0.001)
        memory_efficiency = 1.0 / (statistics.mean(memory_times) + 0.001)
        
        # GPU efficiency (if available)
        gpu_efficiency = self._get_gpu_efficiency(platform)
        
        # Thermal efficiency (lower temperature is better)
        thermal_efficiency = max(0, 1.0 - (statistics.mean(temperatures) - 25) / 60)
        
        # Power efficiency (performance per watt)
        avg_power = statistics.mean(power_usage)
        power_efficiency = cpu_efficiency / avg_power if avg_power > 0 else 0
        
        # Reliability (based on variance in performance)
        cpu_variance = statistics.variance(cpu_times) if len(cpu_times) > 1 else 0
        reliability = max(0, 1.0 - cpu_variance / (statistics.mean(cpu_times) ** 2))
        
        return PlatformPerformance(
            platform=platform,
            cpu_efficiency=cpu_efficiency,
            memory_efficiency=memory_efficiency,
            gpu_efficiency=gpu_efficiency,
            thermal_efficiency=thermal_efficiency,
            power_efficiency=power_efficiency,
            reliability_score=reliability
        )
    
    def _get_gpu_efficiency(self, platform: HardwarePlatform) -> float:
        """Get GPU efficiency for platform"""
        try:
            if platform in [HardwarePlatform.JETSON_ORIN, HardwarePlatform.JETSON_XAVIER]:
                # Jetson devices have efficient GPUs
                return 0.8
            elif platform == HardwarePlatform.JETSON_NANO:
                return 0.4  # Nano GPU is less powerful
            else:
                # x86 platforms - check for NVIDIA GPU
                result = psutil.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_util = float(result.stdout.strip())
                    return gpu_util / 100.0
                return 0.0
        except:
            return 0.0

class AdaptiveResourceMonitor:
    """Adaptive resource monitoring for heterogeneous platforms"""
    
    def __init__(self, capabilities: HardwareCapabilities):
        self.capabilities = capabilities
        self.resource_adapter = ResourceAdapter()
        self.performance_profiler = PerformanceProfiler()
        self.monitoring_mode = ResourceUtilizationMode.ADAPTIVE
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Resource history
        self.resource_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        self.allocation_history: deque = deque(maxlen=100)
        
        # Adaptive thresholds
        self.adaptive_thresholds = self._initialize_thresholds()
        self.threshold_update_interval = 60.0  # seconds
        self.last_threshold_update = 0.0
        
        # Alert callbacks
        self.alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    def start_monitoring(self):
        """Start adaptive resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start performance profiling
        self.performance_profiler.start_profiling(self.capabilities)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.performance_profiler.stop_profiling()
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize platform-specific thresholds"""
        platform = self.capabilities.platform
        
        if platform == HardwarePlatform.JETSON_ORIN:
            return {
                'cpu_warning': 0.8,
                'cpu_critical': 0.95,
                'memory_warning': 0.75,
                'memory_critical': 0.9,
                'temperature_warning': 70.0,
                'temperature_critical': 85.0,
                'power_warning': 25.0,
                'power_critical': 30.0
            }
        elif platform == HardwarePlatform.JETSON_NANO:
            return {
                'cpu_warning': 0.7,
                'cpu_critical': 0.9,
                'memory_warning': 0.8,
                'memory_critical': 0.95,
                'temperature_warning': 60.0,
                'temperature_critical': 75.0,
                'power_warning': 10.0,
                'power_critical': 15.0
            }
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
            return {
                'cpu_warning': 0.85,
                'cpu_critical': 0.95,
                'memory_warning': 0.8,
                'memory_critical': 0.9,
                'temperature_warning': 75.0,
                'temperature_critical': 90.0,
                'power_warning': 100.0,
                'power_critical': 150.0
            }
        else:
            return {
                'cpu_warning': 0.7,
                'cpu_critical': 0.9,
                'memory_warning': 0.75,
                'memory_critical': 0.9,
                'temperature_warning': 65.0,
                'temperature_critical': 80.0,
                'power_warning': 20.0,
                'power_critical': 30.0
            }
    
    def _monitoring_worker(self):
        """Main monitoring worker"""
        while self.monitoring_active:
            try:
                # Get current metrics
                current_metrics = self.resource_adapter.get_current_metrics()
                current_time = time.time()
                
                # Store in history
                self.resource_history.append({
                    'metrics': current_metrics,
                    'timestamp': current_time
                })
                
                # Get performance data
                performance = self.performance_profiler._measure_current_performance()
                self.performance_history.append({
                    'performance': performance,
                    'timestamp': current_time
                })
                
                # Check for alerts
                self._check_alerts(current_metrics, performance)
                
                # Update adaptive thresholds periodically
                if current_time - self.last_threshold_update > self.threshold_update_interval:
                    self._update_adaptive_thresholds()
                    self.last_threshold_update = current_time
                
                # Adjust monitoring mode based on conditions
                self._adjust_monitoring_mode(current_metrics, performance)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_alerts(self, metrics: ResourceMetrics, performance: Dict[str, float]):
        """Check for resource alerts"""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_usage > self.adaptive_thresholds['cpu_critical']:
            alerts.append(('cpu_critical', metrics.cpu_usage))
        elif metrics.cpu_usage > self.adaptive_thresholds['cpu_warning']:
            alerts.append(('cpu_warning', metrics.cpu_usage))
        
        # Memory alerts
        if metrics.memory_usage > self.adaptive_thresholds['memory_critical']:
            alerts.append(('memory_critical', metrics.memory_usage))
        elif metrics.memory_usage > self.adaptive_thresholds['memory_warning']:
            alerts.append(('memory_warning', metrics.memory_usage))
        
        # Temperature alerts
        temperature = performance.get('temperature', 0)
        if temperature > self.adaptive_thresholds['temperature_critical']:
            alerts.append(('temperature_critical', temperature))
        elif temperature > self.adaptive_thresholds['temperature_warning']:
            alerts.append(('temperature_warning', temperature))
        
        # Power alerts
        power = performance.get('power_estimate', 0)
        if power > self.adaptive_thresholds['power_critical']:
            alerts.append(('power_critical', power))
        elif power > self.adaptive_thresholds['power_warning']:
            alerts.append(('power_warning', power))
        
        # Trigger alert callbacks
        for alert_type, value in alerts:
            for callback in self.alert_callbacks.get(alert_type, []):
                try:
                    callback(alert_type, value, metrics, performance)
                except Exception as e:
                    print(f"Alert callback error: {e}")
    
    def _update_adaptive_thresholds(self):
        """Update thresholds based on historical performance"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = [p['performance'] for p in list(self.performance_history)[-20:]]
        
        # Calculate averages and variance
        cpu_usages = [p['cpu_usage'] for p in recent_performance]
        memory_usages = [p['memory_usage'] for p in recent_performance]
        temperatures = [p['temperature'] for p in recent_performance]
        
        cpu_avg = statistics.mean(cpu_usages)
        cpu_std = statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0
        
        # Adaptive thresholds based on usage patterns
        if self.monitoring_mode == ResourceUtilizationMode.ADAPTIVE:
            # Adjust CPU thresholds
            self.adaptive_thresholds['cpu_warning'] = min(0.95, cpu_avg + 2 * cpu_std)
            self.adaptive_thresholds['cpu_critical'] = min(0.98, cpu_avg + 3 * cpu_std)
            
            # Adjust memory thresholds
            memory_avg = statistics.mean(memory_usages)
            memory_std = statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            self.adaptive_thresholds['memory_warning'] = min(0.95, memory_avg + 2 * memory_std)
            self.adaptive_thresholds['memory_critical'] = min(0.98, memory_avg + 3 * memory_std)
    
    def _adjust_monitoring_mode(self, metrics: ResourceMetrics, performance: Dict[str, float]):
        """Adjust monitoring mode based on conditions"""
        current_load = metrics.cpu_usage
        temperature = performance.get('temperature', 0)
        
        if temperature > self.adaptive_thresholds['temperature_warning']:
            # High temperature - be conservative
            self.monitoring_mode = ResourceUtilizationMode.CONSERVATIVE
        elif current_load < 0.3:
            # Low load - can be aggressive
            self.monitoring_mode = ResourceUtilizationMode.AGGRESSIVE
        elif current_load > 0.8:
            # High load - be conservative
            self.monitoring_mode = ResourceUtilizationMode.CONSERVATIVE
        else:
            # Normal conditions - balanced
            self.monitoring_mode = ResourceUtilizationMode.BALANCED
    
    def add_alert_callback(self, alert_type: str, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks[alert_type].append(callback)
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        if self.resource_history:
            return self.resource_history[-1]['metrics']
        else:
            return self.resource_adapter.get_current_metrics()
    
    def predict_resource_availability(self, horizon: float = 30.0) -> ResourceMetrics:
        """Predict future resource availability"""
        if len(self.resource_history) < 5:
            return self.get_resource_metrics()
        
        recent_metrics = [r['metrics'] for r in list(self.resource_history)[-5:]]
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        battery_trend = self._calculate_trend([m.battery_level for m in recent_metrics])
        thermal_trend = self._calculate_trend([p['performance']['temperature'] for p in list(self.performance_history)[-5:]])
        
        current = self.get_resource_metrics()
        steps_ahead = int(horizon)
        
        predicted_cpu = max(0.0, min(1.0, current.cpu_usage + cpu_trend * steps_ahead))
        predicted_memory = max(0.0, min(1.0, current.memory_usage + memory_trend * steps_ahead))
        predicted_battery = max(0.0, min(1.0, current.battery_level + battery_trend * steps_ahead))
        predicted_thermal = max(0.0, 1.0, current.thermal_state + thermal_trend * steps_ahead)
        
        return ResourceMetrics(
            cpu_usage=predicted_cpu,
            memory_usage=predicted_memory,
            battery_level=predicted_battery,
            network_quality=current.network_quality,  # Harder to predict
            thermal_state=predicted_thermal
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0.0
        
        return slope
    
    def can_handle_workload(self, workload: ComputeWorkload) -> Tuple[bool, str, float]:
        """Check if platform can handle workload"""
        current_metrics = self.get_resource_metrics()
        predicted_metrics = self.predict_resource_availability(workload.duration_estimate)
        
        # Check current availability
        available_cpu = 1.0 - current_metrics.cpu_usage
        available_memory = 1.0 - current_metrics.memory_usage
        
        # Check future availability (prediction)
        future_available_cpu = 1.0 - predicted_metrics.cpu_usage
        future_available_memory = 1.0 - predicted_metrics.memory_usage
        
        # Use the more conservative estimate
        cpu_available = min(available_cpu, future_available_cpu)
        memory_available = min(available_memory, future_available_memory)
        
        # Check requirements
        if cpu_available < workload.cpu_requirement:
            return False, f"Insufficient CPU: need {workload.cpu_requirement}, available {cpu_available:.2f}", cpu_available
        
        if memory_available < workload.memory_requirement:
            return False, f"Insufficient memory: need {workload.memory_requirement}, available {memory_available:.2f}", memory_available
        
        # Platform-specific checks
        if self.capabilities.platform in [HardwarePlatform.JETSON_NANO]:
            # Nano is more conservative
            if current_metrics.thermal_state > 0.8:
                return False, "Thermal constraints on Jetson Nano", cpu_available
        
        return True, "Can handle workload", min(cpu_available, memory_available)

class HeterogeneousScheduler:
    """Schedules tasks across heterogeneous platforms"""
    
    def __init__(self, local_capabilities: HardwareCapabilities):
        self.local_capabilities = local_capabilities
        self.platform_performance: Dict[str, PlatformPerformance] = {}
        self.workload_queue = queue.PriorityQueue()
        self.allocation_history: deque = deque(maxlen=100)
        self.scheduler_active = False
        self.scheduler_thread = None
        
        # Performance multipliers for different platforms
        self.platform_multipliers = {
            HardwarePlatform.JETSON_ORIN: {
                TaskComplexity.LIGHT: 1.2,
                TaskComplexity.MEDIUM: 1.5,
                TaskComplexity.HEAVY: 1.8,
                TaskComplexity.EXTREME: 1.5
            },
            HardwarePlatform.JETSON_NANO: {
                TaskComplexity.LIGHT: 0.8,
                TaskComplexity.MEDIUM: 0.6,
                TaskComplexity.HEAVY: 0.3,
                TaskComplexity.EXTREME: 0.1
            },
            HardwarePlatform.X86_LINUX: {
                TaskComplexity.LIGHT: 1.0,
                TaskComplexity.MEDIUM: 1.2,
                TaskComplexity.HEAVY: 1.5,
                TaskComplexity.EXTREME: 2.0
            },
            HardwarePlatform.X86_WINDOWS: {
                TaskComplexity.LIGHT: 1.0,
                TaskComplexity.MEDIUM: 1.1,
                TaskComplexity.HEAVY: 1.3,
                TaskComplexity.EXTREME: 1.8
            }
        }
    
    def start_scheduler(self):
        """Start the scheduler"""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)
    
    def submit_workload(self, workload: ComputeWorkload):
        """Submit workload for scheduling"""
        # Priority queue uses negative priority (higher priority first)
        priority = -workload.priority
        self.workload_queue.put((priority, workload))
    
    def _scheduler_worker(self):
        """Main scheduler worker"""
        while self.scheduler_active:
            try:
                # Get next workload
                try:
                    priority, workload = self.workload_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Allocate workload
                allocation = self._allocate_workload(workload)
                
                if allocation:
                    self.allocation_history.append(allocation)
                    print(f"Allocated workload {workload.workload_id} to {allocation.allocated_platform.value}")
                else:
                    print(f"Could not allocate workload {workload.workload_id}")
                
                self.workload_queue.task_done()
                
            except Exception as e:
                print(f"Scheduler error: {e}")
    
    def _allocate_workload(self, workload: ComputeWorkload) -> Optional[ResourceAllocation]:
        """Allocate workload to best platform"""
        # Get platform scores
        platform_scores = self._evaluate_platforms_for_workload(workload)
        
        if not platform_scores:
            return None
        
        # Select best platform
        best_platform = max(platform_scores.items(), key=lambda x: x[1])
        
        # Create allocation
        allocation = ResourceAllocation(
            workload_id=workload.workload_id,
            allocated_platform=best_platform[0],
            allocated_resources={
                'cpu': workload.cpu_requirement,
                'memory': workload.memory_requirement,
                'gpu': workload.gpu_requirement
            },
            estimated_completion=time.time() + workload.duration_estimate,
            confidence_score=best_platform[1],
            allocation_strategy="performance_based"
        )
        
        return allocation
    
    def _evaluate_platforms_for_workload(self, workload: ComputeWorkload) -> Dict[HardwarePlatform, float]:
        """Evaluate all platforms for workload"""
        scores = {}
        
        # Check local platform first
        local_monitor = AdaptiveResourceMonitor(self.local_capabilities)
        can_handle, reason, availability = local_monitor.can_handle_workload(workload)
        
        if can_handle:
            base_score = self.platform_multipliers.get(self.local_capabilities.platform, {}).get(workload.complexity, 1.0)
            
            # Adjust for availability
            availability_multiplier = availability
            
            # Adjust for thermal state
            metrics = local_monitor.get_resource_metrics()
            thermal_multiplier = 1.0 - metrics.thermal_state * 0.5
            
            score = base_score * availability_multiplier * thermal_multiplier
            scores[self.local_capabilities.platform] = score
        
        # Check remote platforms (would need remote capability info)
        # This is a simplified implementation
        # In practice, you'd query remote nodes for their capabilities
        
        return scores
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'queue_size': self.workload_queue.qsize(),
            'allocations_made': len(self.allocation_history),
            'success_rate': self._calculate_success_rate(),
            'average_allocation_time': self._calculate_avg_allocation_time(),
            'platform_distribution': self._get_platform_distribution()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate allocation success rate"""
        if not self.allocation_history:
            return 0.0
        
        successful = sum(1 for alloc in self.allocation_history if alloc.confidence_score > 0.5)
        return successful / len(self.allocation_history)
    
    def _calculate_avg_allocation_time(self) -> float:
        """Calculate average allocation time"""
        if not self.allocation_history:
            return 0.0
        
        allocation_times = []
        for alloc in self.allocation_history:
            # This would need timing information stored in allocation
            allocation_times.append(alloc.estimated_completion - time.time())
        
        return statistics.mean(allocation_times) if allocation_times else 0.0
    
    def _get_platform_distribution(self) -> Dict[str, int]:
        """Get distribution of allocations by platform"""
        distribution = defaultdict(int)
        for alloc in self.allocation_history:
            distribution[alloc.allocated_platform.value] += 1
        return dict(distribution)