"""
Load Balancer for FHDP Heterogeneous Platforms

Intelligent load balancing and task scheduling optimization across
Jetson Orin Nano, x86 PCs, and other heterogeneous devices.
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
from collections import deque, defaultdict
import heapq
import logging

from .hardware_adapter import HardwarePlatform, HardwareCapabilities
from .heterogeneous_resource import (
    ComputeWorkload, ResourceAllocation, TaskComplexity,
    HeterogeneousScheduler, AdaptiveResourceMonitor
)
from .types import ResourceMetrics, ResourceClass

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_AWARE = "resource_aware"
    LATENCY_AWARE = "latency_aware"
    COST_AWARE = "cost_aware"
    HYBRID = "hybrid"

class NodeState(Enum):
    """Node state in load balancer"""
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class NodeInfo:
    """Node information for load balancing"""
    node_id: str
    platform: HardwarePlatform
    capabilities: HardwareCapabilities
    current_load: float = 0.0  # 0.0-1.0
    current_tasks: int = 0
    max_tasks: int = 1
    state: NodeState = NodeState.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    performance_score: float = 1.0
    reliability_score: float = 1.0
    network_latency: float = 1.0  # ms
    task_history: deque = field(default_factory=lambda: deque(maxlen=100))
    allocation_history: deque = field(default_factory=lambda: deque(maxlen=50))

@dataclass
class TaskInfo:
    """Task information for scheduling"""
    task_id: str
    workload: ComputeWorkload
    priority: int
    submitted_time: float = field(default_factory=time.time)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    placement_constraints: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        # For priority queue (higher priority first)
        return self.priority > other.priority

class PerformancePredictor:
    """Predicts task performance on different platforms"""
    
    def __init__(self):
        self.performance_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.platform_benchmarks: Dict[HardwarePlatform, Dict[TaskComplexity, float]] = {}
        self.benchmark_cache: Dict[str, float] = {}
        
        # Initialize platform benchmarks
        self._initialize_benchmarks()
    
    def _initialize_benchmarks(self):
        """Initialize performance benchmarks for different platforms"""
        self.platform_benchmarks = {
            HardwarePlatform.JETSON_ORIN: {
                TaskComplexity.LIGHT: 0.8,    # 0.8x reference performance
                TaskComplexity.MEDIUM: 1.5,   # 1.5x for edge AI workloads
                TaskComplexity.HEAVY: 1.2,     # Good for heavy ML tasks
                TaskComplexity.EXTREME: 0.8    # Limited for extreme tasks
            },
            HardwarePlatform.JETSON_NANO: {
                TaskComplexity.LIGHT: 0.6,     # Good for light tasks
                TaskComplexity.MEDIUM: 0.4,    # Limited for medium
                TaskComplexity.HEAVY: 0.2,     # Poor for heavy
                TaskComplexity.EXTREME: 0.1    # Very poor for extreme
            },
            HardwarePlatform.JETSON_XAVIER: {
                TaskComplexity.LIGHT: 1.0,
                TaskComplexity.MEDIUM: 1.8,   # Excellent for ML workloads
                TaskComplexity.HEAVY: 1.5,
                TaskComplexity.EXTREME: 1.0
            },
            HardwarePlatform.X86_LINUX: {
                TaskComplexity.LIGHT: 1.2,
                TaskComplexity.MEDIUM: 1.5,
                TaskComplexity.HEAVY: 2.0,     # Best for heavy computation
                TaskComplexity.EXTREME: 2.5    # Excellent for extreme tasks
            },
            HardwarePlatform.X86_WINDOWS: {
                TaskComplexity.LIGHT: 1.1,
                TaskComplexity.MEDIUM: 1.4,
                TaskComplexity.HEAVY: 1.8,
                TaskComplexity.EXTREME: 2.2
            }
        }
    
    def predict_performance(self, workload: ComputeWorkload, 
                          platform: HardwarePlatform,
                          node_info: NodeInfo) -> float:
        """Predict task performance on platform"""
        # Get base benchmark
        base_performance = self.platform_benchmarks.get(platform, {}).get(
            workload.complexity, 1.0
        )
        
        # Adjust for current load
        load_factor = max(0.2, 1.0 - node_info.current_load)
        
        # Adjust for reliability
        reliability_factor = node_info.reliability_score
        
        # Adjust for network latency (for distributed tasks)
        latency_factor = max(0.8, 1.0 - (node_info.network_latency / 1000.0))
        
        # Historical performance adjustment
        history_key = f"{platform.value}_{workload.complexity.value}"
        if node_info.task_id in self.performance_history:
            historical_times = self.performance_history[node_info.task_id][history_key]
            if historical_times:
                historical_factor = np.mean(historical_times)
                base_performance *= historical_factor
        
        # Calculate final performance score
        performance_score = (base_performance * load_factor * 
                           reliability_factor * latency_factor)
        
        return performance_score
    
    def record_task_performance(self, task_id: str, platform: HardwarePlatform,
                               complexity: TaskComplexity, performance: float):
        """Record actual task performance"""
        history_key = f"{platform.value}_{complexity.value}"
        self.performance_history[task_id][history_key].append(performance)

class IntelligentLoadBalancer:
    """Intelligent load balancer for heterogeneous platforms"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HYBRID):
        self.strategy = strategy
        self.nodes: Dict[str, NodeInfo] = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        self.performance_predictor = PerformancePredictor()
        self.load_balancer_active = False
        self.load_balancer_thread = None
        
        # Load balancing parameters
        self.load_threshold = 0.8
        self.rebalance_interval = 30.0  # seconds
        self.heartbeat_timeout = 60.0  # seconds
        
        # Statistics
        self.stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'rebalance_count': 0,
            'average_wait_time': 0.0,
            'average_execution_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_load_balancing(self):
        """Start load balancing service"""
        if self.load_balancer_active:
            return
        
        self.load_balancer_active = True
        self.load_balancer_thread = threading.Thread(
            target=self._load_balancing_worker,
            daemon=True
        )
        self.load_balancer_thread.start()
        self.logger.info("Load balancer started")
    
    def stop_load_balancing(self):
        """Stop load balancing service"""
        self.load_balancer_active = False
        if self.load_balancer_thread:
            self.load_balancer_thread.join(timeout=2.0)
        self.logger.info("Load balancer stopped")
    
    def register_node(self, node_id: str, capabilities: HardwareCapabilities,
                     max_tasks: int = 1, network_latency: float = 1.0):
        """Register a node with the load balancer"""
        node = NodeInfo(
            node_id=node_id,
            platform=capabilities.platform,
            capabilities=capabilities,
            max_tasks=max_tasks,
            network_latency=network_latency,
            performance_score=self._calculate_initial_performance_score(capabilities),
            reliability_score=1.0
        )
        
        self.nodes[node_id] = node
        self.logger.info(f"Registered node {node_id} ({capabilities.platform.value})")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the load balancer"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.state = NodeState.FAILED
            self.logger.info(f"Unregistered node {node_id}")
    
    def submit_task(self, task_info: TaskInfo) -> str:
        """Submit a task for scheduling"""
        self.task_queue.put(task_info)
        self.active_tasks[task_info.task_id] = task_info
        self.logger.debug(f"Submitted task {task_info.task_id}")
        return task_info.task_id
    
    def update_node_load(self, node_id: str, current_load: float, current_tasks: int):
        """Update node load information"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.current_load = current_load
            node.current_tasks = current_tasks
            node.last_heartbeat = time.time()
            
            # Update node state based on load
            if current_load > 0.95:
                node.state = NodeState.OVERLOADED
            elif current_load > 0.8:
                node.state = NodeState.BUSY
            else:
                node.state = NodeState.ACTIVE
    
    def update_heartbeat(self, node_id: str):
        """Update node heartbeat"""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
    
    def _load_balancing_worker(self):
        """Main load balancing worker"""
        last_rebalance = time.time()
        
        while self.load_balancer_active:
            try:
                # Check for tasks to schedule
                try:
                    task_info = self.task_queue.get(timeout=1.0)
                    success = self._schedule_task(task_info)
                    
                    if success:
                        self.stats['tasks_scheduled'] += 1
                    else:
                        # Re-queue if scheduling failed
                        self.task_queue.put(task_info)
                    
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    pass
                
                # Periodic rebalancing
                current_time = time.time()
                if current_time - last_rebalance > self.rebalance_interval:
                    self._rebalance_tasks()
                    last_rebalance = current_time
                
                # Check for failed nodes
                self._check_node_health()
                
                time.sleep(0.1)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Load balancer error: {e}")
                time.sleep(1.0)
    
    def _schedule_task(self, task_info: TaskInfo) -> bool:
        """Schedule a task to an appropriate node"""
        # Get eligible nodes
        eligible_nodes = self._get_eligible_nodes(task_info)
        
        if not eligible_nodes:
            self.logger.warning(f"No eligible nodes for task {task_info.task_id}")
            return False
        
        # Select best node based on strategy
        selected_node = self._select_node(task_info, eligible_nodes)
        
        if not selected_node:
            return False
        
        # Allocate task to node
        allocation = self._allocate_task(task_info, selected_node)
        
        if allocation:
            selected_node.task_history.append(task_info.task_id)
            selected_node.allocation_history.append(allocation)
            
            # Update node load (estimated)
            load_increase = self._estimate_task_load(task_info.workload)
            selected_node.current_load += load_increase
            selected_node.current_tasks += 1
            
            self.logger.info(f"Scheduled task {task_info.task_id} to node {selected_node.node_id}")
            return True
        
        return False
    
    def _get_eligible_nodes(self, task_info: TaskInfo) -> List[NodeInfo]:
        """Get nodes eligible for task execution"""
        eligible_nodes = []
        
        for node in self.nodes.values():
            # Skip failed nodes
            if node.state == NodeState.FAILED:
                continue
            
            # Skip overloaded nodes (for new tasks)
            if node.state == NodeState.OVERLOADED and node.current_tasks > 0:
                continue
            
            # Check platform constraints
            if task_info.workload.platform_preferences:
                if node.platform not in task_info.workload.platform_preferences:
                    continue
            
            # Check placement constraints
            if task_info.placement_constraints:
                if node.node_id not in task_info.placement_constraints:
                    continue
            
            # Check resource availability
            if node.current_tasks >= node.max_tasks:
                continue
            
            # Check load threshold
            if node.current_load + self._estimate_task_load(task_info.workload) > self.load_threshold:
                continue
            
            eligible_nodes.append(node)
        
        return eligible_nodes
    
    def _select_node(self, task_info: TaskInfo, eligible_nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """Select best node based on strategy"""
        if not eligible_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(eligible_nodes)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(task_info, eligible_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(task_info, eligible_nodes)
        elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
            return self._latency_aware_selection(task_info, eligible_nodes)
        elif self.strategy == LoadBalancingStrategy.COST_AWARE:
            return self._cost_aware_selection(task_info, eligible_nodes)
        elif self.strategy == LoadBalancingStrategy.HYBRID:
            return self._hybrid_selection(task_info, eligible_nodes)
        else:
            return eligible_nodes[0]  # Fallback
    
    def _round_robin_selection(self, eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection"""
        # Simple round-robin based on task count
        return min(eligible_nodes, key=lambda n: n.current_tasks)
    
    def _performance_based_selection(self, task_info: TaskInfo, 
                                   eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Performance-based node selection"""
        def performance_score(node: NodeInfo) -> float:
            base_score = self.performance_predictor.predict_performance(
                task_info.workload, node.platform, node
            )
            
            # Adjust for current load
            load_adjustment = max(0.1, 1.0 - node.current_load)
            
            return base_score * load_adjustment * node.reliability_score
        
        return max(eligible_nodes, key=performance_score)
    
    def _resource_aware_selection(self, task_info: TaskInfo, 
                                 eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Resource-aware node selection"""
        def resource_score(node: NodeInfo) -> float:
            # Calculate resource match score
            cpu_available = 1.0 - node.current_load
            memory_available = 1.0 - node.current_load  # Simplified
            
            cpu_match = min(1.0, cpu_available / max(0.1, task_info.workload.cpu_requirement))
            memory_match = min(1.0, memory_available / max(0.1, task_info.workload.memory_requirement))
            
            # Platform-specific resource scoring
            if node.platform == HardwarePlatform.JETSON_ORIN:
                if task_info.workload.gpu_requirement > 0:
                    gpu_boost = 1.5  # Orin is good for GPU tasks
                else:
                    gpu_boost = 1.0
            elif node.platform == HardwarePlatform.JETSON_NANO:
                gpu_boost = 0.7  # Nano GPU is limited
            elif node.platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
                gpu_boost = 1.2  # x86 good for general tasks
            else:
                gpu_boost = 1.0
            
            return (cpu_match * 0.4 + memory_match * 0.3 + 
                   node.reliability_score * 0.2 + gpu_boost * 0.1)
        
        return max(eligible_nodes, key=resource_score)
    
    def _latency_aware_selection(self, task_info: TaskInfo, 
                                eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Latency-aware node selection"""
        def latency_score(node: NodeInfo) -> float:
            # Lower latency is better
            latency_factor = max(0.1, 1.0 - (node.network_latency / 1000.0))
            
            # Consider task type
            if task_info.workload.complexity in [TaskComplexity.LIGHT, TaskComplexity.MEDIUM]:
                # Latency more important for light tasks
                latency_weight = 0.7
            else:
                # Less important for heavy tasks
                latency_weight = 0.3
            
            # Combined score
            other_factors = (node.reliability_score * 0.5 + 
                           (1.0 - node.current_load) * 0.5)
            
            return latency_factor * latency_weight + other_factors * (1 - latency_weight)
        
        return max(eligible_nodes, key=latency_score)
    
    def _cost_aware_selection(self, task_info: TaskInfo, 
                            eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Cost-aware node selection"""
        def cost_score(node: NodeInfo) -> float:
            # Estimate cost based on platform and capabilities
            cost_factors = {
                HardwarePlatform.JETSON_NANO: 0.2,     # Low cost
                HardwarePlatform.JETSON_ORIN: 0.4,      # Medium cost
                HardwarePlatform.JETSON_XAVIER: 0.6,    # Higher cost
                HardwarePlatform.X86_LINUX: 0.5,        # Medium cost
                HardwarePlatform.X86_WINDOWS: 0.6,      # Higher cost
            }
            
            cost = cost_factors.get(node.platform, 0.5)
            
            # Performance per cost
            performance = self.performance_predictor.predict_performance(
                task_info.workload, node.platform, node
            )
            
            return performance / cost
        
        return max(eligible_nodes, key=cost_score)
    
    def _hybrid_selection(self, task_info: TaskInfo, 
                         eligible_nodes: List[NodeInfo]) -> NodeInfo:
        """Hybrid selection combining multiple strategies"""
        def hybrid_score(node: NodeInfo) -> float:
            # Performance score
            perf_score = self.performance_predictor.predict_performance(
                task_info.workload, node.platform, node
            )
            
            # Resource availability score
            resource_score = max(0.1, 1.0 - node.current_load)
            
            # Reliability score
            reliability_score = node.reliability_score
            
            # Latency score
            latency_score = max(0.1, 1.0 - (node.network_latency / 1000.0))
            
            # Load balancing score (prefer less loaded nodes)
            balance_score = max(0.1, 1.0 - (node.current_tasks / node.max_tasks))
            
            # Weighted combination
            weights = {
                'performance': 0.3,
                'resource': 0.25,
                'reliability': 0.2,
                'latency': 0.15,
                'balance': 0.1
            }
            
            total_score = (
                perf_score * weights['performance'] +
                resource_score * weights['resource'] +
                reliability_score * weights['reliability'] +
                latency_score * weights['latency'] +
                balance_score * weights['balance']
            )
            
            # Platform-specific adjustments
            if task_info.workload.complexity in [TaskComplexity.HEAVY, TaskComplexity.EXTREME]:
                if node.platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_WINDOWS]:
                    total_score *= 1.2  # Prefer x86 for heavy tasks
            elif task_info.workload.complexity in [TaskComplexity.LIGHT, TaskComplexity.MEDIUM]:
                if node.platform in [HardwarePlatform.JETSON_ORIN, HardwarePlatform.JETSON_NANO]:
                    total_score *= 1.1  # Prefer edge for light tasks
            
            return total_score
        
        return max(eligible_nodes, key=hybrid_score)
    
    def _allocate_task(self, task_info: TaskInfo, node: NodeInfo) -> Optional[ResourceAllocation]:
        """Allocate task to specific node"""
        try:
            allocation = ResourceAllocation(
                workload_id=task_info.task_id,
                allocated_platform=node.platform,
                allocated_resources={
                    'cpu': task_info.workload.cpu_requirement,
                    'memory': task_info.workload.memory_requirement,
                    'gpu': task_info.workload.gpu_requirement
                },
                estimated_completion=time.time() + task_info.workload.duration_estimate,
                confidence_score=self.performance_predictor.predict_performance(
                    task_info.workload, node.platform, node
                ),
                allocation_strategy=self.strategy.value
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Task allocation failed: {e}")
            return None
    
    def _estimate_task_load(self, workload: ComputeWorkload) -> float:
        """Estimate load impact of task"""
        # Weighted combination of resource requirements
        return (workload.cpu_requirement * 0.4 + 
                workload.memory_requirement * 0.3 + 
                workload.gpu_requirement * 0.3)
    
    def _calculate_initial_performance_score(self, capabilities: HardwareCapabilities) -> float:
        """Calculate initial performance score for node"""
        # Base score based on compute capability
        capability_scores = {
            ComputeCapability.EDGE_AI: 0.8,
            ComputeCapability.SERVER_CLASS: 1.0,
            ComputeCapability.EMBEDDED: 0.5,
            ComputeCapability.BASIC: 0.3
        }
        
        base_score = capability_scores.get(capabilities.compute_capability, 0.5)
        
        # Adjust for hardware specs
        cpu_factor = min(1.0, capabilities.cpu_cores / 8.0)  # Normalize to 8 cores
        memory_factor = min(1.0, capabilities.memory_total / 16.0)  # Normalize to 16GB
        
        return base_score * (cpu_factor * 0.5 + memory_factor * 0.5)
    
    def _rebalance_tasks(self):
        """Rebalance tasks across nodes"""
        self.logger.debug("Starting task rebalancing")
        
        overloaded_nodes = [n for n in self.nodes.values() if n.state == NodeState.OVERLOADED]
        underloaded_nodes = [n for n in self.nodes.values() 
                          if n.state == NodeState.ACTIVE and n.current_load < 0.5]
        
        if not overloaded_nodes or not underloaded_nodes:
            return
        
        # Simple rebalancing: move tasks from most overloaded to least loaded
        for overloaded in overloaded_nodes:
            for underloaded in underloaded_nodes:
                if overloaded.current_load > self.load_threshold and underloaded.current_load < 0.5:
                    # Could implement task migration here
                    # For now, just log the rebalancing decision
                    self.logger.debug(f"Would rebalance from {overloaded.node_id} to {underloaded.node_id}")
        
        self.stats['rebalance_count'] += 1
    
    def _check_node_health(self):
        """Check for failed nodes"""
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_heartbeat > self.heartbeat_timeout:
                if node.state != NodeState.FAILED:
                    node.state = NodeState.FAILED
                    self.logger.warning(f"Node {node_id} marked as failed (heartbeat timeout)")
    
    def complete_task(self, task_id: str, success: bool = True, execution_time: float = 0.0):
        """Mark task as completed"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            
            # Update node load (estimated reduction)
            for node in self.nodes.values():
                if task_id in node.task_history:
                    load_reduction = self._estimate_task_load(task_info.workload)
                    node.current_load = max(0, node.current_load - load_reduction)
                    node.current_tasks = max(0, node.current_tasks - 1)
                    
                    # Update reliability score
                    if success:
                        node.reliability_score = min(1.0, node.reliability_score * 1.01)
                    else:
                        node.reliability_score = max(0.1, node.reliability_score * 0.95)
                    
                    break
            
            # Move to completed tasks
            self.completed_tasks.append({
                'task_id': task_id,
                'completed_time': time.time(),
                'success': success,
                'execution_time': execution_time,
                'wait_time': time.time() - task_info.submitted_time
            })
            
            del self.active_tasks[task_id]
            
            # Update statistics
            if success:
                self.stats['tasks_completed'] += 1
            else:
                self.stats['tasks_failed'] += 1
            
            self.logger.debug(f"Completed task {task_id} (success: {success})")
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        # Calculate average wait and execution times
        if self.completed_tasks:
            wait_times = [t['wait_time'] for t in self.completed_tasks]
            exec_times = [t['execution_time'] for t in self.completed_tasks if t['execution_time'] > 0]
            
            self.stats['average_wait_time'] = np.mean(wait_times) if wait_times else 0.0
            self.stats['average_execution_time'] = np.mean(exec_times) if exec_times else 0.0
        
        # Node statistics
        node_stats = {}
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                'platform': node.platform.value,
                'state': node.state.value,
                'current_load': node.current_load,
                'current_tasks': node.current_tasks,
                'max_tasks': node.max_tasks,
                'performance_score': node.performance_score,
                'reliability_score': node.reliability_score,
                'network_latency': node.network_latency,
                'total_tasks_processed': len(node.task_history)
            }
        
        return {
            'strategy': self.strategy.value,
            'statistics': self.stats,
            'node_statistics': node_stats,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'total_nodes': len(self.nodes),
            'active_nodes': len([n for n in self.nodes.values() if n.state == NodeState.ACTIVE])
        }
    
    def get_node_recommendations(self, task_info: TaskInfo) -> List[Tuple[str, float]]:
        """Get node recommendations for a task"""
        eligible_nodes = self._get_eligible_nodes(task_info)
        
        recommendations = []
        for node in eligible_nodes:
            score = self.performance_predictor.predict_performance(
                task_info.workload, node.platform, node
            )
            recommendations.append((node.node_id, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations