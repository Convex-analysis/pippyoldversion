import numpy as np
import networkx as nx
from itertools import permutations

class PipelineOptimizer:
    """Optimizer for pipeline generation in FLAD."""
    def __init__(self, cluster, model, batch_size=1, utilization=0.5, memory_overhead=1.2):
        """
        Args:
            cluster: FLADCluster object
            model: DNNModel object
            batch_size: Number of samples per batch
            utilization: GPU utilization factor (between 0.3 and 0.7)
            memory_overhead: Memory bandwidth overhead factor (between 1.1 and 1.5)
        """
        self.cluster = cluster
        self.model = model
        self.batch_size = batch_size
        self.utilization = utilization
        self.memory_overhead = memory_overhead
    
    def calculate_path_time(self, path, partition_strategy):
        """Calculate execution time for a path with a specific partition strategy.
        
        Args:
            path: List of vehicle IDs representing execution order
            partition_strategy: Dict mapping vehicle ID to list of assigned partitions
            
        Returns:
            Total execution time as defined in Eq. (5)
        """
        total_time = 0
        n = len(path)
        
        # Update vehicle assignments based on partition strategy
        for v_id, partitions in partition_strategy.items():
            vehicle = self.cluster.vehicles[v_id]
            vehicle.assigned_partitions = partitions
        
        # Calculate computation and communication time
        for i, v_id in enumerate(path):
            vehicle = self.cluster.vehicles[v_id]
            # Add computation time
            total_time += vehicle.compute_time(
                self.batch_size, self.utilization, self.memory_overhead)
            
            # Add communication time for all but the last vehicle
            if i < n - 1:
                total_time += vehicle.communication_time(
                    self.batch_size, self.memory_overhead)
                
        return total_time
    
    def check_constraints(self, partition_strategy):
        """Check if the partition strategy satisfies all constraints.
        
        Args:
            partition_strategy: Dict mapping vehicle ID to list of assigned partitions
            
        Returns:
            Boolean indicating whether all constraints are satisfied
        """
        # Check constraint (1): all partitions are assigned
        assigned_partitions = [p for partitions in partition_strategy.values() 
                              for p in partitions]
        if len(assigned_partitions) != len(self.model.unit_partitions):
            return False
            
        # Check constraint (2): memory constraints
        for v_id, partitions in partition_strategy.items():
            vehicle = self.cluster.vehicles[v_id]
            if sum(p.capacity for p in partitions) > vehicle.memory:
                return False
                
        # Check constraint (5): no partition is assigned to multiple vehicles
        partition_ids = [id(p) for p in assigned_partitions]
        if len(partition_ids) != len(set(partition_ids)):
            return False
            
        return True
    
    def find_valid_paths(self):
        """Find all valid execution paths that respect dependencies."""
        # Create a directed graph from dependencies
        G = nx.DiGraph()
        for v_id in self.cluster.vehicles:
            G.add_node(v_id)
        
        for source, target in self.cluster.dependencies:
            G.add_edge(source, target)
        
        # Check if graph has cycles
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Dependencies form a cycle, which is not allowed in a DAG.")
        
        # Get all topological sorts (valid execution orders)
        valid_paths = list(nx.all_topological_sorts(G))
        return valid_paths
    
    def optimize(self):
        """Find optimal path and partition strategy.
        
        Returns:
            Tuple of (optimal_path, optimal_partition_strategy, min_execution_time)
        """
        valid_paths = self.find_valid_paths()
        
        if not valid_paths:
            raise ValueError("No valid execution paths found.")
            
        # Simplified approach: try all valid paths with a greedy partitioning strategy
        min_time = float('inf')
        optimal_path = None
        optimal_strategy = None
        
        for path in valid_paths:
            # For simplicity, using a greedy partitioning approach here
            # In a real implementation, this would be a more sophisticated algorithm
            partition_strategy = self.greedy_partition(path)
            
            if partition_strategy and self.check_constraints(partition_strategy):
                path_time = self.calculate_path_time(path, partition_strategy)
                if path_time < min_time:
                    min_time = path_time
                    optimal_path = path
                    optimal_strategy = partition_strategy
        
        return optimal_path, optimal_strategy, min_time
    
    def greedy_partition(self, path):
        """Simple greedy partitioning strategy.
        
        In a real implementation, this would be replaced with a more sophisticated algorithm.
        """
        # Reset vehicle assignments
        for vehicle in self.cluster.vehicles.values():
            vehicle.assigned_partitions = []
            
        partition_strategy = {v_id: [] for v_id in self.cluster.vehicles}
        available_partitions = self.model.unit_partitions.copy()
        
        # Sort vehicles by computational capability (highest first)
        sorted_vehicles = sorted(
            [self.cluster.vehicles[v_id] for v_id in path],
            key=lambda v: v.comp_capability,
            reverse=True
        )
        
        # Sort partitions by computational requirements (highest first)
        available_partitions.sort(key=lambda p: p.flops_per_sample, reverse=True)
        
        # Assign partitions to vehicles
        for partition in available_partitions:
            assigned = False
            for vehicle in sorted_vehicles:
                if vehicle.can_accommodate(partition):
                    partition_strategy[vehicle.id].append(partition)
                    vehicle.assign_partition(partition)
                    assigned = True
                    break
            
            if not assigned:
                # Couldn't assign this partition to any vehicle
                return None
        
        return partition_strategy
