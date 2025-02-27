import numpy as np

class Vehicle:
    """Represents a vehicle in the FLAD cluster with computation and memory capabilities."""
    def __init__(self, id, memory, comp_capability, comm_capability):
        """
        Args:
            id: Vehicle identifier
            memory: Available memory in bytes
            comp_capability: Computational capability (e.g., FLOPS)
            comm_capability: Communication bandwidth capability (e.g., bytes/s)
        """
        self.id = id
        self.memory = memory
        self.comp_capability = comp_capability
        self.comm_capability = comm_capability
        self.assigned_partitions = []
        
    def can_accommodate(self, partition):
        """Check if the vehicle can accommodate the given partition based on memory constraints."""
        current_usage = sum(p.capacity for p in self.assigned_partitions)
        return current_usage + partition.capacity <= self.memory
        
    def assign_partition(self, partition):
        """Assign a partition to this vehicle."""
        if self.can_accommodate(partition):
            self.assigned_partitions.append(partition)
            return True
        return False
        
    @property
    def total_capacity(self):
        """Total model capacity assigned to this vehicle."""
        return sum(p.capacity for p in self.assigned_partitions)
        
    @property
    def total_flops(self):
        """Total computational requirements for this vehicle."""
        return sum(p.flops_per_sample for p in self.assigned_partitions)
    
    def compute_time(self, batch_size=1, utilization=0.5, memory_overhead=1.2):
        """Calculate computation time based on Eq. (2)."""
        if not self.assigned_partitions:
            return 0
        total_flops = self.total_flops * batch_size
        return (total_flops * memory_overhead) / (self.comp_capability * utilization)
    
    def communication_time(self, batch_size=1, memory_overhead=1.2):
        """Calculate communication time based on Eq. (3)."""
        if not self.assigned_partitions:
            return 0
        comm_volume = sum(p.communication_volume for p in self.assigned_partitions)
        # Factor of 2 for forward and backward pass
        total_comm = 2 * comm_volume * batch_size * memory_overhead
        return total_comm / self.comm_capability
    
    def __repr__(self):
        return f"Vehicle({self.id}, memory={self.memory}, comp={self.comp_capability}, comm={self.comm_capability})"


class FLADCluster:
    """Represents a cluster of vehicles and their dependencies."""
    def __init__(self, name):
        self.name = name
        self.vehicles = {}  # id -> Vehicle
        self.dependencies = []  # List of (source_id, target_id) tuples
        
    def add_vehicle(self, vehicle):
        """Add a vehicle to the cluster."""
        self.vehicles[vehicle.id] = vehicle
        
    def add_dependency(self, source_id, target_id):
        """Add a dependency between two vehicles (source must complete before target starts)."""
        if source_id in self.vehicles and target_id in self.vehicles:
            self.dependencies.append((source_id, target_id))
        else:
            raise ValueError(f"Vehicle IDs {source_id} or {target_id} not found in cluster")
            
    def get_adjacency_matrix(self):
        """Get adjacency matrix representing the DAG."""
        n = len(self.vehicles)
        adj_matrix = np.zeros((n, n), dtype=int)
        vehicle_ids = list(self.vehicles.keys())
        id_to_idx = {vid: i for i, vid in enumerate(vehicle_ids)}
        
        for source_id, target_id in self.dependencies:
            source_idx = id_to_idx[source_id]
            target_idx = id_to_idx[target_id]
            adj_matrix[source_idx, target_idx] = 1
            
        return adj_matrix, vehicle_ids
    
    def is_valid_path(self, path):
        """Check if the given path respects all dependencies."""
        path_indices = {v_id: i for i, v_id in enumerate(path)}
        
        for source_id, target_id in self.dependencies:
            if source_id in path_indices and target_id in path_indices:
                if path_indices[source_id] > path_indices[target_id]:
                    return False
        return True
