import numpy as np

class ModelComponent:
    """Represents a specific component of a DNN model like RGB, Lidar, etc."""
    def __init__(self, name, flops_per_sample, capacity):
        """
        Args:
            name: Component name (e.g., RGB, Lidar, Encoder, Decoder)
            flops_per_sample: Computational requirement in FLOPS
            capacity: Memory requirement in bytes
        """
        self.name = name
        self.flops_per_sample = flops_per_sample
        self.capacity = capacity
    
    def __repr__(self):
        return f"{self.name}(flops={self.flops_per_sample}, capacity={self.capacity})"


class UnitModelPartition:
    """Represents a unit partition of the model that can be assigned to a vehicle."""
    def __init__(self, name, components, flops_per_sample, capacity, communication_volume):
        """
        Args:
            name: Partition name
            components: List of ModelComponents in this partition
            flops_per_sample: Total computational requirement in FLOPS
            capacity: Total memory requirement in bytes
            communication_volume: Amount of data that needs to be transferred
        """
        self.name = name
        self.components = components
        self.flops_per_sample = flops_per_sample
        self.capacity = capacity
        self.communication_volume = communication_volume
    
    def __repr__(self):
        return f"UnitPartition({self.name}, flops={self.flops_per_sample}, capacity={self.capacity})"


class DNNModel:
    """Represents the entire DNN model that will be partitioned across vehicles."""
    def __init__(self, name, components=None, unit_partitions=None):
        """
        Args:
            name: Model name
            components: List of ModelComponents in this model
            unit_partitions: Predefined unit partitions (if any)
        """
        self.name = name
        self.components = components or []
        self.unit_partitions = unit_partitions or []
        
    def add_component(self, component):
        self.components.append(component)
        
    def add_unit_partition(self, unit_partition):
        self.unit_partitions.append(unit_partition)
        
    @property
    def total_capacity(self):
        return sum(comp.capacity for comp in self.components)
    
    @property
    def total_flops_per_sample(self):
        return sum(comp.flops_per_sample for comp in self.components)
    
    def __repr__(self):
        return f"DNNModel({self.name}, components={len(self.components)}, partitions={len(self.unit_partitions)})"
