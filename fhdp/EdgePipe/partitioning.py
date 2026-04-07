"""Hybrid partitioning algorithm for EdgePipe"""

from typing import List, Dict, Tuple
import math
from .super_neuron import SuperNeuron, SuperNeuronNetwork

class HybridPartitioning:
    """Hybrid partitioning algorithm for EdgePipe"""
    
    def __init__(self, total_layers: int, total_devices: int):
        """
        Initialize the hybrid partitioning algorithm
        
        Args:
            total_layers: Total number of layers in the DNN
            total_devices: Total number of edge devices
        """
        self.total_layers = total_layers
        self.total_devices = total_devices
        self.M_layers = self._calculate_M_layers()
        self.layer_groups = []
        self.device_allocation = []
        self.super_neuron_network = SuperNeuronNetwork(total_layers, self.M_layers)
    
    def _calculate_M_layers(self) -> int:
        """
        Calculate M_layers based on total layers and devices
        
        Returns:
            M_layers: Maximum number of consecutive layers per super neuron
        """
        if self.total_devices < 2:
            return self.total_layers  # Horizontal allocation
        
        group_count = math.floor(self.total_devices / 2)
        M_layers = math.floor(self.total_layers / group_count)
        return max(1, M_layers)
    
    def perform_partitioning(self, neurons_per_layer: List[int]) -> SuperNeuronNetwork:
        """
        Perform hybrid partitioning
        
        Args:
            neurons_per_layer: List of neurons per layer
            
        Returns:
            SuperNeuronNetwork: Network of super neurons
        """
        # Step 1: Layer-level horizontal split
        self._split_layers()
        
        # Step 2: Allocate devices based on layer count ratio
        self._allocate_devices()
        
        # Step 3: Neuron-level vertical split and create super neurons
        self._create_super_neurons(neurons_per_layer)
        
        return self.super_neuron_network
    
    def _split_layers(self):
        """Split layers into groups"""
        if self.total_devices < 2:
            # Horizontal allocation
            self.layer_groups = [[i for i in range(self.total_layers)]]
            return
        
        group_count = math.floor(self.total_devices / 2)
        layers_per_group = self.M_layers
        
        self.layer_groups = []
        current_layer = 0
        
        while current_layer < self.total_layers:
            end_layer = min(current_layer + layers_per_group, self.total_layers)
            self.layer_groups.append(list(range(current_layer, end_layer)))
            current_layer = end_layer
        
        # Handle remaining layers
        if current_layer < self.total_layers:
            for layer in range(current_layer, self.total_layers):
                self.layer_groups.append([layer])
    
    def _allocate_devices(self):
        """Allocate devices to layer groups"""
        if self.total_devices < 2:
            self.device_allocation = [self.total_devices]
            return
        
        total_group_layers = sum(len(group) for group in self.layer_groups)
        self.device_allocation = []
        
        for group in self.layer_groups:
            # Allocate devices proportionally to the number of layers in the group
            devices = round(len(group) / total_group_layers * self.total_devices)
            # Ensure at least 1 device per group
            devices = max(1, devices)
            self.device_allocation.append(devices)
        
        # Adjust to ensure total devices matches
        total_allocated = sum(self.device_allocation)
        if total_allocated > self.total_devices:
            # Reduce devices from groups with more devices
            while total_allocated > self.total_devices:
                for i in range(len(self.device_allocation)):
                    if self.device_allocation[i] > 1:
                        self.device_allocation[i] -= 1
                        total_allocated -= 1
                        if total_allocated == self.total_devices:
                            break
        elif total_allocated < self.total_devices:
            # Add devices to groups with fewer devices
            while total_allocated < self.total_devices:
                for i in range(len(self.device_allocation)):
                    self.device_allocation[i] += 1
                    total_allocated += 1
                    if total_allocated == self.total_devices:
                        break
    
    def _create_super_neurons(self, neurons_per_layer: List[int]):
        """Create super neurons based on partitioning"""
        device_id = 0
        
        for group_idx, (layer_group, device_count) in enumerate(zip(self.layer_groups, self.device_allocation)):
            # For each layer in the group, split neurons vertically
            layer_neuron_splits = {}
            for layer in layer_group:
                neurons = neurons_per_layer[layer]
                # Split neurons among devices
                split = self._split_neurons(neurons, device_count)
                layer_neuron_splits[layer] = split
            
            # Create super neurons for each device in the group
            for device_idx in range(device_count):
                # Collect neurons for this device across all layers in the group
                neurons = {}
                for layer in layer_group:
                    neurons[layer] = layer_neuron_splits[layer][device_idx]
                
                # Create super neuron
                super_neuron = SuperNeuron(layer_group, neurons, device_id)
                self.super_neuron_network.add_super_neuron(super_neuron)
                device_id += 1
    
    def _split_neurons(self, total_neurons: int, device_count: int) -> List[List[int]]:
        """
        Split neurons among devices
        
        Args:
            total_neurons: Total number of neurons in the layer
            device_count: Number of devices to split across
            
        Returns:
            List of neuron indices for each device
        """
        # Base number of neurons per device
        base = total_neurons // device_count
        # Remaining neurons to distribute
        remainder = total_neurons % device_count
        
        splits = []
        start = 0
        
        for i in range(device_count):
            # Distribute remaining neurons from the end
            count = base + (1 if i >= device_count - remainder else 0)
            splits.append(list(range(start, start + count)))
            start += count
        
        return splits
    
    def get_partitioning_info(self) -> Dict:
        """
        Get partitioning information
        
        Returns:
            Dict with partitioning details
        """
        return {
            'total_layers': self.total_layers,
            'total_devices': self.total_devices,
            'M_layers': self.M_layers,
            'layer_groups': self.layer_groups,
            'device_allocation': self.device_allocation,
            'total_super_neurons': self.super_neuron_network.get_total_super_neurons()
        }
