"""Super Neuron implementation for EdgePipe"""

from typing import List, Dict, Tuple
import numpy as np

class SuperNeuron:
    """Super Neuron class representing a group of neurons across adjacent layers"""
    
    def __init__(self, layers: List[int], neurons: Dict[int, List[int]], device_id: int = None):
        """
        Initialize a Super Neuron
        
        Args:
            layers: List of layer indices covered by this super neuron
            neurons: Dictionary mapping layer indices to list of neuron indices in that layer
            device_id: ID of the device this super neuron is assigned to
        """
        self.layers = layers
        self.neurons = neurons
        self.device_id = device_id
        self.id = f"SN_{min(layers)}-{max(layers)}"
        
    def get_layers(self) -> List[int]:
        """Get the layers covered by this super neuron"""
        return self.layers
    
    def get_neurons_in_layer(self, layer_idx: int) -> List[int]:
        """Get neurons in a specific layer"""
        return self.neurons.get(layer_idx, [])
    
    def get_total_neurons(self) -> int:
        """Get total number of neurons in this super neuron"""
        return sum(len(neuron_list) for neuron_list in self.neurons.values())
    
    def set_device(self, device_id: int):
        """Set the device ID for this super neuron"""
        self.device_id = device_id
    
    def get_device(self) -> int:
        """Get the device ID for this super neuron"""
        return self.device_id
    
    def __str__(self) -> str:
        return f"SuperNeuron {self.id} covering layers {self.layers} on device {self.device_id}"

class SuperNeuronNetwork:
    """Network of super neurons"""
    
    def __init__(self, total_layers: int, M_layers: int):
        """
        Initialize a Super Neuron Network
        
        Args:
            total_layers: Total number of layers in the original DNN
            M_layers: Maximum number of consecutive layers per super neuron
        """
        self.total_layers = total_layers
        self.M_layers = M_layers
        self.super_neurons: List[SuperNeuron] = []
        self.layer_to_super_neurons: Dict[int, List[SuperNeuron]] = {}
    
    def add_super_neuron(self, super_neuron: SuperNeuron):
        """Add a super neuron to the network"""
        self.super_neurons.append(super_neuron)
        for layer in super_neuron.get_layers():
            if layer not in self.layer_to_super_neurons:
                self.layer_to_super_neurons[layer] = []
            self.layer_to_super_neurons[layer].append(super_neuron)
    
    def get_super_neurons(self) -> List[SuperNeuron]:
        """Get all super neurons"""
        return self.super_neurons
    
    def get_super_neurons_by_layer(self, layer_idx: int) -> List[SuperNeuron]:
        """Get super neurons covering a specific layer"""
        return self.layer_to_super_neurons.get(layer_idx, [])
    
    def get_super_neurons_by_device(self, device_id: int) -> List[SuperNeuron]:
        """Get super neurons assigned to a specific device"""
        return [sn for sn in self.super_neurons if sn.get_device() == device_id]
    
    def get_total_super_neurons(self) -> int:
        """Get total number of super neurons"""
        return len(self.super_neurons)
