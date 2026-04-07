"""Pipeline scheduler for EdgePipe"""

from typing import List, Dict, Tuple
import time
from .super_neuron import SuperNeuronNetwork

class PipelineScheduler:
    """Pipeline scheduler for EdgePipe"""
    
    def __init__(self, super_neuron_network: SuperNeuronNetwork):
        """
        Initialize the pipeline scheduler
        
        Args:
            super_neuron_network: Network of super neurons
        """
        self.super_neuron_network = super_neuron_network
        self.total_layers = super_neuron_network.total_layers
        self.M_layers = super_neuron_network.M_layers
        self.P = self.total_layers // self.M_layers  # Pipeline stages
        self.super_neurons = super_neuron_network.get_super_neurons()
        self.sn_by_device = self._group_super_neurons_by_device()
    
    def _group_super_neurons_by_device(self) -> Dict[int, List]:
        """
        Group super neurons by device
        
        Returns:
            Dict mapping device IDs to list of super neurons
        """
        sn_by_device = {}
        for sn in self.super_neurons:
            device_id = sn.get_device()
            if device_id not in sn_by_device:
                sn_by_device[device_id] = []
            sn_by_device[device_id].append(sn)
        return sn_by_device
    
    def generate_schedule(self, batch_size: int) -> Dict[int, List]:
        """
        Generate pipeline schedule
        
        Args:
            batch_size: Number of batches
            
        Returns:
            Schedule for each device
        """
        schedule = {}
        
        # Initialize schedule for each device
        for device_id in self.sn_by_device:
            schedule[device_id] = []
        
        # Generate schedule for each batch
        for batch_idx in range(batch_size):
            # Generate forward pass schedule
            for device_id, device_sns in self.sn_by_device.items():
                for sn in device_sns:
                    layers = sn.get_layers()
                    # Add forward pass for each layer in the super neuron
                    for layer in layers:
                        schedule[device_id].append({
                            'type': 'forward',
                            'batch': batch_idx,
                            'layer': layer,
                            'super_neuron': sn.id
                        })
            
            # Generate backward pass schedule
            for device_id, device_sns in self.sn_by_device.items():
                for sn in device_sns:
                    layers = sn.get_layers()
                    # Add backward pass for each layer in reverse order
                    for layer in reversed(layers):
                        # Skip input layer in backward pass
                        if layer == 0:
                            continue
                        schedule[device_id].append({
                            'type': 'backward',
                            'batch': batch_idx,
                            'layer': layer,
                            'super_neuron': sn.id
                        })
        
        return schedule
    
    def calculate_execution_time(self, forward_time_per_layer: float, backward_time_per_layer: float, batch_size: int) -> float:
        """
        Calculate total execution time
        
        Args:
            forward_time_per_layer: Time per layer for forward pass
            backward_time_per_layer: Time per layer for backward pass
            batch_size: Number of batches
            
        Returns:
            Total execution time
        """
        # Calculate time based on EdgePipe's theoretical model
        # Formula: 2L + 2M(N-1) - 1
        L = self.total_layers
        M = self.M_layers
        N = batch_size
        
        # Time per time slot
        slot_time = max(forward_time_per_layer, backward_time_per_layer)
        
        # Calculate total slots
        if M == L:  # Horizontal allocation
            total_slots = (2 * L - 1) * N
        else:  # Non-horizontal allocation
            total_slots = 2 * L + 2 * M * (N - 1) - 1
        
        return total_slots * slot_time
    
    def simulate_execution(self, schedule: Dict[int, List], forward_time_per_layer: float, backward_time_per_layer: float) -> Dict:
        """
        Simulate pipeline execution
        
        Args:
            schedule: Pipeline schedule
            forward_time_per_layer: Time per layer for forward pass
            backward_time_per_layer: Time per layer for backward pass
            
        Returns:
            Execution results
        """
        device_times = {device_id: 0.0 for device_id in schedule}
        total_time = 0.0
        
        # Simulate execution
        for device_id, tasks in schedule.items():
            current_time = 0.0
            for task in tasks:
                # Calculate task time
                if task['type'] == 'forward':
                    task_time = forward_time_per_layer
                else:
                    task_time = backward_time_per_layer
                
                # Update current time
                current_time += task_time
            
            device_times[device_id] = current_time
            total_time = max(total_time, current_time)
        
        return {
            'total_time': total_time,
            'device_times': device_times,
            'average_device_utilization': sum(device_times.values()) / (len(device_times) * total_time) if device_times else 0
        }
    
    def get_pipeline_info(self) -> Dict:
        """
        Get pipeline information
        
        Returns:
            Dict with pipeline details
        """
        return {
            'total_layers': self.total_layers,
            'M_layers': self.M_layers,
            'pipeline_stages': self.P,
            'total_super_neurons': len(self.super_neurons),
            'super_neurons_per_device': {device_id: len(sns) for device_id, sns in self.sn_by_device.items()}
        }
