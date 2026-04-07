"""Performance analysis for EdgePipe"""

from typing import List, Dict, Tuple
import numpy as np
from .super_neuron import SuperNeuronNetwork
from .pipeline_scheduler import PipelineScheduler

class PerformanceAnalyzer:
    """Performance analyzer for EdgePipe"""
    
    def __init__(self, super_neuron_network: SuperNeuronNetwork):
        """
        Initialize the performance analyzer
        
        Args:
            super_neuron_network: Network of super neurons
        """
        self.super_neuron_network = super_neuron_network
        self.total_layers = super_neuron_network.total_layers
        self.M_layers = super_neuron_network.M_layers
        self.scheduler = PipelineScheduler(super_neuron_network)
    
    def calculate_time_complexity(self, batch_size: int) -> Dict:
        """
        Calculate time complexity
        
        Args:
            batch_size: Number of batches
            
        Returns:
            Time complexity analysis
        """
        L = self.total_layers
        M = self.M_layers
        N = batch_size
        
        # Calculate time slots for different allocations
        if M == 1:  # Vertical allocation
            # Same as non-horizontal allocation
            total_slots = 2 * L + 2 * M * (N - 1) - 1
            time_complexity = f"O({N} * {M})"
        elif M == L:  # Horizontal allocation
            total_slots = (2 * L - 1) * N
            time_complexity = f"O({N} * {L})"
        else:  # Hybrid allocation
            total_slots = 2 * L + 2 * M * (N - 1) - 1
            time_complexity = f"O({N} * {M})"
        
        return {
            'total_slots': total_slots,
            'time_complexity': time_complexity,
            'allocation_type': self._get_allocation_type()
        }
    
    def calculate_speedup(self, batch_size: int) -> float:
        """
        Calculate speedup compared to horizontal allocation
        
        Args:
            batch_size: Number of batches
            
        Returns:
            Speedup factor
        """
        L = self.total_layers
        M = self.M_layers
        N = batch_size
        
        if M == L:  # Horizontal allocation
            return 1.0
        
        # Time for horizontal allocation
        horizontal_slots = (2 * L - 1) * N
        
        # Time for current allocation
        current_slots = 2 * L + 2 * M * (N - 1) - 1
        
        # Speedup
        speedup = horizontal_slots / current_slots
        
        return speedup
    
    def analyze_scalability(self, max_devices: int, neurons_per_layer: List[int]) -> Dict:
        """
        Analyze scalability with increasing number of devices
        
        Args:
            max_devices: Maximum number of devices to analyze
            neurons_per_layer: List of neurons per layer
            
        Returns:
            Scalability analysis
        """
        from .partitioning import HybridPartitioning
        
        results = []
        
        for device_count in range(1, max_devices + 1):
            # Create partitioning for current device count
            partitioning = HybridPartitioning(self.total_layers, device_count)
            sn_network = partitioning.perform_partitioning(neurons_per_layer)
            
            # Create analyzer for this configuration
            analyzer = PerformanceAnalyzer(sn_network)
            
            # Calculate metrics
            time_complexity = analyzer.calculate_time_complexity(100)
            speedup = analyzer.calculate_speedup(100)
            
            results.append({
                'device_count': device_count,
                'M_layers': sn_network.M_layers,
                'total_super_neurons': sn_network.get_total_super_neurons(),
                'time_slots': time_complexity['total_slots'],
                'speedup': speedup
            })
        
        return {'scalability_results': results}
    
    def analyze_fault_tolerance(self, failure_probability: float, batch_size: int) -> Dict:
        """
        Analyze fault tolerance
        
        Args:
            failure_probability: Probability of device failure
            batch_size: Number of batches
            
        Returns:
            Fault tolerance analysis
        """
        # Calculate probability of successful completion
        total_devices = len(set(sn.get_device() for sn in self.super_neuron_network.get_super_neurons()))
        
        # Probability that all devices survive for the entire training
        success_probability = (1 - failure_probability) ** total_devices
        
        # Expected number of successful batches
        expected_successful_batches = batch_size * success_probability
        
        return {
            'failure_probability': failure_probability,
            'total_devices': total_devices,
            'success_probability': success_probability,
            'expected_successful_batches': expected_successful_batches
        }
    
    def _get_allocation_type(self) -> str:
        """
        Get allocation type
        
        Returns:
            Allocation type
        """
        if self.M_layers == 1:
            return "Vertical"
        elif self.M_layers == self.total_layers:
            return "Horizontal"
        else:
            return "Hybrid"
    
    def get_performance_summary(self, batch_size: int) -> Dict:
        """
        Get performance summary
        
        Args:
            batch_size: Number of batches
            
        Returns:
            Performance summary
        """
        time_complexity = self.calculate_time_complexity(batch_size)
        speedup = self.calculate_speedup(batch_size)
        pipeline_info = self.scheduler.get_pipeline_info()
        
        return {
            'allocation_type': self._get_allocation_type(),
            'total_layers': self.total_layers,
            'M_layers': self.M_layers,
            'pipeline_stages': pipeline_info['pipeline_stages'],
            'time_complexity': time_complexity['time_complexity'],
            'total_slots': time_complexity['total_slots'],
            'speedup': speedup,
            'super_neurons_per_device': pipeline_info['super_neurons_per_device']
        }
