"""Memory model for Asteroid"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

class MemoryModel:
    """Memory model for calculating memory requirements"""
    
    def __init__(self, profiler_results: Dict):
        """
        Initialize the memory model
        
        Args:
            profiler_results: Profiling results from AsteroidProfiler
        """
        self.profiler_results = profiler_results
    
    def calculate_memory(self, stage_layers: List[int], batch_size: int, K_p: int, optimizer: str = 'sgd') -> Dict:
        """
        Calculate memory requirements for a stage
        
        Args:
            stage_layers: List of layer indices in the stage
            batch_size: Micro-batch size
            K_p: Pipeline concurrency for this stage
            optimizer: Optimizer type
            
        Returns:
            Memory requirements in bytes
        """
        # Calculate model memory (Mem^(MOD))
        model_memory = self._calculate_model_memory(stage_layers)
        
        # Calculate optimizer memory (Mem^(OPT))
        optimizer_memory = self._calculate_optimizer_memory(model_memory, optimizer)
        
        # Calculate activation memory (Mem^(ACT))
        activation_memory = self._calculate_activation_memory(batch_size)
        
        # Total memory
        total_memory = model_memory + optimizer_memory + K_p * activation_memory
        
        return {
            'model_memory': model_memory,
            'optimizer_memory': optimizer_memory,
            'activation_memory': activation_memory,
            'total_memory': total_memory,
            'K_p': K_p
        }
    
    def _calculate_model_memory(self, stage_layers: List[int]) -> int:
        """
        Calculate model weight memory for a stage
        
        Args:
            stage_layers: List of layer indices in the stage
            
        Returns:
            Model memory in bytes
        """
        # This is a simplified implementation
        # In practice, you would calculate the memory for the specific layers
        if 'model' not in self.profiler_results:
            raise ValueError("No model profiling results available")
        
        # Get model size from profiling results
        # Assuming we have at least one batch size profiled
        batch_sizes = list(self.profiler_results['model'].keys())
        if not batch_sizes:
            raise ValueError("No batch sizes profiled")
        
        first_batch_size = batch_sizes[0]
        model_size = self.profiler_results['model'][first_batch_size].get('model_size', 0)
        
        # Scale based on number of layers
        total_layers = len(stage_layers)
        # Assuming uniform layer sizes for simplicity
        return model_size * total_layers / 10  # Arbitrary scaling factor
    
    def _calculate_optimizer_memory(self, model_memory: int, optimizer: str) -> int:
        """
        Calculate optimizer state memory
        
        Args:
            model_memory: Model weight memory
            optimizer: Optimizer type
            
        Returns:
            Optimizer memory in bytes
        """
        # Different optimizers have different memory requirements
        if optimizer == 'sgd':
            # SGD with momentum: 2x model memory
            return model_memory * 2
        elif optimizer == 'adam':
            # Adam: 4x model memory
            return model_memory * 4
        else:
            # Default to 2x
            return model_memory * 2
    
    def _calculate_activation_memory(self, batch_size: int) -> int:
        """
        Calculate activation memory for a given batch size
        
        Args:
            batch_size: Micro-batch size
            
        Returns:
            Activation memory in bytes
        """
        if 'model' not in self.profiler_results:
            raise ValueError("No model profiling results available")
        
        # Get activation size from profiling results
        if batch_size in self.profiler_results['model']:
            return self.profiler_results['model'][batch_size].get('activation_size', 0)
        else:
            # Interpolate if batch size not profiled
            batch_sizes = sorted([bs for bs in self.profiler_results['model'].keys() if isinstance(bs, int)])
            if not batch_sizes:
                raise ValueError("No batch sizes profiled")
            
            # Find closest batch sizes
            lower_bs = max([bs for bs in batch_sizes if bs <= batch_size], default=batch_sizes[0])
            upper_bs = min([bs for bs in batch_sizes if bs >= batch_size], default=batch_sizes[-1])
            
            if lower_bs == upper_bs:
                return self.profiler_results['model'][lower_bs].get('activation_size', 0)
            
            # Linear interpolation
            lower_act = self.profiler_results['model'][lower_bs].get('activation_size', 0)
            upper_act = self.profiler_results['model'][upper_bs].get('activation_size', 0)
            
            ratio = (batch_size - lower_bs) / (upper_bs - lower_bs)
            return int(lower_act + ratio * (upper_act - lower_act))
    
    def calculate_optimal_K_p(self, stage_index: int, total_stages: int) -> int:
        """
        Calculate optimal K_p for a stage
        
        Args:
            stage_index: Stage index (0-based)
            total_stages: Total number of stages
            
        Returns:
            Optimal K_p value
        """
        # Formula from the paper: K_p = 2 × (P - p) - 1
        # where P is total stages, p is stage index (0-based)
        return 2 * (total_stages - stage_index) - 1
    
    def check_memory_constraint(self, memory_requirement: int, device_memory: int) -> bool:
        """
        Check if memory requirement is within device memory
        
        Args:
            memory_requirement: Memory requirement in bytes
            device_memory: Device memory in bytes
            
        Returns:
            True if memory requirement is within device memory
        """
        # Add 10% buffer
        return memory_requirement * 1.1 <= device_memory
