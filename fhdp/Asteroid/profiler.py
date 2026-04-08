"""Asteroid Profiler for device performance collection"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import json

class AsteroidProfiler:
    """Asteroid Profiler for collecting device performance metrics"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the profiler
        
        Args:
            device: Device to profile (e.g., 'cuda', 'cpu')
        """
        self.device = device
        self.profiling_results = {}
        
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...], batch_sizes: List[int] = None) -> Dict:
        """
        Profile a model across different batch sizes
        
        Args:
            model: PyTorch model to profile
            input_shape: Input shape (batch_size, *)
            batch_sizes: List of batch sizes to profile
            
        Returns:
            Profiling results dictionary
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        model = model.to(self.device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create input tensor
                input_tensor = torch.randn((batch_size,) + input_shape[1:], device=self.device)
                
                # Profile forward pass
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                forward_time = time.time() - start_time
                
                # Profile backward pass (requires gradients)
                model.train()
                input_tensor.requires_grad_(True)
                start_time = time.time()
                output = model(input_tensor)
                loss = output.sum()
                loss.backward()
                backward_time = time.time() - start_time
                model.eval()
                
                # Calculate activation size
                activation_size = self._calculate_activation_size(model)
                
                # Calculate model size
                model_size = self._calculate_model_size(model)
                
                results[batch_size] = {
                    'forward_time': forward_time,
                    'backward_time': backward_time,
                    'total_time': forward_time + backward_time,
                    'activation_size': activation_size,
                    'model_size': model_size
                }
                
                print(f"Batch size {batch_size}: forward={forward_time:.4f}s, backward={backward_time:.4f}s, activation={activation_size/1e6:.2f}MB, model={model_size/1e6:.2f}MB")
                
            except Exception as e:
                print(f"Error profiling batch size {batch_size}: {e}")
                results[batch_size] = {'error': str(e)}
        
        self.profiling_results['model'] = results
        return results
    
    def profile_bandwidth(self, devices: List[str]) -> Dict:
        """
        Profile device-to-device bandwidth
        
        Args:
            devices: List of device identifiers
            
        Returns:
            Bandwidth matrix (Gbps)
        """
        # This is a placeholder implementation
        # In practice, you would measure actual D2D bandwidth
        bandwidth_matrix = {}
        for i, dev1 in enumerate(devices):
            bandwidth_matrix[dev1] = {}
            for j, dev2 in enumerate(devices):
                if i == j:
                    bandwidth = float('inf')
                else:
                    # Simulate bandwidth based on device types
                    if 'nx' in dev1.lower() and 'nx' in dev2.lower():
                        bandwidth = 1.0  # 1Gbps
                    elif 'tx2' in dev1.lower() or 'tx2' in dev2.lower():
                        bandwidth = 0.5  # 500Mbps
                    else:  # Nano
                        bandwidth = 0.1  # 100Mbps
                bandwidth_matrix[dev1][dev2] = bandwidth
        
        self.profiling_results['bandwidth'] = bandwidth_matrix
        return bandwidth_matrix
    
    def _calculate_activation_size(self, model: nn.Module) -> int:
        """
        Calculate activation size by tracing the model
        
        Args:
            model: PyTorch model
            
        Returns:
            Activation size in bytes
        """
        # This is a simplified implementation
        # In practice, you would use torch.jit.trace to track activations
        activation_size = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Approximate activation size based on weight size
                activation_size += param.numel() * 4  # Assume float32
        return activation_size
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """
        Calculate model size
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        model_size = 0
        for param in model.parameters():
            model_size += param.numel() * param.element_size()
        return model_size
    
    def save_results(self, filename: str):
        """
        Save profiling results to a file
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.profiling_results, f, indent=2)
        print(f"Profiling results saved to {filename}")
    
    def load_results(self, filename: str):
        """
        Load profiling results from a file
        
        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            self.profiling_results = json.load(f)
        print(f"Profiling results loaded from {filename}")
    
    def get_compute_capability(self, batch_size: int) -> float:
        """
        Calculate compute capability for a given batch size
        
        Args:
            batch_size: Batch size
            
        Returns:
            Compute capability (inverse of execution time)
        """
        if 'model' not in self.profiling_results:
            raise ValueError("No profiling results available")
        
        if batch_size not in self.profiling_results['model']:
            raise ValueError(f"No profiling results for batch size {batch_size}")
        
        result = self.profiling_results['model'][batch_size]
        if 'error' in result:
            raise ValueError(f"Error in profiling: {result['error']}")
        
        total_time = result['total_time']
        return 1.0 / total_time if total_time > 0 else 0
