"""Asteroid Planner for HPP configuration generation"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from .memory_model import MemoryModel

class AsteroidPlanner:
    """Asteroid Planner for generating HPP configurations"""
    
    def __init__(self, profiler_results: Dict, device_specs: Dict):
        """
        Initialize the planner
        
        Args:
            profiler_results: Profiling results from AsteroidProfiler
            device_specs: Device specifications (memory, compute capability)
        """
        self.profiler_results = profiler_results
        self.device_specs = device_specs
        self.memory_model = MemoryModel(profiler_results)
        self.plan = None
    
    def generate_plan(self, model: nn.Module, input_shape: Tuple[int, ...], global_batch_size: int, num_stages: int) -> Dict:
        """
        Generate HPP plan using dynamic programming
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            global_batch_size: Global batch size
            num_stages: Number of pipeline stages
            
        Returns:
            HPP plan
        """
        # Step 1: Get model layers
        layers = self._get_model_layers(model)
        num_layers = len(layers)
        
        # Step 2: Get device list sorted by memory (descending)
        devices = sorted(self.device_specs.keys(), key=lambda x: self.device_specs[x]['memory'], reverse=True)
        num_devices = len(devices)
        
        # Step 3: Initialize DP table
        # Q(l, n, p) = optimal HPP latency when last l layers are split into p stages deployed on last n devices
        Q = {}
        
        # Base case: p=1
        for l in range(1, num_layers + 1):
            for n in range(1, num_devices + 1):
                key = (l, n, 1)
                Q[key] = self._calculate_stage_latency(layers[:l], devices[:n], global_batch_size, 0, num_stages)
        
        # Fill DP table for p > 1
        for p in range(2, num_stages + 1):
            for l in range(p, num_layers + 1):
                for n in range(p, num_devices + 1):
                    min_latency = float('inf')
                    best_split = None
                    
                    # Try all possible splits
                    for l_prime in range(p-1, l):
                        for n_prime in range(p-1, n):
                            if (l_prime, n_prime, p-1) in Q:
                                # Calculate latency for this split
                                sub_latency = Q[(l_prime, n_prime, p-1)]
                                new_latency = self._calculate_stage_latency(
                                    layers[l_prime:l], 
                                    devices[n_prime:n], 
                                    global_batch_size, 
                                    p-1, 
                                    num_stages
                                )
                                # The dominant step is the maximum of sub-latency and new-latency
                                dominant_latency = max(sub_latency, new_latency)
                                
                                if dominant_latency < min_latency:
                                    min_latency = dominant_latency
                                    best_split = (l_prime, n_prime)
                    
                    if best_split:
                        Q[(l, n, p)] = min_latency
        
        # Step 4: Extract the optimal plan
        self.plan = self._extract_plan(Q, layers, devices, num_layers, num_devices, num_stages, global_batch_size)
        
        return self.plan
    
    def _get_model_layers(self, model: nn.Module) -> List[str]:
        """
        Get model layers
        
        Args:
            model: PyTorch model
            
        Returns:
            List of layer names
        """
        layers = []
        def extract_layers(module, prefix=''):
            for name, child in module.named_children():
                layer_name = f"{prefix}.{name}" if prefix else name
                layers.append(layer_name)
                extract_layers(child, layer_name)
        extract_layers(model)
        return layers
    
    def _calculate_stage_latency(self, layers: List[str], devices: List[str], global_batch_size: int, stage_index: int, total_stages: int) -> float:
        """
        Calculate latency for a stage
        
        Args:
            layers: List of layers in the stage
            devices: List of devices in the stage
            global_batch_size: Global batch size
            stage_index: Stage index
            total_stages: Total number of stages
            
        Returns:
            Latency in seconds
        """
        # Calculate optimal K_p
        K_p = self.memory_model.calculate_optimal_K_p(stage_index, total_stages)
        
        # Calculate micro-batch size
        num_devices = len(devices)
        micro_batch_size = global_batch_size // (K_p * num_devices)
        
        # Calculate execution time
        # This is a simplified implementation
        # In practice, you would use profiling results
        execution_time = 0.0
        for layer in layers:
            # Simulate layer execution time
            execution_time += 0.001  # 1ms per layer
        
        # Calculate AllReduce time
        # This is a simplified implementation
        allreduce_time = 0.0
        if num_devices > 1:
            # Estimate AllReduce time based on data size
            data_size = 10 * 1024 * 1024  # 10MB
            bandwidth = min([self.profiler_results['bandwidth'][d1][d2] for d1 in devices for d2 in devices if d1 != d2])
            if bandwidth > 0:
                allreduce_time = (data_size * 8) / (bandwidth * 1e9)  # Convert to seconds
        
        # Calculate total latency
        # HPP-Round Latency = max(T^w + T^e + T^a)
        total_latency = execution_time + allreduce_time
        
        return total_latency
    
    def _extract_plan(self, Q: Dict, layers: List[str], devices: List[str], num_layers: int, num_devices: int, num_stages: int, global_batch_size: int) -> Dict:
        """
        Extract the optimal plan from the DP table
        
        Args:
            Q: DP table
            layers: List of layers
            devices: List of devices
            num_layers: Number of layers
            num_devices: Number of devices
            num_stages: Number of stages
            global_batch_size: Global batch size
            
        Returns:
            HPP plan
        """
        plan = {
            'stages': [],
            'global_batch_size': global_batch_size,
            'num_stages': num_stages,
            'total_latency': Q.get((num_layers, num_devices, num_stages), float('inf'))
        }
        
        # Backtrack to find the split points
        current_l = num_layers
        current_n = num_devices
        current_p = num_stages
        
        while current_p > 0:
            if current_p == 1:
                # Base case
                stage_layers = layers[:current_l]
                stage_devices = devices[:current_n]
                K_p = self.memory_model.calculate_optimal_K_p(current_p-1, num_stages)
                micro_batch_size = global_batch_size // (K_p * len(stage_devices))
                
                plan['stages'].insert(0, {
                    'layers': stage_layers,
                    'devices': stage_devices,
                    'K_p': K_p,
                    'micro_batch_size': micro_batch_size
                })
                break
            
            # Find the best split
            min_latency = float('inf')
            best_split = None
            
            for l_prime in range(current_p-1, current_l):
                for n_prime in range(current_p-1, current_n):
                    if (l_prime, n_prime, current_p-1) in Q:
                        sub_latency = Q[(l_prime, n_prime, current_p-1)]
                        new_latency = self._calculate_stage_latency(
                            layers[l_prime:current_l], 
                            devices[n_prime:current_n], 
                            global_batch_size, 
                            current_p-1, 
                            num_stages
                        )
                        dominant_latency = max(sub_latency, new_latency)
                        
                        if dominant_latency == Q[(current_l, current_n, current_p)]:
                            best_split = (l_prime, n_prime)
                            break
                if best_split:
                    break
            
            if not best_split:
                break
            
            l_prime, n_prime = best_split
            
            # Add current stage
            stage_layers = layers[l_prime:current_l]
            stage_devices = devices[n_prime:current_n]
            K_p = self.memory_model.calculate_optimal_K_p(current_p-1, num_stages)
            micro_batch_size = global_batch_size // (K_p * len(stage_devices))
            
            plan['stages'].insert(0, {
                'layers': stage_layers,
                'devices': stage_devices,
                'K_p': K_p,
                'micro_batch_size': micro_batch_size
            })
            
            # Update current state
            current_l = l_prime
            current_n = n_prime
            current_p -= 1
        
        return plan
    
    def assign_micro_batches(self, stage_index: int, devices: List[str], micro_batch_size: int) -> Dict:
        """
        Assign micro-batches to devices using memory-aware balancing and straggler offloading
        
        Args:
            stage_index: Stage index
            devices: List of devices in the stage
            micro_batch_size: Micro-batch size
            
        Returns:
            Micro-batch assignment
        """
        # Step 1: Memory-aware balancing
        assignment = {device: 0 for device in devices}
        
        # Calculate compute capabilities
        compute_capabilities = {}
        for device in devices:
            try:
                compute_capabilities[device] = self.profiler_results['compute_capabilities'].get(device, 1.0)
            except:
                # Fallback to 1.0 if no compute capability available
                compute_capabilities[device] = 1.0
        
        # Total compute capability
        total_capability = sum(compute_capabilities.values())
        
        # Initial assignment based on compute capability
        total_micro_batches = len(devices)  # Simplified
        for device in devices:
            assignment[device] = int(total_micro_batches * compute_capabilities[device] / total_capability)
        
        # Distribute remaining micro-batches
        remaining = total_micro_batches - sum(assignment.values())
        for i in range(remaining):
            # Assign to device with highest compute capability
            device = max(devices, key=lambda x: compute_capabilities[x])
            assignment[device] += 1
        
        # Step 2: Straggler offloading
        # This is a simplified implementation
        # In practice, you would iteratively migrate load from slowest to fastest devices
        
        return assignment
    
    def save_plan(self, filename: str):
        """
        Save plan to a file
        
        Args:
            filename: Output filename
        """
        if self.plan is None:
            raise ValueError("No plan generated yet")
        
        import json
        with open(filename, 'w') as f:
            json.dump(self.plan, f, indent=2)
        print(f"Plan saved to {filename}")
    
    def load_plan(self, filename: str):
        """
        Load plan from a file
        
        Args:
            filename: Input filename
        """
        import json
        with open(filename, 'r') as f:
            self.plan = json.load(f)
        print(f"Plan loaded from {filename}")
