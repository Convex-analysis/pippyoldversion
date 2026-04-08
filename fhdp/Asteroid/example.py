"""Asteroid example usage"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
from .profiler import AsteroidProfiler
from .planner import AsteroidPlanner
from .worker import AsteroidWorker
from .fault_tolerance import FaultToleranceManager

class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def asteroid_example():
    """
    Asteroid example usage
    """
    print("=== Asteroid Example ===")
    
    # 1. Preprocessing Phase
    print("\nStep 1: Preprocessing Phase (Profiling)")
    
    # Create model
    model = SimpleModel()
    input_shape = (32, 3, 32, 32)  # (batch_size, channels, height, width)
    
    # Create profiler
    profiler = AsteroidProfiler()
    
    # Profile model
    profiler_results = profiler.profile_model(model, input_shape)
    
    # Profile bandwidth
    devices = ["jetson_nano_1", "jetson_nano_2", "jetson_nano_3", "jetson_nano_4", "jetson_nano_5"]
    bandwidth_matrix = profiler.profile_bandwidth(devices)
    
    # Save profiling results
    profiler.save_results("profiling_results.json")
    
    # 2. Planning Phase
    print("\nStep 2: Planning Phase (Generating HPP Plan)")
    
    # Device specifications (Env A: 5×Nano @100Mbps)
    device_specs = {
        "jetson_nano_1": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
        "jetson_nano_2": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
        "jetson_nano_3": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
        "jetson_nano_4": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
        "jetson_nano_5": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0}
    }
    
    # Add compute capabilities to profiler results
    profiler.profiling_results['compute_capabilities'] = {
        "jetson_nano_1": 1.0,
        "jetson_nano_2": 1.0,
        "jetson_nano_3": 1.0,
        "jetson_nano_4": 1.0,
        "jetson_nano_5": 1.0
    }
    
    # Create planner
    planner = AsteroidPlanner(profiler.profiling_results, device_specs)
    
    # Generate plan
    global_batch_size = 2048
    num_stages = 3
    plan = planner.generate_plan(model, input_shape, global_batch_size, num_stages)
    
    # Print plan
    print(f"Generated HPP plan:")
    print(f"Global batch size: {plan['global_batch_size']}")
    print(f"Number of stages: {plan['num_stages']}")
    print(f"Total latency: {plan['total_latency']:.4f} seconds")
    print("Stages:")
    for i, stage in enumerate(plan['stages']):
        print(f"  Stage {i}:")
        print(f"    Layers: {len(stage['layers'])} layers")
        print(f"    Devices: {stage['devices']}")
        print(f"    K_p: {stage['K_p']}")
        print(f"    Micro-batch size: {stage['micro_batch_size']}")
    
    # Save plan
    planner.save_plan("hpp_plan.json")
    
    # 3. Execution Phase
    print("\nStep 3: Execution Phase (Running Training)")
    
    # Create fault tolerance manager
    ft_manager = FaultToleranceManager(devices)
    ft_manager.start()
    
    # Create workers for each device
    workers = {}
    for device in devices:
        # Create model replica
        model_replica = SimpleModel()
        # Create worker
        worker = AsteroidWorker(model_replica, "cuda" if torch.cuda.is_available() else "cpu", ft_manager)
        worker.start()
        workers[device] = worker
    
    # Simulate training
    print("Simulating training...")
    
    # Create dummy batch
    dummy_input = torch.randn(input_shape)
    dummy_target = torch.randint(0, 10, (input_shape[0],))
    dummy_batch = (dummy_input, dummy_target)
    
    # Run a training step for each stage
    for i, stage in enumerate(plan['stages']):
        for device in stage['devices']:
            worker = workers.get(device)
            if worker:
                worker.run_step(dummy_batch, i, num_stages)
                worker.update_weights()
                print(f"Ran training step on {device} for stage {i}")
    
    # Stop workers
    for worker in workers.values():
        worker.stop()
    
    # Stop fault tolerance manager
    ft_manager.stop()
    
    print("\n=== Asteroid Example Complete ===")

if __name__ == "__main__":
    asteroid_example()
