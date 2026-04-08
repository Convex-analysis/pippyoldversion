# Asteroid Implementation

## Overview

Asteroid is a resource-efficient hybrid pipeline parallelism framework for collaborative DNN training on heterogeneous edge devices, implementing the paper "Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices". This implementation includes the core components of Asteroid, including performance profiler, HPP planner, micro-batch scheduler, fault tolerance manager, and training worker.

## Core Components

### 1. AsteroidProfiler

The Asteroid Profiler collects device performance metrics, including:
- Layer execution times for different batch sizes
- Activation sizes
- Model weights sizes
- Device-to-device bandwidth

### 2. MemoryModel

The Memory Model calculates memory requirements based on the formula:
```
Mem_p(β) = Mem^(MOD)_p + Mem^(OPT)_p + K_p × Mem^(ACT)_p(β)
```
- `Mem^(MOD)`: Model weight memory
- `Mem^(OPT)`: Optimizer state memory
- `Mem^(ACT)`: Single micro-batch activation memory
- `K_p`: Pipeline concurrency for stage p

### 3. AsteroidPlanner

The Asteroid Planner generates HPP configurations using dynamic programming, including:
- Model layer partitioning
- Device grouping
- Micro-batch size assignment
- K_p calculation

### 4. MicroBatchScheduler

The Micro-batch Scheduler implements 1F1B scheduling strategy, which:
- Processes forward pass for micro-batch i
- Then processes backward pass for micro-batch i-P+1
- Balances memory usage and parallel efficiency

### 5. FaultToleranceManager

The Fault Tolerance Manager handles device failures with:
- Heartbeat monitoring
- Model replication
- Lightweight layer migration
- Pipeline reconfiguration

### 6. AsteroidWorker

The Asteroid Worker executes the training process with:
- In-memory task pool
- Model executor
- Tensor dispatcher
- 1F1B micro-batch scheduling

## Three-Stage Workflow

### 1. Preprocessing Phase (Offline)
- **Asteroid Profiler** collects device performance
- Records: layer execution times, activation sizes, bandwidth

### 2. Planning Phase (Planning)
- **Asteroid Planner** generates HPP configuration
- Outputs: model split points, device groups, batch allocation

### 3. Execution Phase (Execution)
- **Asteroid Worker** executes training
- 1F1B micro-batch scheduling + fault tolerance replay mechanism

## Hybrid Pipeline Parallelism (HPP) Architecture

- **Inter-group**: Pipeline Parallelism
- **Intra-group**: Data Parallelism + Ring AllReduce
- **Advantage**: Reduces communication by 1.9×-2.7× compared to pure DP

## Usage Example

```python
from fhdp.Asteroid import AsteroidProfiler, AsteroidPlanner, AsteroidWorker, FaultToleranceManager

# 1. Preprocessing Phase
profiler = AsteroidProfiler()
profiler_results = profiler.profile_model(model, input_shape)
bandwidth_matrix = profiler.profile_bandwidth(devices)
profiler.save_results("profiling_results.json")

# 2. Planning Phase
device_specs = {
    "jetson_nano_1": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
    "jetson_nano_2": {"memory": 4 * 1024 * 1024 * 1024, "compute_capability": 1.0},
    # Add more devices...
}

planner = AsteroidPlanner(profiler.profiling_results, device_specs)
plan = planner.generate_plan(model, input_shape, global_batch_size, num_stages)
planner.save_plan("hpp_plan.json")

# 3. Execution Phase
ft_manager = FaultToleranceManager(devices)
ft_manager.start()

workers = {}
for device in devices:
    model_replica = SimpleModel()
    worker = AsteroidWorker(model_replica, "cuda" if torch.cuda.is_available() else "cpu", ft_manager)
    worker.start()
    workers[device] = worker

# Run training
for i, stage in enumerate(plan['stages']):
    for device in stage['devices']:
        worker = workers.get(device)
        if worker:
            worker.run_step(batch, i, num_stages)
            worker.update_weights()

# Cleanup
for worker in workers.values():
    worker.stop()
ft_manager.stop()
```

## Running the Example

```bash
python -m fhdp.Asteroid.example
```

## Key Features

- **Resource Efficiency**: Optimizes memory usage with 1F1B scheduling
- **Fault Tolerance**: Handles device failures with lightweight recovery
- **Scalability**: Supports 2-8 device linear scaling
- **Heterogeneity**: Adapts to different device capabilities
- **Communication Reduction**: Uses hybrid parallelism to reduce communication

## Performance Benefits

- **Speedup**: 3.0× - 4.5× faster than single-device training
- **Communication Reduction**: 1.9× - 2.7× less communication than pure DP
- **Fault Recovery**: 14× faster recovery than heavyweight re-planning
- **Memory Optimization**: Reduces peak memory by ~50% with 1F1B scheduling

## Dependencies

- Python 3.7+
- PyTorch
- NumPy

## References

- Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices
