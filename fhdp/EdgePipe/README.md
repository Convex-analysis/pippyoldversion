# EdgePipe Implementation

## Overview

EdgePipe is a deep learning framework designed for volatile wireless edge devices, implementing the paper "EdgePipe: Tailoring Pipeline Parallelism With Deep Neural Networks for Volatile Wireless Edge Devices". This implementation includes the core components of EdgePipe, including super neurons, hybrid partitioning, neuron-to-device mapping, pipeline scheduling, and performance analysis.

## Core Components

### 1. Super Neuron

The Super Neuron is a group of neurons across adjacent layers, designed to handle wireless link failures and device malfunctions. It allows local forward/backward computations even when communication between devices fails.

### 2. Hybrid Partitioning

The hybrid partitioning algorithm divides the DNN model into super neurons using a two-stage process:
- Layer-level horizontal split
- Neuron-level vertical split
- Device allocation based on layer count ratio

### 3. Neuron to Device Mapping

Uses a genetic algorithm to find the optimal mapping of super neurons to devices, maximizing the Hadamard product of the super neuron network and device network.

### 4. Pipeline Scheduler

Manages the execution schedule for forward and backward passes, optimizing the pipeline parallelism based on the super neuron structure.

### 5. Performance Analysis

Analyzes the time complexity, speedup, fault tolerance, and scalability of the EdgePipe implementation.

## Usage Example

```python
from fhdp.EdgePipe import HybridPartitioning, NeuronDeviceMapping, PipelineScheduler, PerformanceAnalyzer

# Example configuration
total_layers = 6
total_devices = 4
neurons_per_layer = [784, 128, 128, 128, 128, 10]
batch_size = 100

# Step 1: Perform hybrid partitioning
partitioning = HybridPartitioning(total_layers, total_devices)
super_neuron_network = partitioning.perform_partitioning(neurons_per_layer)

# Step 2: Create device network (PRR matrix)
device_network = np.random.uniform(0.7, 1.0, (total_devices, total_devices))
np.fill_diagonal(device_network, 1.0)

# Step 3: Optimize neuron to device mapping
mapping = NeuronDeviceMapping(super_neuron_network, device_network)
best_mapping = mapping.optimize_mapping(generations=1000)

# Step 4: Generate pipeline schedule
scheduler = PipelineScheduler(super_neuron_network)
schedule = scheduler.generate_schedule(batch_size)

# Step 5: Analyze performance
analyzer = PerformanceAnalyzer(super_neuron_network)
performance_summary = analyzer.get_performance_summary(batch_size)
```

## Running the Example

```bash
python -m fhdp.EdgePipe.example
```

## Jetson Device Deployment

### Prerequisites
- Two Jetson devices (e.g., Jetson Orin and Jetson Nano)
- Python 3.7+ installed on all devices
- PyTorch and torchvision installed on all devices
- Network connectivity between devices

### Usage

#### 1. Start the Server (coordination only)

```bash
python -m fhdp.EdgePipe.edgepipe_jetson --mode server --host 0.0.0.0 --port 5000
```

#### 2. Start Device 0 (Jetson Orin)

```bash
python -m fhdp.EdgePipe.edgepipe_jetson --mode device --role device0 --device-id orin --server-host <server-ip> --server-port 5000
```

#### 3. Start Device 1 (Jetson Nano)

```bash
python -m fhdp.EdgePipe.edgepipe_jetson --mode device --role device1 --device-id nano --server-host <server-ip> --server-port 5000
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| --mode | Run mode (server, device, validate) | Required |
| --host | Server host | 0.0.0.0 |
| --port | Server port | 5000 |
| --server-host | Server host (device mode) | localhost |
| --server-port | Server port (device mode) | 5000 |
| --device-id | Device ID | device_001 |
| --role | Device role (device0, device1) | Required for device mode |
| --device0-id | Device0 ID | orin |
| --device1-id | Device1 ID | nano |
| --template-id | Pipeline template ID | vit_b16_2stage_v1 |
| --rounds | Total training rounds | 1 |
| --micro-batches | Micro-batches per round | 1 |
| --dataset | Dataset name | cifar10 |
| --num-classes | Number of classes | 10 |
| --image-size | Input image size | 224 |
| --data-dir | Dataset root directory | ./data |
| --download | Download CIFAR-10 if missing | False |
| --auto-exit | Exit after completing all rounds | False |
| --listen-host | Device listen host for peer pipeline data | 0.0.0.0 |
| --listen-port | Device listen port for peer pipeline data | 6000 (device0), 6001 (device1) |
| --advertise-host | Host/IP to advertise to peers | Auto-detect |

## Key Features

- **Super Neuron Concept**: Cross-layer neuron groups for fault tolerance
- **Hybrid Partitioning**: Balances parallelism and reliability
- **Genetic Algorithm Mapping**: Optimizes device allocation
- **Pipeline Parallelism**: Improves training speed
- **Performance Analysis**: Provides insights into scalability and fault tolerance

## Performance Benefits

- **Speedup**: Up to 5x faster than horizontal allocation
- **Fault Tolerance**: Maintains training even with device failures
- **Scalability**: Improves with more devices
- **Resource Efficiency**: Optimizes device utilization

## Dependencies

- Python 3.7+
- NumPy

## References

- EdgePipe: Tailoring Pipeline Parallelism With Deep Neural Networks for Volatile Wireless Edge Devices
