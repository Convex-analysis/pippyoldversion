# FLAD: Federated Learning for Autonomous Driving

This package implements advanced pipeline optimization algorithms for Federated Learning in Autonomous Driving (FLAD). It provides tools for partitioning Deep Neural Network (DNN) models across vehicles with heterogeneous capabilities while determining optimal execution orders.

## Overview

FLAD (Federated Learning for Autonomous Driving) addresses the challenge of distributing deep learning model training and inference across multiple vehicles with varying computational capabilities. The system treats a DNN model as a directed acyclic graph (DAG) where nodes represent modules like convolutional layers, attention mechanisms, etc., and edges represent data dependencies.

## Key Components

- `model.py`: Defines DNN model structure, components, and partitioning mechanisms
- `vehicle.py`: Implements vehicle abstractions with computation, memory, and communication capabilities
- `optimizer.py`: Contains the baseline pipeline optimizer using exhaustive search with pruning
- `swift_optimizer.py`: Implements the SWIFT algorithm for fast, efficient pipeline optimization
- `dqn_model.py`: Deep Q-Network implementation for reinforcement learning-based optimization
- `visualizer.py`: Visualization tools for cluster DAGs and pipeline schedules
- `swift_example.py`: Example usage of the SWIFT optimization algorithm
- `cluster_util.py`: Utilities for cluster configuration, XML import/export, and visualization
- `model_util.py`: Utilities for model configuration, XML import/export, and visualization

## Optimization Algorithms

### Baseline Optimizer
The baseline optimizer (`PipelineOptimizer`) finds optimal pipelines through exhaustive search of valid execution paths, with greedy partition assignment strategies.

### SWIFT Optimizer
SWIFT (Speedy Weight-based Intelligent Fast Two-phase scheduler) is an optimized algorithm that:

1. Uses stability scoring to identify reliable vehicles
2. Implements a two-phase approach:
   - Phase 1: Initial pipeline generation using greedy, stability-aware assignment
   - Phase 2: DQN-based optimization for generating alternative pipelines

## Mathematical Model

The optimization is based on the following formulation:

### Computation Time
```
t_compute(v) = (FLOPs * batch_size * memory_overhead) / (comp_capability * utilization)
```

### Communication Time
```
t_comm(v) = (2 * comm_volume * batch_size * memory_overhead) / comm_capability
```

### Total Pipeline Time
```
t_pipeline = ∑ t_compute(v) + ∑ t_comm(v)
```

### Constraints
1. Memory constraints: Sum of partition capacities assigned to a vehicle must not exceed its memory
2. DAG precedence constraints: Execution order must respect dependencies between vehicles
3. Complete assignment: All model partitions must be assigned to exactly one vehicle

## XML Configuration

FLAD supports defining both cluster and model configurations through XML files, enabling easy reuse, sharing, and modification of configurations.

### Cluster XML Format
```xml
<cluster name="ClusterName">
    <vehicles>
        <vehicle id="v1" memory="4e9" comp_capability="8e9" comm_capability="1e9" />
        <vehicle id="v2" memory="5e9" comp_capability="12e9" comm_capability="1.2e9" />
    </vehicles>
    <dependencies>
        <dependency source="v1" target="v3" />
        <dependency source="v2" target="v4" />
    </dependencies>
</cluster>
```

### Model XML Format
```xml
<model name="ModelName">
    <components>
        <component name="RGB_Backbone" flops_per_sample="5e9" capacity="2e9" />
        <component name="Lidar_Backbone" flops_per_sample="8e9" capacity="3e9" />
    </components>
    <partitions>
        <partition name="RGB_Only" communication_volume="0.5e9">
            <component_ref name="RGB_Backbone" />
        </partition>
        <partition name="Lidar_Only" communication_volume="0.8e9">
            <component_ref name="Lidar_Backbone" />
        </partition>
    </partitions>
</model>
```

## Usage

### Basic Usage
```python
from flad.model import ModelComponent, UnitModelPartition, DNNModel
from flad.vehicle import Vehicle, FLADCluster
from flad.optimizer import PipelineOptimizer

# Create model components
rgb_backbone = ModelComponent("RGB_Backbone", flops_per_sample=5e9, capacity=2e9)
lidar_backbone = ModelComponent("Lidar_Backbone", flops_per_sample=8e9, capacity=3e9)

# Create model partitions
model = DNNModel("MyModel")
model.add_component(rgb_backbone)
model.add_component(lidar_backbone)
model.add_unit_partition(UnitModelPartition("RGB", [rgb_backbone], ...))

# Create vehicles
cluster = FLADCluster("MyCluster")
cluster.add_vehicle(Vehicle("v1", memory=4e9, comp_capability=8e9, comm_capability=1e9))
cluster.add_vehicle(Vehicle("v2", memory=5e9, comp_capability=12e9, comm_capability=1.2e9))
cluster.add_dependency("v1", "v2")

# Run optimization
optimizer = PipelineOptimizer(cluster, model)
path, strategy, execution_time = optimizer.optimize()
```

### SWIFT Optimization with XML Configuration
```python
from flad.swift_optimizer import SWIFTOptimizer
from flad.cluster_util import load_cluster_from_xml
from flad.model_util import load_model_from_xml

# Load model and cluster from XML files
cluster = load_cluster_from_xml("configs/my_cluster.xml")
model = load_model_from_xml("configs/my_model.xml")

# Create SWIFT optimizer
swift_optimizer = SWIFTOptimizer(
    cluster, 
    model, 
    batch_size=32,
    utilization=0.5, 
    memory_overhead=1.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train the DQN model
rewards = swift_optimizer.train_dqn_model(episodes=100)

# Find optimal pipelines
pipelines = swift_optimizer.optimize()

# Get best pipeline
best_path, best_strategy, best_time = swift_optimizer.get_best_pipeline()
```

### Command-line Interface
FLAD includes a command-line interface for running the SWIFT optimization workflow:

```bash
python swift_example.py --cluster=configs/my_cluster.xml --model=configs/my_model.xml --episodes=100 --output=results
```

Options:
- `--cluster`, `-c`: Path to cluster XML configuration file
- `--model`, `-m`: Path to model XML configuration file
- `--output`, `-o`: Output directory for results (default: "output")
- `--episodes`, `-e`: Number of training episodes (default: 50)
- `--no-baseline`: Skip comparison with baseline optimizer
- `--cpu`: Force CPU usage even if CUDA is available

## Visualization

The package provides visualization capabilities through the `visualizer` module:

```python
from flad.visualizer import plot_cluster_dag, plot_pipeline_schedule
from flad.cluster_util import generate_cluster_dag
from flad.model_util import model_to_graphviz

# Visualize cluster dependencies
generate_cluster_dag(cluster, output_path="cluster_dag.png", show=True)

# Visualize model structure
model_to_graphviz(model, output_path="model_graph", show_components=True)

# Visualize execution schedule of a pipeline
plot_pipeline_schedule(path, partition_strategy, execution_times)

# Or use SWIFT's built-in visualizer
swift_optimizer.visualize_pipelines(save_path="pipelines.png")
```

## Examples

The repository includes several example files:
- `swift_example.py`: Main example for XML-based workflow with SWIFT optimizer
- `cluster_example.py`: Examples for cluster configuration and utilities
- `model_example.py`: Examples for model configuration and utilities

Example configuration XML files can be found in the `example_configs` directory.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- NetworkX
- Matplotlib
- tqdm
- Graphviz (optional, for enhanced model visualization)