
# FLAD Pipeline Optimizer

This package implements a pipeline generation algorithm for Federated Learning in Autonomous Driving (FLAD). It optimizes the partitioning of Deep Neural Network (DNN) models across vehicles with different capabilities and determines the optimal execution order.

## Background

The algorithm treats a DNN model as a directed acyclic graph (DAG) where nodes represent modules like Conv, MaxPool, Attention, etc., and edges encode data dependencies. The optimization problem involves:

1. Partitioning the model across vehicles with different memory and computation capabilities
2. Determining the execution order to minimize total training time
3. Respecting memory constraints and execution dependencies

## Components

- `model.py`: Defines the DNN model structure and partitioning
- `vehicle.py`: Represents vehicles and their capabilities in the FLAD cluster
- `optimizer.py`: Implements the optimization algorithm to find the optimal partition strategy and execution order
- `visualizer.py`: Provides visualization tools for the cluster DAG and pipeline schedule
- `main.py`: Demonstrates the optimizer with a sample model and cluster

## Mathematical Formulation

The implementation is based on the following formulation:

### Computation Time
```
t_cmp = (M_cmp * ν) / (cmp_v * μ)
```
where:
- M_cmp is the computation cost (FLOPS)
- ν is the memory bandwidth overhead factor
- cmp_v is the computation capability of vehicle v
- μ is the GPU utilization factor

### Communication Time
```
t_com = (2 * M_cap * N_batch * ν) / com_v
```
where:
- M_cap is the model capacity
- N_batch is the batch size
- com_v is the communication capability of vehicle v

### Objective Function
```
min t_path(p, P) = Σ(t_cmp) + Σ(t_com)
```

## Usage

```python
from flad.model import ModelComponent, UnitModelPartition, DNNModel
from flad.vehicle import Vehicle, FLADCluster
from flad.optimizer import PipelineOptimizer

# Create your model
model = DNNModel("YourModel")
# Add components and partitions...

# Create your cluster
cluster = FLADCluster("YourCluster")
# Add vehicles and dependencies...

# Run optimization
optimizer = PipelineOptimizer(cluster, model)
path, strategy, time = optimizer.optimize()
```

## Requirements

- Python 3.7+
- NumPy
- NetworkX
- Matplotlib