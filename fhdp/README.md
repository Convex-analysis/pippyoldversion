# FHDP (Federated Highway-based Distributed Pipeline) System

A comprehensive federated learning system designed for vehicular environments with decentralized two-level decision hierarchy and transient pipeline constructs.

## Overview

FHDP is a specialized federated learning architecture optimized for vehicular networks, featuring:

- **Decentralized two-level decision hierarchy** with edge server coordination
- **Transient pipeline constructs** for collaborative training
- **Persistent edge server state** for system continuity  
- **Resource-aware participation** with fairness mechanisms
- **Communication-efficient training** with lazy error propagation

## Architecture

### Edge Server Components

- **Mobility Prediction**: DTMC-based vehicle movement prediction
- **Template Management**: Basket-based pipeline template organization (<5ms lookup)
- **Asynchronous Aggregation**: Efficient federated model aggregation
- **Resource Classification**: Dynamic vehicle capability assessment

### Vehicle Layer Components

- **V2V Communication**: Multi-protocol neighbor discovery and messaging
- **Pipeline Formation**: Greedy selection algorithm for optimal pipelines (<1.5s)
- **Training Execution**: Short-horizon training with communication optimization
- **Resource Monitoring**: Real-time resource tracking and participation control

### Core System Features

- **Hybrid Participation**: Individual and pipeline training modes
- **Short-Horizon Training**: 1-2 epochs for vehicular constraints
- **Lazy Error Propagation**: Communication reduction through accumulation
- **Frequency-Based Fairness**: Equitable participation opportunities
- **Asynchronous Coordination**: Non-blocking system operations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fhdp

# Install dependencies
pip install -r requirements.txt

# Install FHDP package
pip install -e .
```

## Quick Start

### 1. Run Edge Server

```bash
python -m fhdp server --port 8080 --coverage 1000x1000
```

### 2. Run Vehicle

```bash
python -m fhdp vehicle vehicle_001 --position 100,50 --velocity 20 --protocols dsrc
```

### 3. Run Simulation

```bash
python -m fhdp simulate --num-vehicles 20 --duration 120
```

### 4. Run Benchmarks

```bash
python -m fhdp benchmark
```

## Configuration

The system uses YAML configuration files. See `config/default_config.yaml` for all available options:

```yaml
system:
  max_vehicles_per_region: 50
  pipeline_formation_interval: 5.0
  enable_pipeline_training: true
  fairness_enabled: true

edge_server:
  mobility_prediction:
    prediction_horizon: 10.0
    update_interval: 1.0
  
  template_management:
    cache_size: 1000
    lookup_latency_threshold: 0.005  # 5ms

vehicle_layer:
  training_execution:
    epochs_range: [1, 2]  # Short-horizon training
    communication_bundle_size: 65536  # 64KB
```

## Examples

### Basic Vehicle Setup

```python
from fhdp.vehicle_layer import Vehicle
from fhdp.core.types import VehicleInfo

# Create vehicle
vehicle = Vehicle(
    vehicle_id="test_vehicle_001",
    initial_position=(100, 50),
    initial_velocity=25.0,  # m/s
    resources={
        'cpu': 0.8,
        'memory': 0.7,
        'battery': 0.9
    }
)

# Start vehicle with DSRC protocol
vehicle.start_vehicle(['dsrc'])

# Update position
vehicle.update_position((110, 52), 25.0, 0.1)
```

### Edge Server Setup

```python
from fhdp.edge_server import EdgeServer
from fhdp.core.types import VehicleInfo

# Create edge server
server = EdgeServer()

# Set coverage area (1000m x 1000m)
server.set_coverage_area(1000, 1000)

# Start server
server.start_server()

# Register vehicle
vehicle_info = VehicleInfo(
    vehicle_id="vehicle_001",
    position=(100, 50),
    velocity=25.0,
    resources={'cpu': 0.8}
)
server.register_vehicle(vehicle_info)
```

### System Integration

```python
from fhdp.core import FHDPSystem, SystemConfiguration

# Configure system
config = SystemConfiguration(
    max_vehicles_per_region=100,
    pipeline_formation_interval=5.0,
    enable_pipeline_training=True
)

# Create and start system
system = FHDPSystem(config)
system.start_system()

# Register vehicles
for i in range(20):
    vehicle_info = create_vehicle_info(i)
    system.register_vehicle(vehicle_info)

# Monitor system status
status = system.get_system_status()
print(f"Active vehicles: {status['registered_vehicles']}")
print(f"Active pipelines: {status['active_pipelines']}")
```

## Performance Requirements

FHDP meets the following key performance requirements:

- **Template Lookup**: <5ms latency guaranteed
- **Pipeline Formation**: <1.5s recomposition time  
- **Memory Usage**: Optimized for vehicular constraints
- **Communication**: Bundle-based optimization
- **Scalability**: Supports 100+ vehicles per region

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m fhdp.tests.test_fhdp_system

# Run specific test categories
python -m fhdp.tests.test_fhdp_system TestMobilityPredictor
python -m fhdp.tests.test_fhdp_system TestTemplateManager
python -m fhdp.tests.test_fhdp_system TestPerformanceRequirements
```

## Examples and Benchmarks

### Simple Simulation

```bash
cd examples
python simple_simulation.py
```

This runs a 60-second simulation with 8 vehicles demonstrating:
- Vehicle registration and resource monitoring
- Pipeline formation and training
- Asynchronous model aggregation
- Fair participation management

### Performance Benchmarking

```bash
cd examples
python benchmark_example.py [benchmark_type]
```

Available benchmarks:
- `template`: Template lookup performance
- `pipeline`: Pipeline formation speed  
- `mobility`: Mobility prediction efficiency
- `scalability`: System scalability tests
- `memory`: Memory usage analysis

## Key Features in Detail

### Hybrid Participation Model

FHDP supports both individual and pipeline training modes:

```python
# Individual training
vehicle.training_mode = TrainingMode.INDIVIDUAL

# Pipeline training
vehicle.training_mode = TrainingMode.PIPELINE
vehicle.initiate_pipeline_formation(target_vehicles)
```

### Fairness Mechanism

Frequency-based fairness ensures equitable participation:

```python
# Get fairness metrics
fairness = system.resource_classifier.get_fairness_metrics(vehicle_id)
print(f"Priority weight: {fairness.priority_weight}")
print(f"Contribution score: {fairness.contribution_score}")
```

### Lazy Error Propagation

Reduces communication overhead through error accumulation:

```python
# Error signals are accumulated and propagated together
training_executor.error_propagation.accumulate_error(
    vehicle_id, error_signal, target_vehicles
)
```

### Mobility Prediction

DTMC-based prediction for pipeline stability:

```python
# Predict vehicle position in 10 seconds
predictions = edge_server.mobility_predictor.predict_mobility(
    vehicle_id, horizon=10.0
)

# Predict pipeline stability
stability = edge_server.predict_pipeline_stability(
    vehicle_ids, duration=20.0
)
```

## Configuration Reference

### System Parameters

- `max_vehicles_per_region`: Maximum vehicles per server region (default: 50)
- `pipeline_formation_interval`: Pipeline formation check interval (default: 5.0s)
- `aggregation_interval`: Asynchronous aggregation interval (default: 2.0s)
- `fairness_enabled`: Enable fairness mechanisms (default: true)

### Performance Parameters

- `template_lookup_latency`: Maximum template lookup time (default: 5ms)
- `pipeline_recomposition_time`: Maximum pipeline formation time (default: 1.5s)
- `communication_bundle_size`: Bundle size for optimization (default: 64KB)

### Resource Thresholds

- `high_resource_cpu`: CPU threshold for high classification (default: 0.8)
- `medium_resource_cpu`: CPU threshold for medium classification (default: 0.5)
- `battery_critical`: Critical battery level (default: 0.15)

## API Reference

### Core Classes

- `FHDPSystem`: Main system orchestrator
- `EdgeServer`: Edge server implementation  
- `Vehicle`: Complete vehicle implementation
- `TemplateManager`: Pipeline template management
- `MobilityPredictor`: Vehicle movement prediction

### Key Methods

- `system.register_vehicle(vehicle_info)`: Register new vehicle
- `system.get_system_status()`: Get comprehensive status
- `server.find_pipeline_template(vehicle_ids)`: Find suitable template
- `vehicle.initiate_pipeline_formation(targets)`: Start pipeline formation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use FHDP in your research, please cite:

```
FHDP: Federated Highway-based Distributed Pipeline for Vehicular Federated Learning
[Your paper details here]
```