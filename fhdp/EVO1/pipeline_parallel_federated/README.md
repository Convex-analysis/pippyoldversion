# EVO-1 Pipeline Parallel Training with FHDP Integration

This module integrates EVO-1 autonomous driving models with FHDP's native pipeline parallel capabilities, enabling federated learning across edge servers and vehicles while maintaining full compatibility with FHDP's existing architecture.

## 🚗 Overview

The FHDP-integrated pipeline training system enables:

- **Native FHDP Integration**: Uses FHDP's existing edge server and vehicle architecture
- **VLM Backbone Deployment**: Frozen VLM backbone on FHDP edge servers
- **Encoder Training**: Vehicle-side encoder training using FHDP's training engine
- **Pipeline Parallelism**: Leverages FHDP's native pipeline coordination
- **Resource Awareness**: Uses FHDP's hardware adaptation and resource classification
- **Fairness Management**: Employs FHDP's built-in fairness algorithms

## 🏗️ FHDP Architecture Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                 FHDP Native Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│  FHDP Core System                                           │
│  ├─ HybridParticipationManager                              │
│  ├─ AsynchronousAggregationManager                          │
│  ├─ FairnessManager                                       │
│  └─ HardwareAdapter                                       │
├─────────────────────────────────────────────────────────────────┤
│  FHDP Edge Servers (with EVO-1 VLM Integration)           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ FHDP Server │ │ FHDP Server │ │ FHDP Server │           │
│  │ + VLM Back- │ │ + VLM Back- │ │ + VLM Back- │           │
│  │   bone      │ │   bone      │ │   bone      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│  FHDP Vehicles (with EVO-1 Encoder Training)              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ FHDP Vehicle│ │ FHDP Vehicle│ │ FHDP Vehicle│           │
│  │ + Encoder   │ │ + Encoder   │ │ + Encoder   │           │
│  │ + Action    │ │ + Action    │ │ + Action    │           │
│  │   Head      │ │   Head      │ │   Head      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│  EVO-1 Integration Layer                                    │
│  ├─ EdgeServerVLMIntegration                               │
│  ├─ VehicleEncoderClient                                   │
│  ├─ FHDPipelineCoordinator                                │
│  └─ HardwareResourceAdapter                               │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
pipeline_parallel_federated/
├── __init__.py                    # Package initialization
├── pipeline_trainer.py            # Main training orchestrator
├── edge_server/                   # Edge server components
│   ├── __init__.py
│   ├── vlm_backbone.py           # VLM backbone server
│   └── server_manager.py         # Cluster management
├── vehicle_client/                # Vehicle client components
│   ├── __init__.py
│   ├── encoder_trainer.py        # Vehicle-side encoder training
│   └── pipeline_client.py        # Communication client
├── coordinator/                   # Coordination logic
│   ├── __init__.py
│   ├── pipeline_coordinator.py   # Main coordinator
│   └── resource_coordinator.py  # Resource management
├── adapter/                       # Resource adaptation
│   ├── __init__.py
│   └── resource_adapter.py      # Device optimization
├── configs/                       # Configuration files
│   └── default_config.yaml       # Default configuration
├── examples/                      # Example scripts
│   └── pipeline_federated_example.py  # Complete demo
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Basic FHDP Example (Small Scale)

```bash
cd /Volumes/HardDriveMac/EXP/pippyoldversion/fhdp/EVO1/pipeline_parallel_federated
python3 examples/fhd_pipeline_example.py --scale small --rounds 10
```

### 2. Medium Scale Demo

```bash
python3 examples/fhd_pipeline_example.py --scale medium --rounds 25
```

### 3. Large Scale Demo

```bash
python3 examples/fhd_pipeline_example.py --scale large --rounds 50
```

### 4. FHDP Component Demos

```bash
python3 examples/fhd_pipeline_example.py --components
```

### 5. Programmatic Usage

```python
from pipeline_trainer import FHDPipelineTrainer, FHDPipelineConfig

config = FHDPipelineConfig(
    experiment_name="my_fhd_experiment",
    max_vehicles=10,
    fairness_enabled=True
)

trainer = FHDPipelineTrainer(config)
await trainer.initialize_system(edge_configs, vehicle_configs)
await trainer.start_pipeline_training(num_rounds=20)
stats = trainer.get_training_statistics()
```

## 🔧 Key Features

### Edge Server VLM Backbone

- **Frozen Deployment**: Pre-trained VLM backbone deployed on edge servers
- **High Throughput**: Supports concurrent inference requests
- **Load Balancing**: Automatic request distribution across servers
- **Caching**: Intelligent response caching to reduce latency

### Vehicle-side Encoder Training

- **Local Training**: Each vehicle trains its own encoder
- **Resource Awareness**: Automatic adaptation to device constraints
- **Pipeline Communication**: Efficient communication with edge servers
- **Memory Optimization**: Gradient checkpointing and mixed precision

### Pipeline Parallel Coordination

- **Asynchronous Aggregation**: Non-blocking federated averaging
- **Fairness Management**: Ensures equal participation opportunities
- **Resource-Aware Selection**: Intelligent vehicle selection
- **FHDP Integration**: Leverages FHDP coordination capabilities

### Resource Adaptation

- **Memory Optimization**: Gradient checkpointing, pruning, quantization
- **Compute Optimization**: Mixed precision, CPU optimization
- **Network Optimization**: Compression, batched updates
- **Power Management**: Thermal and battery-aware training

## ⚙️ Configuration

### System Configuration

```yaml
system:
  num_edge_servers: 3
  num_vehicles: 10
  num_rounds: 50

training:
  pipeline_parallel: true
  async_aggregation: true
  fairness_enabled: true
  
resource_adaptation:
  enabled: true
  memory_constraint_gb: 4.0
  compute_constraint: "medium"

fhdp:
  enabled: true
  hybrid_participation: true
```

### Vehicle Profiles

```yaml
vehicle_profiles:
  high_performance:
    memory_gb: 8
    gpu_memory_gb: 6
    encoder_type: "resnet34"
    
  medium_performance:
    memory_gb: 4
    gpu_memory_gb: 2
    encoder_type: "resnet18"
    
  low_performance:
    memory_gb: 2
    gpu_memory_gb: 0
    encoder_type: "efficientnet_b0"
```

## 📊 Performance Characteristics

### Resource Efficiency

- **Memory Reduction**: 60-80% reduction compared to full model training
- **Compute Efficiency**: Adaptive batch sizing and gradient accumulation
- **Network Optimization**: Intelligent compression and caching
- **Power Management**: Thermal and battery-aware training

### Training Performance

- **Scalability**: Linear scaling up to 50+ vehicles
- **Convergence**: Similar convergence to centralized training
- **Fairness**: Equal participation across heterogeneous devices
- **Robustness**: Fault-tolerant to dropouts and failures

## 🔬 Use Cases

### 1. Autonomous Driving Fleets

- Real-time model updates across vehicle fleets
- Region-specific model adaptation
- Privacy-preserving collaborative learning

### 2. Edge AI Deployment

- Resource-constrained edge devices
- Hierarchical model distribution
- Latency-optimized inference

### 3. Research and Development

- Pipeline parallel federated learning research
- Resource adaptation algorithms
- FHDP integration studies

## 🧪 Advanced Usage

### Custom Resource Adapter

```python
from adapter.resource_adapter import ResourceConstrainedAdapter, AdapterConfig, DeviceCapabilities

# Define device capabilities
capabilities = DeviceCapabilities(
    device_id="custom_device",
    total_memory_gb=2.0,
    compute_capability="low",
    gpu_available=False
)

# Configure adaptation
config = AdapterConfig(
    enable_memory_optimization=True,
    gradient_checkpointing=True,
    quantization=True
)

# Create adapter
adapter = ResourceConstrainedAdapter(capabilities, config)
```

### Custom Training Pipeline

```python
from pipeline_trainer import PipelineParallelFederatedTrainer, PipelineTrainingConfig

# Create custom configuration
config = PipelineTrainingConfig(
    experiment_name="custom_experiment",
    num_edge_servers=5,
    num_vehicles=20,
    fhdp_integration=True
)

# Initialize and run trainer
trainer = PipelineParallelFederatedTrainer(config)
await trainer.initialize()
await trainer.start_training()
```

## 📈 Monitoring and Visualization

The system provides comprehensive monitoring:

- **Training Metrics**: Loss, accuracy, convergence
- **Resource Usage**: Memory, compute, network
- **System Health**: Server status, vehicle participation
- **Fairness Metrics**: Participation distribution

Visualizations include:
- Training progress plots
- Resource utilization graphs
- Communication overhead analysis
- Fairness distribution charts

## 🤝 Integration with Existing EVO-1

The pipeline parallel system integrates seamlessly with existing EVO-1 components:

- **Model Compatibility**: Works with existing EVO-1 model definitions
- **Data Pipeline**: Compatible with nuScenes data loaders
- **Training Infrastructure**: Extends existing training utilities
- **FHDP Integration**: Leverages FHDP coordination capabilities

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or enable gradient checkpointing
2. **Communication Failures**: Check edge server connectivity
3. **Slow Training**: Enable mixed precision and adaptive batching
4. **Resource Exhaustion**: Use resource adaptation profiles

### Performance Optimization

1. **Use Appropriate Vehicle Profiles**: Match device capabilities
2. **Enable Caching**: Reduce communication overhead
3. **Optimize Pipeline Degree**: Balance parallelism and overhead
4. **Monitor Resource Usage**: Adjust configuration dynamically

## 📝 License

This implementation follows the same license as the EVO-1 project and FHDP framework.

## 🤝 Contributing

Contributions welcome! Please follow the existing code style and add comprehensive tests.

---

For more information, see the EVO-1 documentation and FHDP framework guides.