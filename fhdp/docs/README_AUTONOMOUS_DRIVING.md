# FHDP + EVO-1 Autonomous Driving Simulation

This repository integrates the **EVO-1 (Lightweight Vision-Language-Action Model)** with the **FHDP (Federated Learning for Heterogeneous Devices and Pipelines)** framework to create a comprehensive autonomous driving simulation with federated learning capabilities.

## 🚗 Overview

The simulation demonstrates how multiple autonomous vehicles can:
- Use the EVO-1 model for vision-based driving decisions
- Participate in federated learning to improve collective driving performance
- Process nuScenes dataset for realistic driving scenarios
- Maintain safety and efficiency metrics

## 🧠 Key Components

### 1. EVO-1 Integration
- **Vision-Language-Action Model**: Combines computer vision with natural language understanding
- **Multi-camera Processing**: Handles 6-camera input (nuScenes standard)
- **WebSocket Communication**: Client-server architecture for model inference
- **Fallback Policy**: Safe driving behavior when model server unavailable

### 2. FHDP Federated Learning
- **Heterogeneous Vehicle Management**: Different vehicle capabilities and resources
- **Pipeline Formation**: Dynamic collaboration networks
- **Model Aggregation**: Federated averaging of model updates
- **Resource-Aware Training**: Adaptive training based on vehicle resources

### 3. Autonomous Driving Features
- **Real-time Decision Making**: Continuous action prediction (steering, throttle, brake)
- **Safety Monitoring**: Track safety violations and comfort metrics
- **Performance Evaluation**: Distance, efficiency, and driving quality metrics
- **Scenario Diversity**: Highway, urban, and night driving scenarios

## 📋 Prerequisites

### System Requirements
- Python 3.10 (recommended)
- CUDA support (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended)
- 50GB+ storage space (for nuScenes dataset)

### Software Dependencies
- PyTorch 1.11+
- Transformers 4.20+
- OpenCV 4.5+
- WebSocket libraries
- nuScenes-devkit
- Flash Attention (for efficiency)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd fhdp

# Run the setup script
./setup_evo1.sh

# Or manual setup
conda create -n fhdp_evo1 python=3.10 -y
conda activate fhdp_evo1
pip install -r requirements_evo1.txt
```

### 2. EVO-1 Server Setup (Optional)

```bash
# Clone EVO-1 repository
git clone https://github.com/MINT-SJTU/Evo-1.git evo1_repo
cd evo1_repo

# Install EVO-1 dependencies
pip install -r requirements.txt

# Start the EVO-1 server
python scripts/Evo1_server.py
```

### 3. Dataset Preparation

```bash
# Download nuScenes mini dataset (4GB)
# Register at: https://www.nuscenes.org/download
# Extract to: ./data/nuscenes/
```

### 4. Run Simulation

```bash
# Run with the convenience script
./run_autonomous_driving.sh

# Or run directly
python examples/autonomous_driving_simulation.py
```

## 📁 Project Structure

```
fhdp/
├── examples/
│   └── autonomous_driving_simulation.py    # Main simulation script
├── requirements_evo1.txt                    # EVO-1 dependencies
├── setup_evo1.sh                           # Setup automation script
├── config_evo1.yaml                        # Configuration file
├── run_autonomous_driving.sh               # Run script
├── data/
│   └── nuscenes/                          # Dataset storage
├── models/
│   └── evo1/                              # Model storage
└── logs/
    └── evo1/                              # Training logs
```

## 🎮 Usage Examples

### Basic Autonomous Driving

```python
from examples.autonomous_driving_simulation import AutonomousDrivingVehicle, EVO1ModelClient

# Create EVO-1 client
evo1_client = EVO1ModelClient("ws://localhost:8765")
await evo1_client.connect()

# Create autonomous vehicle
vehicle = AutonomousDrivingVehicle(
    vehicle_id="test_vehicle",
    initial_position=(0, 0),
    evo1_client=evo1_client,
    data_loader=data_loader
)

# Start driving for 60 seconds
await vehicle.start_autonomous_driving(60)
```

### Federated Learning Training

```python
from examples.autonomous_driving_simulation import FederatedEVO1Trainer

# Create federated trainer
trainer = FederatedEVO1Trainer()

# Train on vehicle data
trainer.train_vehicle_model(vehicle_id, observations, actions)

# Aggregate updates from multiple vehicles
trainer.aggregate_model_updates(vehicle_ids)
```

## 📊 Performance Metrics

The simulation tracks comprehensive metrics:

### Driving Performance
- **Total Distance**: Distance traveled by each vehicle
- **Average Speed**: Mean velocity during simulation
- **Safety Violations**: Aggressive maneuvers count
- **Comfort Score**: Smoothness of driving (0-100)
- **Efficiency Score**: Fuel/time efficiency (0-100)

### Learning Performance
- **Training Time**: Time spent on model updates
- **Samples Trained**: Number of training examples
- **Model Loss**: Training convergence metrics
- **Aggregation Rounds**: Federated learning iterations

## ⚙️ Configuration

### EVO-1 Model Settings

```yaml
evo1:
  server_url: "ws://localhost:8765"
  model_name: "OpenGVLab/InternVL3-1B"
  image_size: 448
  horizon: 50
  dropout: 0.2
```

### Federated Learning Settings

```yaml
federated_learning:
  aggregation_interval: 45  # seconds
  training_interval: 30     # seconds
  num_training_epochs: 3
  batch_size: 8
```

### Simulation Parameters

```yaml
simulation:
  num_vehicles: 4
  simulation_duration: 120  # seconds
  scenario_types: ["highway", "urban", "night"]
```

## 🔧 Advanced Features

### Custom Driving Scenarios

```python
# Create custom scenarios
scenarios = [
    {
        "name": "emergency_stop",
        "description": "Sudden obstacle detection",
        "prompt": "紧急刹车并避让障碍物",
        "duration": 30
    }
]
```

### Multi-Modal Input Processing

```python
# Process multi-camera input with different resolutions
camera_configs = {
    "CAM_FRONT": {"resolution": (1920, 1080), "fov": 60},
    "CAM_FRONT_LEFT": {"resolution": (1920, 1080), "fov": 60},
    # ... other cameras
}
```

### Custom Aggregation Algorithms

```python
class CustomAggregator:
    def aggregate(self, model_updates):
        # Custom federated learning logic
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **EVO-1 Server Connection Failed**
   - Ensure server is running on port 8765
   - Check firewall settings
   - Fallback policy will be used automatically

2. **Flash Attention Installation Failed**
   - Reduce MAX_JOBS environment variable
   - Ensure CUDA toolkit compatibility
   - Consider CPU-only version

3. **Memory Issues**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable model checkpointing

4. **Dataset Loading Errors**
   - Verify nuScenes dataset path
   - Check file permissions
   - Ensure sufficient disk space

### Performance Optimization

```python
# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Use mixed precision training
with torch.cuda.amp.autocast():
    outputs = model(inputs)

# Optimize DataLoader with multiple workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## 📚 References

### Research Papers
- [EVO-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment](https://arxiv.org/abs/2403.xxxxx)
- [FHDP: Federated Learning for Heterogeneous Devices and Pipelines](https://arxiv.org/abs/xxxx.xxxxx)

### Documentation
- [EVO-1 GitHub Repository](https://github.com/MINT-SJTU/Evo-1)
- [nuScenes Dataset](https://www.nuscenes.org)
- [FHDP Documentation](./README.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_evo1.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Code formatting
black examples/
flake8 examples/
```

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

## 🙏 Acknowledgments

- **MINT-SJTU** for the EVO-1 model
- **nuScenes** team for the dataset
- **PyTorch** team for the ML framework
- **HuggingFace** for the transformers library

---

## 📞 Support

For questions and support:
- Open an [Issue](https://github.com/your-repo/issues)
- Contact: [your-email@example.com]
- Documentation: [Full Documentation Link]

Happy Autonomous Driving! 🚗🤖