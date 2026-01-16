# EVO-1 Usage Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements_final.txt

# Or install manually
pip install torch torchvision transformers accelerate opencv-python matplotlib
```

### 2. Test Model

```bash
cd /home/xta/fhdp/EVO1

# Test basic functionality
PYTHONPATH=. python -c "
from model.evo1_driving import EVO1Driving, ModelConfig
config = ModelConfig()
model = EVO1Driving(config)
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

### 3. Basic Usage

```python
import torch
from model.evo1_driving import EVO1Driving, ModelConfig

# Create configuration
config = ModelConfig()
config.vision_encoder = "OpenGVLab/InternVL3-1B"
config.sequence_length = 32

# Create model
model = EVO1Driving(config)

# Prepare inputs
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)
text_inputs = torch.randint(0, 1000, (batch_size, 32))

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(images, text_inputs)
    
print(f"Actions: {outputs.actions.shape}")
print(f"Waypoints: {outputs.waypoints.shape}")
```

## 📊 Model Features

### Core Components
- **InternVL3 Vision-Language Encoder**: Processes visual and textual inputs
- **State Encoder**: Encodes vehicle dynamics and sensor data
- **Action Head**: Predicts control actions and waypoints
- **Flow Matching**: Advanced action prediction technique

### Outputs
- `actions`: Control actions [steering, throttle, brake]
- `waypoints`: Future trajectory waypoints
- `uncertainty`: Prediction uncertainty estimates

## 🔧 Configuration Options

```python
@dataclass
class ModelConfig:
    vision_encoder: str = "OpenGVLab/InternVL3-1B"
    language_model: str = "Qwen/Qwen2.5-0.5B"
    sequence_length: int = 32
    hidden_dim: int = 4096
    vision_model_name: str = "OpenGVLab/InternVL3-1B"
    image_size: int = 224
    max_waypoints: int = 20
    max_speed: float = 30.0  # m/s
    max_steering: float = 0.6  # radians
    control_frequency: float = 10.0  # Hz
    trajectory_horizon: float = 3.0  # seconds
```

## 🤝 Federated Learning

EVO-1 supports federated learning for distributed training:

```python
from model.evo1_driving import FederatedEVO1Driving, ModelConfig

config = ModelConfig()
fed_model = FederatedEVO1Driving(config, num_clients=4)

# Federated aggregation
fed_model.aggregate_client_updates(client_weights, client_data)
```

## 📈 Training

### Federated Training with nuScence Dataset

#### 1. Dataset Preparation

1. Download the nuScence mini dataset from the official website:
   - Go to [https://www.nuscenes.org/download](https://www.nuscenes.org/download)
   - Download the "v1.0-mini" dataset (approximately 3.3GB)
   - Extract the dataset to `/home/xta/fhdp/EVO1/data/nuscenes`

2. Verify the dataset structure:
```bash
ls -la /home/xta/fhdp/EVO1/data/nuscenes
# Should contain: maps/ samples/ scenes/ sweeps/ v1.0-mini/
```

#### 2. Configuration Setup

Edit the configuration file at `scripts/configs/default_config.yaml`:

```yaml
data:
  version: "v1.0-mini"
  data_root: "/home/xta/fhdp/EVO1/data/nuscenes"

training:
  federated_learning: true
  num_clients: 1
  local_epochs: 1
  mixed_precision: false
```

#### 3. Run Federated Training

```bash
cd /home/xta/fhdp/EVO1

python scripts/train_federated.py \
  --config scripts/configs/test_config_no_amp.yaml \
  --data_root /home/xta/fhdp/EVO1/data/nuscenes \
  --experiment_name "my_evo1_experiment" \
  --output_dir "./outputs" \
  --num_rounds 1 \
  --num_clients 1 \
  --local_epochs 1 \
  --log_level INFO
```

#### 4. Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Configuration file path | `scripts/configs/default_config.yaml` |
| `--data_root` | Path to nuScence dataset | `/home/xta/fhdp/EVO1/data/nuscenes` |
| `--experiment_name` | Experiment name | `"my_evo1_experiment"` |
| `--output_dir` | Output directory | `./outputs` |
| `--num_rounds` | Number of federated rounds | 100 |
| `--num_clients` | Number of federated clients | 10 |
| `--local_epochs` | Epochs per client | 2 |
| `--log_level` | Logging level | INFO |

#### 5. Training Output

After training completes, you'll find the following files in the output directory:

- `final_metrics.json`: Final evaluation metrics
- `experiment_summary.json`: Experiment configuration and run info
- `checkpoints/`: Model checkpoints (if enabled)

### Basic Training Loop (Advanced)

```python
import torch.optim as optim

# Setup
model = EVO1Driving(config)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images, text, targets = batch
        
        # Forward pass
        outputs = model(images, text)
        loss = criterion(outputs.actions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## ⚠️ Important Notes

1. **Dependencies**: Uses fallback implementation when original EVO-1 components are not available
2. **Memory**: Model requires ~2GB GPU memory for inference
3. **Requirements**: PyTorch 2.5+, Transformers 4.39+
4. **Data**: Supports multi-camera images and text prompts

## 🔍 Troubleshooting

### Common Issues

1. **Import Error**: Set PYTHONPATH environment variable
   ```bash
   export PYTHONPATH=/path/to/EVO1:$PYTHONPATH
   ```

2. **Memory Issues**: Reduce batch size or use gradient checkpointing

3. **CUDA Issues**: Ensure PyTorch CUDA version matches your system

### Model Parameters

The fallback EVO-1 implementation has:
- **Total Parameters**: ~21.6M
- **Trainable Parameters**: ~21.6M
- **Memory Usage**: ~2GB (inference)

## 📚 Advanced Usage

### Custom Vision Encoder

```python
# Replace vision encoder
model.vl_embedder = CustomVisionEncoder(config)
```

### Multi-Modal Input

```python
# Multi-modal forward pass
outputs = model(
    images=images,
    text=text_inputs,
    sensor_data=sensor_features,
    vehicle_state=vehicle_dynamics
)
```

## 🚀 Production Deployment

For production deployment:

1. **Quantization**: Use torch.quantization for smaller models
2. **Optimization**: Enable TorchScript for faster inference
3. **Monitoring**: Add performance and safety monitoring

```python
# TorchScript export
traced_model = torch.jit.trace(model, example_inputs)
traced_model.save("evo1_optimized.pt")
```

## 📞 Support

For issues and questions:
1. Check the error messages in console
2. Verify all dependencies are installed
3. Test with smaller batch sizes first
4. Check GPU memory availability

The EVO-1 model is now ready for autonomous driving research and development!