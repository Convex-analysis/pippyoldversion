# EVO-1 Autonomous Driving Integration for FHDP Framework

This directory contains the complete implementation of the EVO-1 vision-language-action model integrated with the FHDP federated learning framework, specifically adapted for autonomous driving tasks using the nuScenes dataset.

## 🚗 Overview

The EVO-1 FHDP integration provides:

- **Federated Learning**: Distributed training across multiple vehicles/clients
- **Vision-Language-Action Model**: Multi-modal understanding for driving decisions
- **Real-time Inference**: Optimized pipeline for autonomous driving
- **Comprehensive Evaluation**: Driving-specific metrics and safety assessment
- **FHDP Integration**: Seamless integration with the existing FHDP architecture

## 📁 Project Structure

```
EVO1/
├── __init__.py                    # Package initialization
├── utils/
│   └── config.py                  # Configuration system
├── model/
│   └── evo1_driving.py           # EVO-1 model adapted for driving
├── data/
│   ├── nuscenes/                  # nuScenes dataset
│   ├── augmentation.py            # Driving-specific data augmentation
├── training/
│   ├── federated_trainer.py      # Federated training pipeline
│   └── utils.py                   # Training utilities
├── evaluation/
│   └── driving_metrics.py        # Autonomous driving evaluation metrics
├── inference/
│   └── driving_inference.py       # Real-time inference pipeline
├── scripts/
│   ├── train_federated.py        # Main training script
│   ├── inference.py              # Inference script
│   ├── evaluate.py               # Comprehensive evaluation script
│   └── configs/
│       ├── default_config.yaml   # Default configuration
│       ├── test_config.yaml      # Test configuration
│       └── test_config_no_amp.yaml  # Test config without AMP
├── outputs/                       # Training outputs and checkpoints
├── evaluation_outputs/            # Evaluation results
├── inference_outputs/             # Inference results
├── debug_outputs/                 # Debug outputs
├── test_outputs/                  # Test outputs
├── USAGE.md                       # Additional usage documentation
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.5.1+
- CUDA-compatible GPU (recommended)
- nuScenes dataset
- FHDP framework

### Setup

1. **Navigate to the project**:
```bash
cd /home/xta/fhdp/EVO1
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **nuScenes dataset**:
- The dataset is already available in `./data/nuscenes/`
- Contains v1.0-mini version for testing

4. **Environment Setup**:
```bash
# Set data root environment variable (optional)
export NUSCENES_ROOT="./data/nuscenes"
```

## 🚀 Quick Start

### 1. Training

Start federated training with default configuration:

```bash
python scripts/train_federated.py \
    --config scripts/configs/default_config.yaml \
    --experiment_name "federated_training" \
    --output_dir "./outputs"
```

**Key Configuration Parameters:**
- `--config`: Path to configuration file
- `--experiment_name`: Name for the experiment
- `--output_dir`: Directory to save outputs
- `--log_level`: Logging level (INFO, DEBUG, etc.)

### 2. Inference

Run inference on a trained model:

```bash
python scripts/inference.py \
    --model_path "./outputs/checkpoints/global_model_round_0099.pt" \
    --config_path "./outputs/config.yaml"
```

### 3. Evaluation

Run comprehensive model evaluation:

```bash
python scripts/evaluate.py \
    --model_path "./outputs/checkpoints/global_model_round_0099.pt" \
    --config_path "./outputs/config.yaml" \
    --split "val" \
    --batch_size 4 \
    --visualize
```

**Evaluation Options:**
- `--split`: Dataset split (val, test)
- `--batch_size`: Batch size for evaluation
- `--eval_steps`: Number of batches to evaluate
- `--visualize`: Generate performance visualizations
- `--output_dir`: Directory to save evaluation results

### 4. WebSocket Streaming

Real-time inference streaming is available through the inference pipeline. The system processes images, vehicle state, and driving instructions to generate control commands.

**Input Format:**
```json
{
    "images": ["image1.jpg", "image2.jpg", "image3.jpg"],
    "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "instruction": "Continue driving safely"
}
```

**Output Format:**
```json
{
    "controls": [steering, throttle, brake],
    "waypoints": [[x1, y1, z1], [x2, y2, z2], ...],
    "confidence": 0.95,
    "trajectory": [[x1, y1, z1], [x2, y2, z2], ...]
}
```

## ⚙️ Configuration

### Key Configuration Parameters

The project uses YAML configuration files with the following structure:

#### Model Configuration
```yaml
model:
  vision_model_name: "OpenGVLab/InternVL3-1B"
  vision_layers: 14
  image_size: 448
  num_views: 3
  action_dim: 8  # [steering, throttle, brake, + 5 auxiliary]
  action_hidden_dim: 512
  action_num_layers: 6
  flow_matching_steps: 50
  max_waypoints: 20
  trajectory_horizon: 3.0
  control_frequency: 10.0
  max_speed: 30.0
  max_steering: 0.6
```

#### Training Configuration
```yaml
training:
  federated_learning: true
  num_clients: 1
  local_epochs: 2
  aggregation_rounds: 10
  client_fraction: 0.3
  learning_rate: 0.0001
  weight_decay: 0.0001
  batch_size: 2
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 1000
  min_lr: 1e-06
  mixed_precision: false
  gradient_checkpointing: false
  use_flash_attention: false
  max_memory_gb: 8.0
```

#### Data Configuration
```yaml
data:
  data_root: "./data/nuscenes"
  version: "v1.0-mini"
  split: "train"
  image_size: [448, 448]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  sequence_length: 10
  sequence_stride: 5
  max_speed: 30.0
  min_speed: 0.0
  max_steering: 0.6
  max_acceleration: 3.0
```

### Environment-Specific Configurations

#### Default Configuration (`default_config.yaml`)
- Standard settings for general use
- Balanced performance and memory usage
- Suitable for most training scenarios

#### Test Configuration (`test_config.yaml`)
- Optimized for quick testing
- Smaller batch sizes and fewer rounds
- Enables mixed precision training

#### Test Configuration (No AMP) (`test_config_no_amp.yaml`)
- Test configuration without automatic mixed precision
- Useful for debugging FP16-related issues

## 📊 Evaluation

### Autonomous Driving Metrics

The `evaluate.py` script provides comprehensive evaluation metrics:

1. **Trajectory Metrics**:
   - Average Displacement Error (ADE)
   - Final Displacement Error (FDE)
   - Miss Rate
   - Trajectory Length Error
   - Heading Error

2. **Control Metrics**:
   - Steering MAE/RMSE
   - Throttle MAE/RMSE  
   - Brake MAE/RMSE
   - Control Smoothness

3. **Safety Metrics**:
   - Collision Rate
   - Off-road Rate
   - Traffic Violation Rate
   - Comfort Score

4. **Efficiency Metrics**:
   - Fuel Efficiency Score
   - Path Efficiency
   - Average Speed Error

### Evaluation Output

The evaluation script generates:
- `evaluation_results.json`: Detailed evaluation metrics
- `visualization.png`: Performance visualizations (if `--visualize`)
- `evaluation.log`: Evaluation execution logs

### Example Evaluation Results

```json
{
  "experiment_name": "evaluation_run",
  "model_path": "./outputs/checkpoints/global_model_round_0099.pt",
  "dataset_split": "val",
  "evaluation_time": 8.62,
  "avg_inference_time": 0.478,
  "batches_evaluated": 5,
  "trajectory_metrics": [
    {
      "ade": "7.3387",
      "fde": "9.4979",
      "miss_rate": 100.0,
      "trajectory_length_error": "8.5010",
      "heading_error": "0.3931"
    }
  ]
}
```

## 🔧 Advanced Features

### 1. Federated Learning

**Client Selection**: Randomly selects subset of clients per round
**Aggregation**: Federated Averaging (FedAvg)
**Early Stopping**: Automatically stops training when loss plateaus
**Metrics Tracking**: Comprehensive round-by-round metrics collection

### 2. Data Augmentation

**Weather Effects**: Rain, fog, snow, night simulation
**Sensor Noise**: Gaussian noise, motion blur, compression artifacts
**Geometric**: Random crops, flips, perspective changes

### 3. Real-time Optimization

**Mixed Precision**: Optional FP16 training for speedup
**Gradient Checkpointing**: Memory optimization for large models
**Device Management**: Automatic CPU/GPU detection and allocation

## 📈 Performance

### Current Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| ADE (Average Displacement Error) | ~7.3m | Average trajectory error |
| FDE (Final Displacement Error) | ~9.5m | Final position error |
| Inference Time | ~478ms/batch | Average inference time |
| Training Rounds | 10 | Completed federated rounds |
| Model Size | ~8GB | Memory usage |

### Scalability

- **Single Vehicle**: Real-time inference capability
- **Federated Training**: Supports multiple clients
- **Evaluation**: Comprehensive metrics for model assessment

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Disable mixed precision training
   - Use smaller model configuration

2. **Training Stopping Early**:
   - Adjust early stopping threshold in `federated_trainer.py`
   - Increase `aggregation_rounds` in configuration

3. **Inference Errors**:
   - Ensure model checkpoint exists at specified path
   - Check CUDA device availability
   - Verify input tensor device placement

4. **Evaluation Errors**:
   - Ensure `evaluate.py` is run with correct parameters
   - Check that model checkpoint is valid
   - Verify dataset path and split

### Debug Mode

Enable debug logging:
```bash
python scripts/train_federated.py --log_level DEBUG
python scripts/evaluate.py --log_level DEBUG
```

## 🤝 Integration with FHDP

The EVO-1 integration works with the FHDP framework:

1. **Vehicle Layer**: EVO-1 provides driving policy and control commands
2. **Training**: Federated learning across multiple clients
3. **Communication**: Uses standard protocols for model aggregation
4. **Evaluation**: Comprehensive metrics for model performance assessment

## 📄 License

This project extends the original EVO-1 and FHDP frameworks under their respective licenses.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration documentation
3. Examine the log files in the output directories
4. Run evaluation to assess model performance

## 🗺️ Roadmap

- [ ] Improve trajectory prediction accuracy
- [ ] Enhance real-time inference performance
- [ ] Add support for additional datasets
- [ ] Implement advanced federated learning techniques
- [ ] Develop visualization tools for model interpretability

---

**Note**: This implementation uses the nuScenes v1.0-mini dataset for training and evaluation. The system includes fallback implementations for standalone usage without the original EVO-1 codebase.

**Latest Updates**:
- Added comprehensive `evaluate.py` script for model evaluation
- Fixed training loop early termination issues
- Improved inference pipeline with proper device management
- Updated logging configuration for better monitoring
- Added detailed evaluation metrics and visualization support