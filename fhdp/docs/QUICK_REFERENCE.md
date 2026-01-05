# FHDP Quick Reference

One-page guide for FHDP and EVO-1 integration.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo>
cd fhdp

# Install (choose one)
./fix_flash_attn.sh                    # Quick fix for any issues
./deploy_jetson_stage1.sh              # Jetson deployment
python install_evo1_deps.py             # Staged installation

# Run examples
python examples/simple_simulation.py       # Basic FHDP
python examples/enhanced_simulation.py    # With real training
python examples/autonomous_driving_simulation.py  # EVO-1 autonomous driving
python examples/evo1_stage1_federated.py  # Stage 1 on Jetson
```

## 📋 Project Structure

```
fhdp/
├── examples/                    # Simulation scripts
│   ├── simple_simulation.py              # Basic FHDP demo
│   ├── enhanced_simulation.py           # Real neural network training
│   ├── autonomous_driving_simulation.py # EVO-1 autonomous driving
│   └── evo1_stage1_federated.py     # Stage 1 on Jetson
├── docs/                        # All documentation (this folder!)
├── requirements*.txt              # Dependency files
├── deploy*.sh                   # Deployment scripts
├── test*.py                     # Test scripts
└── fhdp/                       # Core FHDP library
```

## 🧠 Key Features

### FHDP Core
- **Federated Learning**: Multi-vehicle collaboration without data sharing
- **Heterogeneous Support**: Different device capabilities
- **Pipeline Formation**: Dynamic collaboration networks
- **Resource Management**: Adaptive training based on resources

### EVO-1 Integration
- **Stage 1**: Action Expert Alignment (VLM frozen, ~10K parameters)
- **Stage 2**: Full model fine-tuning (all parameters trainable)
- **Autonomous Driving**: Vision-language-action model for vehicles
- **Jetson Optimization**: Edge device deployment

## 📱 Device Support

| Device | Recommended Setup | Memory | Performance |
|--------|------------------|---------|-------------|
| Desktop/Server | Full EVO-1 | 16GB+ | Maximum |
| Jetson Orin | Stage 1 only | 6GB | High |
| Jetson Nano | Stage 1 only | 2-3GB | Medium |
| Laptop | Basic FHDP | 8GB | Medium |

## 🔧 Common Commands

### Testing
```bash
python test_install_fix.py              # Installation check
python test_jetson_stage1.py           # Jetson validation
python test_autonomous_driving.py        # EVO-1 test
```

### Running Simulations
```bash
# Basic federated learning
python examples/simple_simulation.py

# With real neural networks
python examples/enhanced_simulation.py

# Complete EVO-1 autonomous driving
python examples/autonomous_driving_simulation.py

# Stage 1 optimized for Jetson
python examples/evo1_stage1_federated.py
```

### Monitoring
```bash
python jetson_monitor.py               # Jetson performance
python jetson_memory_optimizer.py       # Memory management
```

## 🎯 Use Cases

### Research
- **Federated Learning**: Multi-device training
- **Heterogeneous Systems**: Different device types
- **Autonomous Driving**: EVO-1 model deployment

### Production
- **Edge Deployment**: Jetson devices
- **Cloud Integration**: Federated server
- **Real-time Control**: Low-latency inference

### Education
- **Learning**: Federated learning concepts
- **Hands-on**: Practical implementations
- **Research**: Algorithm development

## 🔍 Key Files

| File | Purpose | When to Use |
|-------|---------|-------------|
| `examples/simple_simulation.py` | Basic FHDP demo | Learning federated learning |
| `examples/evo1_stage1_federated.py` | Jetson Stage 1 | Edge device deployment |
| `examples/autonomous_driving_simulation.py` | Full EVO-1 | Complete autonomous driving |
| `requirements_jetson_stage1.txt` | Jetson dependencies | Jetson setup |
| `deploy_jetson_stage1.sh` | Jetson deployment | Automated setup |

## ⚡ Performance Tips

### Memory Optimization
```python
# Reduce batch size
batch_size = 2  # For Jetson
batch_size = 16  # For desktop

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache frequently
if step % 10 == 0:
    torch.cuda.empty_cache()
```

### Jetson Optimization
```bash
# Set maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature
watch -n 1 cat /sys/class/thermal/thermal_zone0/temp
```

### Federated Learning
```python
# Reduce communication frequency
aggregation_interval = 60  # seconds

# Use compression
compression_enabled = True
quantization_bits = 8
```

## 📊 Quick Benchmarks

### Training Performance
| System | Batch Size | FPS | Power |
|---------|-------------|------|-------|
| Desktop RTX 3080 | 32 | 45 | 250W |
| Jetson Orin | 4 | 15 | 15W |
| Jetson Nano | 2 | 8 | 8W |

### Model Sizes
| Model | Parameters | Memory | Training Time |
|-------|------------|---------|---------------|
| Stage 1 Action Head | 10K | 40MB | ~5s/epoch |
| Full EVO-1 | 1B+ | 8GB+ | ~30s/epoch |

## 🚨 Troubleshooting Quick Fixes

### Flash-Attention Issues
```bash
export MAX_JOBS=2
pip install flash-attn --no-build-isolation || echo "Skip (optional)"
```

### Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=2

# Clear swap
sudo swapoff -a
sudo swapon /swapfile
```

### CUDA Issues
```bash
# Check installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Documentation Deep-Dive

### For Complete Understanding
- [Main Docs Hub](./README.md) - All documentation index
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md) - Technical details
- [EVO-1 Stage 1](../README_EVO1_STAGE1.md) - Jetson optimization

### For Quick Answers
- [Installation Guide](./INSTALLATION.md) - Step-by-step setup
- [Installation Fix](../INSTALLATION_FIX.md) - Troubleshooting
- [Testbed Architecture](./TESTBED_ARCHITECTURE.md) - System design

---

💡 **Tip**: Start with simple simulation, then progress to advanced features. All scripts are self-contained and include helpful error messages.

🎯 **Next Steps**: Choose your use case and follow the corresponding documentation in the [Docs Hub](./README.md).