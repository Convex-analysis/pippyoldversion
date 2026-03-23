# FHDP: Federated Learning for Heterogeneous Devices and Pipelines

🚀 **FHDP** is a comprehensive framework for federated learning across heterogeneous devices, now with integrated **EVO-1** autonomous driving capabilities optimized for edge devices like NVIDIA Jetson.

## 🎯 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd fhdp

# Quick installation (handles all dependencies)
bash fhdp/scripts/fix_flash_attn.sh

# Run basic demo
python examples/simple_simulation.py

# Run EVO-1 autonomous driving
python examples/autonomous_driving_simulation.py

# Run Stage 1 on Jetson
python examples/evo1_stage1_federated.py
```

## 📚 Documentation Hub

📁 **All documentation is organized in the [docs/](docs/) folder:**

### 🚀 For New Users
- [**Installation Guide**](docs/INSTALLATION.md) - Complete setup instructions
- [**Quick Reference**](docs/QUICK_REFERENCE.md) - One-page cheat sheet
- [**Docs Hub**](docs/README.md) - Complete documentation index

### 🧠 For EVO-1 Users  
- [**Autonomous Driving**](docs/README_AUTONOMOUS_DRIVING.md) - Full EVO-1 integration
- [**Stage 1 on Jetson**](docs/README_EVO1_STAGE1.md) - Edge device optimization
- [**Installation Fix**](docs/INSTALLATION_FIX.md) - Flash-attention troubleshooting

### 🔧 For Developers
- [**Implementation Summary**](docs/IMPLEMENTATION_SUMMARY.md) - Technical details
- [**Heterogeneous Adaptation**](docs/HETEROGENEOUS_ADAPTATION.md) - Device optimization
- [**Testbed Architecture**](docs/TESTBED_ARCHITECTURE.md) - System design

## 🎮 Key Features

### 🌐 Federated Learning
- **Multi-Device Training**: Collaborative learning without data sharing
- **Heterogeneous Support**: Different device capabilities and resources
- **Dynamic Pipelines**: Adaptive collaboration networks
- **Resource Management**: Memory, compute, and power optimization

### 🧠 EVO-1 Integration
- **Stage 1**: Action Expert Alignment (VLM frozen, ~10K parameters)
- **Stage 2**: Full model fine-tuning (1B+ parameters)  
- **Autonomous Driving**: Vision-language-action model for vehicles
- **Real-time Inference**: Low-latency decision making

### 📱 Edge Device Optimization
- **Jetson Support**: Optimized for Orin/Nano devices
- **Memory Efficiency**: Staged training with minimal resources
- **Thermal Management**: Automatic performance throttling
- **Power Optimization**: Device-specific power modes

## 🛠️ Project Structure

```
fhdp/
├── 📁 docs/                     # All documentation (📚)
│   ├── README.md                  # Documentation hub
│   ├── QUICK_REFERENCE.md        # One-page guide
│   ├── INSTALLATION.md          # Setup instructions
│   ├── IMPLEMENTATION_SUMMARY.md # Technical details
│   ├── README_EVO1_STAGE1.md    # Jetson Stage 1
│   └── README_AUTONOMOUS_DRIVING.md # EVO-1 driving
├── 🐍 examples/                  # Simulation scripts
│   ├── simple_simulation.py       # Basic FHDP demo
│   ├── enhanced_simulation.py    # Real training
│   ├── autonomous_driving_simulation.py # Full EVO-1
│   └── evo1_stage1_federated.py # Stage 1 on Jetson
├── 🔧 fhdp/                     # Core library
├── 📦 requirements*.txt          # Dependencies
├── 🚀 deploy*.sh                # Deployment scripts
└── 🧪 test*.py                  # Test scripts
```

## 🚀 Quick Navigation

| Goal | Document | Command |
|------|----------|----------|
| **Get Started** | [Installation Guide](docs/INSTALLATION.md) | `bash fhdp/scripts/fix_flash_attn.sh` |
| **EVO-1 Driving** | [Autonomous Driving](docs/README_AUTONOMOUS_DRIVING.md) | `python examples/autonomous_driving_simulation.py` |
| **Jetson Deployment** | [Stage 1 Guide](docs/README_EVO1_STAGE1.md) | `python examples/evo1_stage1_federated.py` |
| **Basic Demo** | [Implementation](docs/IMPLEMENTATION_SUMMARY.md) | `python examples/simple_simulation.py` |
| **Troubleshooting** | [Installation Fix](docs/INSTALLATION_FIX.md) | `python test_install_fix.py` |

## 🎯 Choose Your Use Case

### 🧪 Learning Federated Learning
```bash
# Start with basic simulation
python examples/simple_simulation.py

# Progress to real training
python examples/enhanced_simulation.py
```

### 🚗 Autonomous Driving
```bash
# Full EVO-1 system
python examples/autonomous_driving_simulation.py

# Stage 1 on edge devices
python examples/evo1_stage1_federated.py
```

### 📱 Jetson Deployment
```bash
# Automated deployment
bash fhdp/scripts/Jetson/deploy_jetson_stage1.sh

# For Orin Nano specifically
bash fhdp/scripts/Jetson/deploy_jetson_orin_nano.sh

# Manual setup
pip install -r fhdp/requirements/jetson.txt
```

## 🔧 System Requirements

### Minimum
- **Python**: 3.8-3.10
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB

### For EVO-1 Full System
- **RAM**: 16GB+  
- **GPU**: CUDA with 8GB+ VRAM
- **Storage**: 50GB+

### For Jetson Devices
- **Device**: Orin/Nano
- **RAM**: 6GB usable (8GB+ total)
- **Power**: 15W (Orin) / 8W (Nano)

## 📊 Key Performance

| Feature | Desktop | Jetson Orin | Jetson Nano |
|----------|----------|---------------|-------------|
| Stage 1 Training | ~45 FPS | ~15 FPS | ~8 FPS |
| Memory Usage | 8GB | 6GB | 3GB |
| Power Consumption | 250W | 15W | 8W |
| Model Size (Stage 1) | 40MB | 40MB | 40MB |

## 🎉 Getting Help

### 📚 Documentation
- **[Docs Hub](docs/README.md)** - Complete documentation index
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - One-page guide
- **[Installation Guide](docs/INSTALLATION.md)** - Step-by-step setup

### 🔧 Common Issues
- **Flash-attention**: [Installation Fix](docs/INSTALLATION_FIX.md)
- **CUDA errors**: Check PyTorch compatibility
- **Memory issues**: Reduce batch sizes

### 🧪 Testing
```bash
# Installation verification
python test_install_fix.py

# Component testing
python test_jetson_stage1.py

# Full system test
python test_autonomous_driving.py
```

## 🤝 Contributing

1. **Code**: Follow existing patterns in `examples/`
2. **Tests**: Add to `test*.py` files
3. **Docs**: Update relevant `.md` files in `docs/`
4. **Structure**: Keep documentation in `docs/` folder

## 📄 License

[License Information]

## 🙏 Acknowledgments

- **EVO-1 Team**: Vision-language-action model
- **NVIDIA**: Jetson platform and CUDA
- **Federated Learning Community**: Research foundations
- **PyTorch Team**: ML framework

---

🚀 **Ready to start?** Head to the [docs hub](docs/README.md) for complete documentation!

💡 **Quick tip**: Use `./fix_flash_attn.sh` for any installation issues - it handles everything automatically!