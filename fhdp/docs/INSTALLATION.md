# FHDP Installation Guide

Complete installation guide for FHDP and EVO-1 integration.

## 🎯 Quick Installation

### Option 1: Basic Setup
```bash
# Clone repository
git clone <repository-url>
cd fhdp

# Install core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm psutil
```

### Option 2: EVO-1 Autonomous Driving
```bash
# Install EVO-1 dependencies
pip install -r requirements_evo1.txt

# If flash-attn fails, use fix script
./fix_flash_attn.sh
```

### Option 3: Jetson Stage 1
```bash
# Jetson deployment
./deploy_jetson_stage1.sh

# Or manual
pip install -r requirements_jetson_stage1.txt
```

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.8-3.10
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free
- **OS**: Linux/macOS/Windows

### For EVO-1 Full System
- **RAM**: 16GB+ (for VLM)
- **GPU**: CUDA-capable with 8GB+ VRAM
- **Storage**: 50GB+ (for models and datasets)

### For Jetson Deployment
- **Device**: Jetson Orin/Nano
- **RAM**: 6-8GB usable
- **Storage**: 32GB+ SD/eMMC
- **Power**: 15W (Orin) / 8W (Nano)

## 📦 Dependencies Overview

### Core Dependencies
```bash
# PyTorch ecosystem
torch>=1.11.0
torchvision>=0.12.0
torchaudio>=0.11.0

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# System utilities
psutil>=5.9.0
click>=8.0.0
tqdm>=4.64.0
```

### EVO-1 Dependencies
```bash
# Computer vision
opencv-python>=4.5.0
Pillow>=8.0.0

# WebSocket & async
websockets>=10.0
aiohttp>=3.8.0

# HuggingFace
transformers>=4.20.0
huggingface_hub>=0.10.0

# Autonomous driving
nuscenes-devkit>=1.1.0
```

### Jetson-Specific
```bash
# Headless OpenCV (for Jetson)
opencv-python-headless>=4.5.0

# Jetson utilities
jetson-stats>=4.2.0
pycuda>=2022.1
```

## 🚀 Installation Methods

### Method 1: Automated Scripts

#### Standard Installation
```bash
# Basic setup
python install_evo1_deps.py

# Or step-by-step
./deploy_jetson_stage1.sh
```

#### Fix Common Issues
```bash
# Fix flash-attn issues
./fix_flash_attn.sh

# Test installation
python test_install_fix.py
```

### Method 2: Manual Installation

#### Step 1: PyTorch
```bash
# CUDA systems
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only systems
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Jetson devices
# Use JetPack-specific PyTorch
```

#### Step 2: Core Dependencies
```bash
pip install numpy pandas matplotlib tqdm psutil pyyaml
```

#### Step 3: EVO-1 Specific
```bash
pip install opencv-python websockets transformers nuscenes-devkit
```

#### Step 4: Optional Components
```bash
# Flash attention (optional, may fail to compile)
export MAX_JOBS=2
pip install flash-attn --no-build-isolation || echo "Skipped (optional)"

# Additional ML tools
pip install accelerate deepspeed timm
```

### Method 3: Staged Installation

```bash
# Stage 1: Core dependencies
pip install -r requirements_evo1_stages.txt

# Stage 2: ML and advanced features
pip install -r requirements_evo1_stage2.txt

# Stage 3: Optional and specialized
pip install -r requirements_evo1_stage3.txt
```

## 🔍 Verification

### Test Installation
```bash
# Basic test
python -c "import torch; print('PyTorch:', torch.__version__)"

# EVO-1 test
python test_autonomous_driving.py

# Jetson Stage 1 test
python test_jetson_stage1.py

# Installation verification
python test_install_fix.py
```

### Component Testing
```bash
# Test federated learning
python examples/simple_simulation.py

# Test autonomous driving
python examples/enhanced_simulation.py

# Test Stage 1 EVO-1
python examples/evo1_stage1_federated.py
```

## ⚠️ Troubleshooting

### Common Issues

#### PyTorch Installation
```bash
# CUDA not found
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Version conflicts
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

#### Flash-Attention Issues
```bash
# Compilation failed
export MAX_JOBS=2
pip install flash-attn --no-build-isolation

# Or skip entirely (it's optional)
pip install -r requirements_jetson_stage1.txt
```

#### Memory Issues
```bash
# Reduce batch size in config
# Set swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### CUDA Issues
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Match PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Platform-Specific

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip build-essential
```

#### Jetson
```bash
# Set maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Install JetPack packages
pip install jetson-stats pycuda
```

#### macOS
```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# For Apple Silicon (M1/M2)
# Use PyTorch with MPS support
```

## 🎯 Post-Installation

### Configuration
```bash
# Create config
cp config_example.yaml config.yaml
# Edit for your setup
```

### Datasets
```bash
# Download test data
mkdir -p data
# Add your datasets here
```

### First Run
```bash
# Quick test
python examples/simple_simulation.py

# Full EVO-1 test
python examples/autonomous_driving_simulation.py
```

## 📚 Documentation Links

- [Project Overview](../README.md) - Complete project introduction
- [EVO-1 Integration](../README_AUTONOMOUS_DRIVING.md) - Autonomous driving system
- [Jetson Stage 1](../README_EVO1_STAGE1.md) - Edge device optimization
- [Implementation Details](../IMPLEMENTATION_SUMMARY.md) - Technical details
- [Troubleshooting](../INSTALLATION_FIX.md) - Installation issues

## 🤝 Support

### Getting Help
1. Check [Installation Fix](../INSTALLATION_FIX.md) for common issues
2. Verify with [Test Scripts](../#verification)
3. Check platform-specific instructions above
4. Open an issue with system information

### System Information for Bug Reports
```bash
# Collect system info
python -c "
import sys, torch, platform
print('Python:', sys.version)
print('Platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device:', torch.cuda.get_device_name(0))
"
```

---

🎉 **Congratulations!** You now have FHDP and EVO-1 installed. Check the documentation for your specific use case.