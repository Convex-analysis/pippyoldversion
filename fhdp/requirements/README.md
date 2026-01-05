# FHDP Requirements Configuration

This directory contains modular requirements files for different FHDP use cases.

## 📁 File Structure

- **`minimal.txt`** - Core dependencies for basic FHDP functionality
- **`base.txt`** - Standard FHDP system requirements
- **`ml.txt`** - Machine learning and advanced features
- **`evo1.txt`** - EVO-1 vision-language-action model
- **`jetson.txt`** - Optimized for Jetson devices
- **`simulation.txt`** - Simulation and testing environments
- **`development.txt`** - Development and testing tools
- **`complete.txt`** - All dependencies for full functionality

## 🚀 Quick Start

### Basic FHDP System
```bash
pip install -r requirements/base.txt
```

### EVO-1 Model Support
```bash
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
pip install -r requirements/evo1.txt
```

### Jetson Device Optimization
```bash
pip install -r requirements/jetson.txt
```

### Complete Installation
```bash
pip install -r requirements/complete.txt
```

### Development Environment
```bash
pip install -r requirements/complete.txt
pip install -r requirements/development.txt
```

## 🔧 Installation Scripts

Use the intelligent installer for automatic setup:

```bash
python scripts/install_requirements.py --help
```

## 📋 Component Details

### Minimal (core only)
- PyTorch, OpenCV, NumPy, Pandas
- Basic configuration and system utilities
- **Size**: ~2GB
- **Use case**: Basic FHDP without ML features

### Base (standard FHDP)
- All minimal dependencies
- Advanced data processing and visualization
- Async programming support
- **Size**: ~3GB
- **Use case**: Standard FHDP functionality

### ML (machine learning)
- Accelerate, DeepSpeed, TIMM
- Monitoring and visualization tools
- Image processing utilities
- **Size**: ~1GB additional
- **Use case**: Model training and inference

### EVO-1 (vision-language-action)
- HuggingFace ecosystem
- nuScenes dataset support
- WebSocket communication
- **Size**: ~1.5GB additional
- **Use case**: Autonomous driving with EVO-1

### Jetson (optimized)
- Headless OpenCV for embedded devices
- Memory-optimized dependencies
- Compilation-safe (no flash-attn)
- **Size**: ~2GB
- **Use case**: Jetson Nano/Xavier/Orin deployment

### Optional (advanced features)
- Point cloud processing (Open3D)
- Advanced simulation environments
- Performance optimizations (flash-attn)
- **Size**: Variable (1-3GB additional)
- **Use case**: Advanced research and development

## ⚠️ Important Notes

1. **Flash Attention**: Optional dependency that may fail compilation on some systems. Available in `optional.txt`.

2. **Open3D**: Point cloud processing library that may cause compilation issues. Available in `optional.txt`.

3. **Installation Order**: Some dependencies require others to be installed first. The `complete.txt` file handles this automatically.

4. **Platform Compatibility**: Jetson requirements use `opencv-python-headless` for embedded deployment.

5. **Memory Constraints**: Jetson devices may need additional memory management for larger models.

6. **Optional Dependencies**: Heavy packages that may fail compilation are separated into `optional.txt` for manual installation.

## 🔍 Troubleshooting

If you encounter installation issues:

1. Use the installer script: `python scripts/install_requirements.py --detect-platform`
2. Try minimal installation first: `pip install -r requirements/minimal.txt`
3. Check for platform-specific requirements
4. Ensure Python version compatibility (3.8+ recommended)

## 📞 Support

For installation issues, check the documentation at `docs/INSTALLATION.md` or use the verification script: `python verify_organization.py`