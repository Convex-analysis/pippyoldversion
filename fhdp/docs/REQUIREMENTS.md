# FHDP Requirements Management

FHDP uses a modular requirements system to provide flexible, platform-specific dependency management.

## 🚀 Quick Start

### Intelligent Installation (Recommended)
```bash
# Auto-detect platform and install appropriate dependencies
python scripts/install_requirements.py --detect-platform

# Show system information and recommendations
python scripts/install_requirements.py --info
```

### Manual Installation
```bash
# Basic FHDP system
pip install -r requirements/base.txt

# Complete FHDP with all features
pip install -r requirements/complete.txt

# Jetson device optimized
pip install -r requirements/jetson.txt

# Minimal installation (core only)
pip install -r requirements/minimal.txt
```

## 📁 Requirements Structure

The `requirements/` folder contains modular dependency files:

| File | Purpose | Size | Use Case |
|------|---------|------|----------|
| `minimal.txt` | Core dependencies only | ~2GB | Basic FHDP functionality |
| `base.txt` | Standard FHDP system | ~3GB | Complete FHDP without ML |
| `ml.txt` | Machine learning features | ~1GB | Model training and inference |
| `evo1.txt` | EVO-1 model support | ~1.5GB | Autonomous driving |
| `jetson.txt` | Jetson optimized | ~2GB | Embedded deployment |
| `simulation.txt` | Simulation environments | ~500MB | Testing and development |
| `development.txt` | Development tools | ~1GB | Code development |
| `complete.txt` | All dependencies | ~7GB | Full FHDP system |

## 🔧 Installation Options

### Platform-Specific

#### Standard Systems (Linux/Mac/Windows)
```bash
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
pip install -r requirements/evo1.txt
```

#### Jetson Devices (Nano/Xavier/Orin)
```bash
pip install -r requirements/jetson.txt
```

#### Development Environment
```bash
pip install -r requirements/complete.txt
pip install -r requirements/development.txt
```

### Feature-Specific

#### Basic FHDP (No ML)
```bash
pip install -r requirements/base.txt
```

#### EVO-1 Autonomous Driving
```bash
pip install -r requirements/base.txt
pip install -r requirements/evo1.txt
```

#### Simulation Testing
```bash
pip install -r requirements/base.txt
pip install -r requirements/simulation.txt
```

## 🎯 Intelligent Installer

The `scripts/install_requirements.py` script provides automated installation:

```bash
# Available commands
python scripts/install_requirements.py --help
python scripts/install_requirements.py --detect-platform  # Auto-install
python scripts/install_requirements.py --minimal           # Minimal
python scripts/install_requirements.py --complete          # Complete
python scripts/install_requirements.py --info              # System info
python scripts/install_requirements.py --list              # Available files
```

### Auto-Detection Features

The installer automatically detects:
- **Platform**: Linux, macOS, Windows, or Jetson
- **Hardware**: CUDA availability, GPU memory
- **Python Version**: Compatibility checks
- **System Resources**: Memory and storage constraints

## ⚡ Performance Optimizations

### Jetson Optimizations
- **Headless OpenCV**: Saves ~500MB memory
- **Memory-optimized dependencies**: Reduced footprint
- **Compilation-safe**: Excludes problematic flash-attn
- **GPU acceleration**: CUDA-enabled PyTorch

### Standard Systems
- **Flash Attention**: Optional performance boost
- **GPU acceleration**: CUDA support when available
- **Parallel processing**: Multi-threaded operations
- **Memory management**: Efficient garbage collection

## 🛠️ Troubleshooting

### Common Issues

#### Flash Attention Installation
```bash
# If flash-attn fails to compile
export MAX_JOBS=2
pip install flash-attn --no-build-isolation

# Or use the jetson requirements (excludes flash-attn)
pip install -r requirements/jetson.txt
```

#### Memory Issues
```bash
# Use minimal requirements for constrained systems
pip install -r requirements/minimal.txt

# Or install components separately
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
```

#### Platform-Specific Issues
```bash
# Check platform detection
python scripts/install_requirements.py --info

# Use platform-specific installer
python scripts/install_requirements.py --platform-specific
```

### Verification

After installation, verify your setup:

```bash
# Test basic imports
python -c "import torch, cv2, numpy; print('✅ Basic dependencies working')"

# Test ML components
python -c "import transformers, accelerate; print('✅ ML components working')"

# Test EVO-1 components  
python -c "import nuscenes, websockets; print('✅ EVO-1 components working')"

# Run verification script
python verify_organization.py
```

## 🔄 Migration from Old System

If you were using the old requirements files:

1. **Old files** have been cleaned up automatically
2. **New system** uses modular `requirements/` folder
3. **Smart installer** handles platform detection
4. **Backward compatibility** maintained via root `requirements.txt`

### Migration Steps
```bash
# Remove old installation (optional)
pip uninstall -r old_requirements.txt -y

# Install with new system
python scripts/install_requirements.py --detect-platform

# Verify installation
python verify_organization.py
```

## 📞 Support

For installation issues:

1. Check system compatibility: `python scripts/install_requirements.py --info`
2. Try minimal installation: `pip install -r requirements/minimal.txt`
3. Consult documentation: `docs/INSTALLATION.md`
4. Use verification: `python verify_organization.py`

## 📈 Requirements Evolution

The requirements system is designed to:

- **Scale**: Add new modules without breaking existing installations
- **Adapt**: Platform-specific optimizations
- **Maintain**: Clear version management and dependency resolution
- **Automate**: Intelligent installation and error handling