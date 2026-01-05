# EVO-1 Installation Fix for Flash-Attention Issue

## 🚨 Problem Identified

The `flash-attn` installation failed because:
1. **Dependency Order**: `flash-attn` requires PyTorch to be installed first
2. **Compilation Issues**: Complex C++/CUDA compilation on some systems
3. **Build Environment**: Missing CUDA headers or incompatible compiler

## 🔧 Solutions (Choose One)

### Solution 1: Automated Fix (Recommended)

```bash
# Run the automated fix script
./fix_flash_attn.sh
```

This script will:
- ✅ Install PyTorch first
- ✅ Set proper environment variables
- ✅ Try multiple installation methods
- ✅ Skip flash-attn if it fails (it's optional)

### Solution 2: Staged Installation

```bash
# Stage 1: Install PyTorch and core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Stage 2: Install core packages
pip install -r requirements_jetson_stage1.txt

# Stage 3: Try flash-attn (optional)
export MAX_JOBS=2
pip install flash-attn --no-build-isolation || echo "flash-attn skipped (optional)"
```

### Solution 3: Skip Flash-Attention (Easiest)

Since `flash-attn` is optional (just provides faster attention):

```bash
# Install without flash-attn
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm psutil opencv-python-headless pyyaml
```

The system will work fine with standard attention, just a bit slower.

### Solution 4: Manual Flash-Attention Installation

If you really want flash-attn:

```bash
# 1. Ensure PyTorch is installed
python -c "import torch; print(torch.__version__)"

# 2. Set environment variables
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# 3. Try different methods
# Method A
pip install flash-attn --no-build-isolation

# Method B (if A fails)
pip install flash-attn --no-cache-dir

# Method C (if B fails)
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

## 🧪 Test Installation

```bash
# Test current installation
python test_install_fix.py
```

This will check:
- ✅ PyTorch installation
- ✅ Core dependencies
- ⚠️  flash-attn (optional)

## 📱 Jetson-Specific Instructions

### For Jetson Orin:
```bash
# Install PyTorch for Jetson Orin
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Stage 1 requirements
pip install -r requirements_jetson_stage1.txt
```

### For Jetson Nano:
```bash
# Install PyTorch for Jetson Nano (CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Stage 1 requirements
pip install -r requirements_jetson_stage1.txt
```

### Full Jetson Deployment:
```bash
# Run complete deployment script
./deploy_jetson_stage1.sh
```

## 🔍 Troubleshooting

### Error: "No module named 'torch'"
**Cause**: Trying to install flash-attn before PyTorch
**Solution**: Install PyTorch first
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "Failed to build wheel"
**Cause**: Compilation issues with CUDA/g++
**Solution**: Set environment variables and reduce jobs
```bash
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
pip install flash-attn --no-build-isolation
```

### Error: "MemoryError during build"
**Cause**: Not enough RAM for compilation
**Solution**: Reduce parallel jobs and free memory
```bash
export MAX_JOBS=1
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
pip install flash-attn
```

### Error: "CUDA not found"
**Cause**: Missing CUDA toolkit
**Solution**: Install CUDA or use CPU version
```bash
# Check if CUDA is available
nvidia-smi

# If not available, use CPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🎯 Quick Fix Summary

1. **Recommended**: `./fix_flash_attn.sh`
2. **Easiest**: Skip flash-attn (it's optional)
3. **Jetson**: `./deploy_jetson_stage1.sh`
4. **Test**: `python test_install_fix.py`

## 🚀 After Installation

Once dependencies are installed, you can run:

```bash
# Test the installation
python test_install_fix.py

# Run Stage 1 simulation
python examples/evo1_stage1_federated.py

# Run autonomous driving
python examples/autonomous_driving_simulation.py

# Run performance tests
python test_jetson_stage1.py
```

## 💡 Important Notes

1. **flash-attn is optional**: The system works fine without it
2. **Standard attention works**: Just 20-30% slower
3. **PyTorch first**: Always install PyTorch before flash-attn
4. **Environment matters**: Use `MAX_JOBS=2` for limited systems
5. **CUDA compatibility**: Ensure CUDA version matches PyTorch

## 🎉 Success Indicators

✅ **PyTorch installed**: `python -c "import torch"`  
✅ **Core dependencies work**: All numpy, pandas, etc.  
✅ **Flash-attn optional**: Either installed or skipped  
✅ **Can run simulation**: `python examples/evo1_stage1_federated.py`  

If you have these 4 things, you're ready to go! 🚀