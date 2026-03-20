# Jetson Installation Guide

## Prerequisites

- NVIDIA Jetson device (Orin Nano, Orin, Orin NX, AGX Orin, etc.)
- JetPack 4.6+, 5.x, or 6.x installed
- Python 3.8-3.11 recommended
- At least 5GB free disk space

## Quick Installation

### Option 1: Automated Script (Recommended)

```bash
# For Jetson Orin Nano specifically
bash ./deploy_jetson_orin_nano.sh

# For any Jetson device
bash ./deploy_jetson_stage1.sh
```

### Option 2: Manual Installation

#### Step 1: Check JetPack Version

```bash
cat /etc/nv_tegra_release
```

Look for the JetPack version (e.g., R35, R36, etc.)

#### Step 2: Install PyTorch for Your JetPack Version

**IMPORTANT:** You MUST install PyTorch from NVIDIA's Jetson repository, NOT from PyPI.

##### For JetPack 6.0 (R36)

```bash
pip3 install --upgrade pip
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch
```

##### For JetPack 5.x (R35)

PyTorch for JetPack 5.x is distributed as direct wheel files instead of a pip index.

```bash
pip3 install --upgrade pip
pip3 uninstall -y torch torchvision torchaudio

# Install PyTorch from wheel
pip3 install --no-cache-dir https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

**Important:** Pre-built torchvision and torchaudio wheels are NOT available for PyTorch 2.1.0 on JetPack 5.x. You have two options:

**Option 1: Compile from Source (Optional)**

```bash
# Compile torchvision (requires 30-60 minutes)
git clone https://github.com/pytorch/vision.git
cd vision
git fetch --tags
git checkout tags/v0.16.0
rm -rf build/ dist/ *.egg-info/ __pycache__/
pip3 uninstall -y torchvision
export CUDA_HOME=/usr/local/cuda
python3 setup.py install
cd ..

# Compile torchaudio (optional, requires additional dependencies)
git clone https://github.com/pytorch/audio.git
cd audio
python3 setup.py install
```

**Option 2: Skip torchvision/torchaudio**

If your project doesn't require image or audio processing, you can skip installing torchvision/torchaudio. PyTorch alone provides CUDA acceleration.

##### For JetPack 4.x (R34)

```bash
pip3 install --upgrade pip
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch
```

#### Step 3: Install Additional Dependencies

```bash
# Install packaging module first
pip3 install packaging

# Install Jetson-optimized requirements (recommended)
pip3 install -r requirements/jetson.txt

# Or install EVO-1 specific requirements
pip3 install -r requirements/evo1.txt

# Install flash-attn (optional, may take 10-30 minutes)
export MAX_JOBS=2
pip3 install flash-attn --no-build-isolation
```

## Troubleshooting

### PyTorch Installation Fails with 404

**Error:** `404 Client Error: Not Found for url`

**Solution:** Make sure you're using the correct installation method for your JetPack version:

- **JetPack 6.0 (R36)**: Use pip index URL: `--index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch`
- **JetPack 5.x (R35)**: Use direct wheel URL, NOT pip index. See Step 2 above.
- **JetPack 4.x (R34)**: Use pip index URL: `--index-url https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch`

Check your version with:
```bash
cat /etc/nv_tegra_release
```

### PyTorch Shows "CUDA available: False"

**Error:** PyTorch is installed but CUDA is not available

**Solutions:**

1. **Verify PyTorch was built with CUDA:**
   ```bash
   python3 -c "import torch; print('Built with CUDA:', torch.version.cuda)"
   ```
   If it shows `None`, you installed a CPU-only version. Reinstall from NVIDIA's Jetson repository.

2. **Update ldconfig cache:**
   ```bash
   sudo ldconfig
   ```

3. **Set CUDA environment variables:**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda
   ```

4. **For JetPack 5.x**, ensure you installed from the wheel URL, not PyPI.

### torchvision/torchaudio Not Available for JetPack 5.x

**Problem:** Pre-built torchvision/torchaudio wheels are not available for PyTorch 2.1.0 on JetPack 5.x

**Solutions:**

1. **Option 1 - Compile from source (Optional):**
   ```bash
   # torchvision
   git clone https://github.com/pytorch/vision.git
   cd vision
   git fetch --tags
   git checkout tags/v0.16.0
   rm -rf build/ dist/ *.egg-info/ __pycache__/
   pip3 uninstall -y torchvision
   export CUDA_HOME=/usr/local/cuda
   python3 setup.py install

   # torchaudio (optional)
   git clone https://github.com/pytorch/audio.git
   cd audio
   python3 setup.py install
   ```

2. **Option 2 - Skip if not needed:**
   torchvision is only needed for image processing. If your project doesn't use images, you can skip it.

### Flash Attention Installation Fails

**Error:** Various build errors during flash-attn compilation

**IMPORTANT: Jetson-Specific Considerations**

Jetson devices (especially Orin Nano with 6GB shared memory) have unique challenges with flash-attn:

| Jetson Model | GPU Architecture | Compute Capability | Recommended Action |
|--------------|-----------------|-------------------|-------------------|
| Orin Nano | Ampere | 8.7 | Skip (recommended) |
| Orin NX | Ampere | 8.7 | Skip (recommended) |
| AGX Orin | Ampere | 8.7 | Optional compile |
| Xavier NX | Volta | 7.2 | Skip (recommended) |
| AGX Xavier | Volta | 7.2 | Skip (recommended) |

**Why Skip Flash-Attention on Jetson?**

1. **Compilation Time**: 30-60 minutes on Jetson (vs 5-10 minutes on desktop)
2. **Memory Constraints**: Orin Nano has only 6GB shared memory (CPU+GPU), compilation often fails with OOM
3. **Limited Performance Gain**: Standard attention is only 20-30% slower on Jetson
4. **No Pre-built Wheels**: flash-attn only provides x86_64 wheels, not AArch64/Jetson

**Solution 1: Skip flash-attn (Recommended for Jetson)**

The easiest solution is to skip flash-attn entirely. The system will automatically use standard attention:

```bash
# Just skip the flash-attn installation step
# Your code will work with standard attention (slower but functional)
```

If using `deploy_jetson_orin_nano.sh`, flash-attn is skipped by default. To verify:

```bash
# The script will output:
# ⏭️  Skipping flash-attn installation (recommended for Jetson)
# ℹ️  Will use standard attention mechanism
```

**Solution 2: Force Compilation (Experimental)**

If you still want to compile flash-attn on Jetson, you **MUST** set the correct architecture:

```bash
# CRITICAL: Set TORCH_CUDA_ARCH_LIST for your Jetson model
# Orin series (Orin Nano, Orin NX, AGX Orin)
export TORCH_CUDA_ARCH_LIST="8.7"   # Ampere SM 8.7

# Xavier series
# export TORCH_CUDA_ARCH_LIST="7.2"   # Volta SM 7.2

# Other Jetson settings
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=1   # CRITICAL: Reduce to 1 for Jetson memory constraints

# Now install
pip3 install flash-attn --no-build-isolation
```

**Common Errors on Jetson and Fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `unsupported gpu architecture '8.0'` | Wrong TORCH_CUDA_ARCH_LIST | Set to `8.7` for Orin |
| `MemoryError` | MAX_JOBS too high | Set `MAX_JOBS=1` |
| `failed to build wheel` | Missing packaging | `pip3 install packaging` |
| `killed during compilation` | OOM | Skip flash-attn, not worth it |

**Solution 3: Force flash-attn in deployment script**

If using `deploy_jetson_orin_nano.sh` and want to force compilation:

```bash
FORCE_FLASH_ATTN=1 bash ./deploy_jetson_orin_nano.sh
```

The script will then:
- Automatically detect Jetson model and set `TORCH_CUDA_ARCH_LIST="8.7"` (for Orin)
- Set `MAX_JOBS=1` to prevent OOM
- Attempt compilation with proper environment variables

**Verification:**

After installation (or skipping), verify your setup works:

```bash
# Check if flash-attn is installed
python3 -c "import flash_attn; print('flash-attn version:', flash_attn.__version__)" 2>/dev/null && echo "✅ flash-attn installed" || echo "⚠️  flash-attn not installed (using standard attention)"

# Test PyTorch CUDA is working (most important)
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Solutions Summary:**

1. **Install `packaging` module first**: `pip3 install packaging`
2. **Reduce parallel jobs**: `export MAX_JOBS=1` (CRITICAL for Jetson)
3. **Set correct CUDA architecture**: `export TORCH_CUDA_ARCH_LIST="8.7"` (for Orin)
4. **Set CUDA environment variables**:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```
5. **Skip flash-attn (recommended for Jetson)**: The code will use slower attention mechanism
6. **Make sure PyTorch is installed before flash-attn**

### Memory Issues

**Error:** Out of memory during training

**Solutions:**
1. Reduce batch size in config file
2. Enable gradient checkpointing (already enabled in default config)
3. Use the memory optimizer utility: `python3 jetson_memory_optimizer.py`

### Thermal Issues

**Error:** Device overheating

**Solutions:**
1. Monitor temperature: `python3 jetson_monitor.py`
2. Enable thermal throttling in config
3. Ensure proper cooling/airflow

## Performance Optimization

### Set Power Mode to Maximum

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Use Memory Optimizer

```python
from jetson_memory_optimizer import JetsonMemoryOptimizer

optimizer = JetsonMemoryOptimizer(target_memory_percent=75.0)

# Check memory pressure
if optimizer.check_memory_pressure():
    # Optimize memory
    result = optimizer.optimize_memory(force=True)
```

### Monitor Performance

```bash
# Start performance monitor
python3 jetson_monitor.py

# Monitor in another terminal
watch -n 1 cat jetson_performance.log
```

## Resource Limits

| Device | Max Memory | Batch Size | Notes |
|--------|------------|------------|-------|
| Jetson AGX Orin | 16GB | 8 | High performance |
| Jetson Orin NX 8GB | 8GB | 6 | Balanced |
| Jetson Orin Nano 8GB | 6GB | 4 | Balanced |
| Jetson Orin Nano 4GB | 3GB | 2 | Power-optimized |
| Jetson Nano | 3GB | 2 | Legacy device |

## Verification

After installation, verify with:

```bash
# Check Python packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

# Quick test
python3 -c "import torch; x = torch.randn(3,3).cuda(); print('GPU test passed')"
```

## Additional Resources

- [NVIDIA JetPack Documentation](https://developer.nvidia.com/embedded/jetpack)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-2-0-now-available/72048)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

## Support

For issues specific to:
- Jetson devices: Check NVIDIA Developer Forums
- FHDP System: Check project documentation
- PyTorch on Jetson: Check NVIDIA PyTorch forum threads
