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

| JetPack Version | PyTorch URL |
|----------------|-------------|
| JetPack 6.0 (R36) | https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch |
| JetPack 5.x (R35) | https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch |
| JetPack 5.3/5.4 | https://developer.download.nvidia.com/compute/redist/jp/v504/pytorch |
| JetPack 4.x (R34) | https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch |

Example for JetPack 6.0:
```bash
pip3 install --upgrade pip
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch
```

#### Step 3: Install Additional Dependencies

```bash
# Install packaging module first
pip3 install packaging

# Install Stage 1 requirements
pip3 install -r requirements_jetson_stage1.txt

# Install flash-attn (optional, may take 10-30 minutes)
export MAX_JOBS=2
pip3 install flash-attn --no-build-isolation
```

## Troubleshooting

### PyTorch Installation Fails with 404

**Error:** `404 Client Error: Not Found for url`

**Solution:** Make sure you're using the correct URL for your JetPack version. Check with:
```bash
cat /etc/nv_tegra_release
```

### Flash Attention Installation Fails

**Error:** Various build errors during flash-attn compilation

**Solutions:**
1. Install `packaging` module first: `pip3 install packaging`
2. Reduce parallel jobs: `export MAX_JOBS=1`
3. Skip flash-attn (optional): The code will use slower attention mechanism
4. Make sure PyTorch is installed before flash-attn

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
