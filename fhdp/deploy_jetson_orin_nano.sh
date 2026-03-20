#!/bin/bash

# Simplified EVO-1 Stage 1 Deployment Script for Jetson Orin Nano
# This script handles the actual installation without complex model detection

echo "🚀 EVO-1 Stage 1 Deployment for Jetson Orin Nano"
echo "=================================================="

# Environment setup
echo ""
echo "🔧 Setting up environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "   Python version: $PYTHON_VERSION"

case "$PYTHON_VERSION" in
    3.8|3.9|3.10|3.11)
        ;;
    *)
        echo "⚠️  Warning: Python 3.8-3.11 recommended, found $PYTHON_VERSION"
        ;;
esac

# Jetson-specific optimizations
echo ""
echo "⚡ Applying Jetson optimizations..."

# Set power mode for performance
if command -v nvpmodel >/dev/null 2>&1; then
    echo "   Setting maximum power mode..."
    sudo nvpmodel -m 0 2>/dev/null || echo "   (nvpmodel not available or permissions required)"
fi

# Set maximum performance for CPU cores
if command -v jetson_clocks >/dev/null 2>&1; then
    echo "   Setting maximum CPU clocks..."
    sudo jetson_clocks 2>/dev/null || echo "   (jetson_clocks not available or permissions required)"
fi

# Install dependencies
echo ""
echo "📦 Installing Jetson-optimized dependencies..."

# Install required build dependencies first
echo "   Installing build dependencies (wheel, packaging)..."
pip3 install wheel packaging

# Check if this is a Jetson JetPack environment
IS_JETPACK=0
JETPACK_VERSION=""
if [ -f /etc/nv_tegra_release ]; then
    IS_JETPACK=1
    JETPACK_VERSION=$(grep -oE 'R[0-9]+' /etc/nv_tegra_release | head -1)
fi

echo "   JetPack environment: $([ "$IS_JETPACK" -eq 1 ] && echo "YES (${JETPACK_VERSION})" || echo "NO (generic ARM64 system)")"

# Install PyTorch for Jetson
echo "   Installing PyTorch..."

# Correct PyTorch wheel URLs for Jetson (from NVIDIA)
PYTORCH_URL=""

if [ "$IS_JETPACK" -eq 1 ]; then
    # Use NVIDIA Jetson-specific PyTorch only on real JetPack systems
    case "$JETPACK_VERSION" in
        R36)
            # JetPack 6.0 series (Orin)
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
            echo "   Using JetPack 6.0 PyTorch"
            ;;
        R35)
            # JetPack 5.x series - For JetPack 5.x, NVIDIA provides direct wheel files instead of pip index
            echo "   Using JetPack 5.x PyTorch (direct wheel installation)"
            echo "   Note: torchvision/torchaudio may need to be compiled from source"
            ;;
        R34)
            # JetPack 4.x series
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch"
            echo "   Using JetPack 4.x PyTorch"
            ;;
        *)
            # Default to JetPack 5.x (most common for Orin)
            echo "   ⚠️  Unknown JetPack version: ${JETPACK_VERSION:-unknown}"
            echo "   Defaulting to JetPack 5.x PyTorch"
            ;;
    esac

    # Set CUDA library paths before installing PyTorch
    if [ -d "/usr/local/cuda/lib64" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        echo "   Added CUDA library path to LD_LIBRARY_PATH"
    fi
    pip3 install --no-cache-dir --upgrade pip

    # Check current PyTorch CUDA support and install if needed
    echo "   Checking current PyTorch CUDA support..."
    if python3 -c "import torch; exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
        echo "   ✓ PyTorch already has CUDA support"
    else
        echo "   ⚠️  Current PyTorch is CPU-only, installing CUDA-enabled version..."

        if [ "$JETPACK_VERSION" = "R35" ]; then
            # For JetPack 5.x, install PyTorch from direct wheel URL
            echo "   Installing PyTorch from wheel for JetPack 5.x..."
            pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
            pip3 install --no-cache-dir https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl || echo "   ⚠️  PyTorch installation failed"

            # Note: torchvision/torchaudio not available as pre-built wheels for PyTorch 2.1.0 on JetPack 5.x
            echo ""
            echo "   ℹ️  PyTorch installed. For torchvision and torchaudio:"
            echo "   Option 1: Compile from source (optional, requires 30-60 minutes)"
            echo "      git clone https://github.com/pytorch/vision.git && cd vision"
            echo "      git fetch --tags && git checkout tags/v0.16.0"
            echo "      rm -rf build/ dist/ *.egg-info/ __pycache__/"
            echo "      pip3 uninstall -y torchvision"
            echo "      python3 setup.py install"
            echo "   Option 2: Skip if not needed (torchvision is for image processing)"
            echo ""
        elif [ -n "$PYTORCH_URL" ]; then
            # For other JetPack versions, use pip index
            echo "   Installing from: $PYTORCH_URL"
            pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
            pip3 install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url "$PYTORCH_URL" || echo "   ⚠️  PyTorch installation failed"
        fi
    fi
else
    # Not a JetPack system - install generic ARM64 PyTorch (CPU-only)
    echo "   ℹ️  Not a JetPack system, installing generic ARM64 PyTorch (CPU-only)"
    echo "   Note: CUDA will NOT be available on this system"
    pip3 install --no-cache-dir --upgrade pip
    pip3 install --no-cache-dir torch torchvision torchaudio || echo "   ⚠️  PyTorch installation failed"
fi

# Install Stage 1 requirements
echo "   Installing Stage 1 requirements..."
if [ -f requirements/jetson.txt ]; then
    echo "   Using optimized Jetson requirements..."
    pip3 install -r requirements/jetson.txt || echo "   ⚠️  Some dependencies failed"
elif [ -f requirements_jetson_stage1.txt ]; then
    pip3 install -r requirements_jetson_stage1.txt || echo "   ⚠️  Some dependencies failed"
elif [ -f requirements.txt ]; then
    pip3 install -r requirements.txt || echo "   ⚠️  Some dependencies failed"
else
    echo "   ⚠️  No requirements file found, installing minimal dependencies..."
    pip3 install numpy pandas psutil websockets aiohttp pyyaml python-dotenv tqdm
fi

# Install required build dependencies first
echo "   Installing build dependencies (wheel, packaging)..."
pip3 install wheel packaging

# Try flash-attn separately (optional)
echo "   Attempting flash-attn installation (optional, may take a while)..."
if [ "$IS_JETPACK" -eq 1 ]; then
    # Detect and set CUDA_HOME for flash-attn compilation
    if [ -z "$CUDA_HOME" ]; then
        echo "   Detecting CUDA_HOME..."
        for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
            if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
                export CUDA_HOME="$cuda_path"
                export PATH="$CUDA_HOME/bin:$PATH"
                export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
                echo "   Found CUDA at: $CUDA_HOME"
                break
            fi
        done
    fi

    if [ -n "$CUDA_HOME" ]; then
        # Verify nvcc is available
        if command -v nvcc >/dev/null 2>&1; then
            echo "   CUDA_HOME is set to: $CUDA_HOME"
            echo "   nvcc version: $(nvcc --version | grep release | head -1)"
            echo ""
            echo "   ⚙️  Flash-Attention Options for Jetson Orin Nano:"
            echo "   Jetson Orin Nano uses Ampere architecture (Compute Capability 8.7)"
            echo ""
            echo "   Option 1: Skip flash-attn (Recommended)"
            echo "   - Compiling flash-attn on Jetson takes 30-60 minutes"
            echo "   - 6GB shared memory makes compilation risky (OOM)"
            echo "   - Standard attention is only 20-30% slower"
            echo ""
            echo "   Option 2: Compile flash-attn (Experimental)"
            echo "   - Set TORCH_CUDA_ARCH_LIST=\"8.7\" for correct architecture"
            echo "   - Requires MAX_JOBS=1 to prevent memory overflow"
            echo ""
            echo "   Default: Skipping flash-attn (set FORCE_FLASH_ATTN=1 to compile)"
            echo ""

            # Check if user wants to force flash-attn compilation
            if [ "$FORCE_FLASH_ATTN" = "1" ]; then
                echo "   🔨 FORCE_FLASH_ATTN=1 detected, attempting compilation..."
                echo "   Setting TORCH_CUDA_ARCH_LIST=8.7 for Jetson Orin Nano"
                echo "   This may take 30-60 minutes with MAX_JOBS=1..."
                # Use env to ensure environment variables are passed to pip
                # CRITICAL: TORCH_CUDA_ARCH_LIST must be "8.7" for Jetson Orin Nano (Ampere SM 8.7)
                env CUDA_HOME="$CUDA_HOME" PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
                    MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="8.7" \
                    pip3 install flash-attn --no-build-isolation || {
                        echo "   ⚠️  flash-attn compilation failed (as expected on Jetson)"
                        echo "   ℹ️  This is normal - falling back to standard attention"
                    }
            else
                echo "   ⏭️  Skipping flash-attn installation (recommended for Jetson)"
                echo "   ℹ️  Will use standard attention mechanism"
                echo "   💡 To force compilation, run: FORCE_FLASH_ATTN=1 bash ./deploy_jetson_orin_nano.sh"
            fi
        else
            echo "   ⚠️  nvcc not found at $CUDA_HOME/bin/nvcc"
            echo "   ℹ️  Flash attention requires CUDA toolkit with nvcc compiler"
            echo "   ⚠️  Will use standard attention (slower)"
        fi
    else
        echo "   ⚠️  CUDA not found, skipping flash-attn installation"
        echo "   ℹ️  Flash attention requires CUDA to compile"
        echo "   ⚠️  Will use standard attention (slower)"
    fi
else
    echo "   ℹ️  Skipping flash-attn on non-JetPack system (requires CUDA)"
    echo "   ⚠️  Will use standard attention (slower)"
fi

# Create configuration file
echo ""
echo "⚙️  Creating Jetson Orin Nano configuration..."

cat > config_jetson_orin_nano.yaml << EOF
# EVO-1 Stage 1 Configuration for Jetson Orin Nano
jetson_config:
  model: "NVIDIA Orin Nano Developer Kit"
  is_jetpack: $([ "$IS_JETPACK" -eq 1 ] && echo "true" || echo "false")
  max_memory_mb: 6144
  max_batch_size: 4
  precision: "float16"
  power_optimized: false
  cuda_enabled: $([ "$IS_JETPACK" -eq 1 ] && echo "true" || echo "false")

stage1_training:
  vlm_frozen: true
  trainable_modules: ["integration_module", "action_head"]
  max_epochs_per_round: 2
  learning_rate: 1e-4
  weight_decay: 1e-3
  gradient_checkpointing: true
  mixed_precision: true

federated_learning:
  aggregation_interval: 45  # seconds
  training_interval: 30     # seconds
  communication_protocol: "websocket"
  compression_enabled: true
  update_quantization: "int8"

memory_optimization:
  enable_gc_frequency: 5      # Clear cache every 5 steps
  max_cache_size_mb: 512     # Limit cache size
  tensor_recycling: true
  memory_monitoring: true

thermal_management:
  temperature_threshold: 85.0   # Celsius
  performance_throttling: true
  thermal_check_interval: 5    # seconds

logging:
  level: "INFO"
  log_directory: "./logs/jetson_orin_nano"
  enable_tensorboard: false      # Save memory
  performance_monitoring: true
EOF

echo "✅ Configuration saved to config_jetson_orin_nano.yaml"

# Installation verification
echo ""
echo "🔍 Verifying installation..."

# Check Python packages
echo "   Checking Python packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not installed"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" 2>/dev/null || echo "❌ TorchVision not installed"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not installed"
python3 -c "import psutil; print(f'PSUtil: {psutil.__version__}')" 2>/dev/null || echo "❌ PSUtil not installed"

# Check CUDA availability
echo "   Checking CUDA availability..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
echo "   CUDA available: $CUDA_AVAILABLE"

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "   CUDA device: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")"
    python3 -c "import torch; print(f'   CUDA version: {torch.version.cuda}')" 2>/dev/null
else
    # Provide detailed diagnostic information when CUDA is not available
    echo ""
    echo "   🔍 CUDA Diagnostics:"
    echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
    echo "   CUDA_HOME: ${CUDA_HOME:-<not set>}"
    echo "   PATH includes CUDA: $(echo $PATH | grep -o cuda || echo 'no')"
    echo ""
    echo "   Checking for CUDA libraries:"
    [ -f /usr/local/cuda/lib64/libcudart.so ] && echo "   ✓ /usr/local/cuda/lib64/libcudart.so found" || echo "   ✗ /usr/local/cuda/lib64/libcudart.so NOT found"
    [ -f /usr/local/cuda/lib64/libnvrtc.so ] && echo "   ✓ /usr/local/cuda/lib64/libnvrtc.so found" || echo "   ✗ /usr/local/cuda/lib64/libnvrtc.so NOT found"
    [ -f /usr/local/cuda/lib64/libcublas.so ] && echo "   ✓ /usr/local/cuda/lib64/libcublas.so found" || echo "   ✗ /usr/local/cuda/lib64/libcublas.so NOT found"
    echo ""
    echo "   Checking /usr/local/lib for CUDA libraries:"
    ls -la /usr/local/lib/libcudart* 2>/dev/null | head -3 || echo "   No libcudart found in /usr/local/lib"
    echo ""
    echo "   Checking PyTorch CUDA build:"
    python3 -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   PyTorch built with CUDA: {torch.version.cuda if torch.version.cuda else \"No\"}')" 2>/dev/null || echo "   Failed to check PyTorch CUDA info"
    echo ""
    echo "   ⚠️  CUDA is not available in PyTorch"
    echo "   Possible reasons:"
    echo "   1. PyTorch was built/installed without CUDA support (CPU-only version)"
    echo "   2. CUDA runtime libraries cannot be found by dynamic linker"
    echo "   3. PyTorch CUDA version doesn't match system CUDA version"
    echo ""
    echo "   Recommended actions:"
    echo "   1. Verify PyTorch was installed from NVIDIA JetPack repository"
    echo "   2. Check: python3 -c \"import torch; print(torch.cuda.is_available())\""
    echo "   3. If still false, try reinstalling PyTorch from JetPack wheels:"
    echo "      pip3 uninstall torch torchvision torchaudio"
    echo "      pip3 install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch"
fi

echo ""
echo "✅ Jetson Orin Nano Stage 1 deployment completed!"
echo ""
echo "📋 System Info:"
echo "- JetPack Environment: $([ "$IS_JETPACK" -eq 1 ] && echo "YES (${JETPACK_VERSION})" || echo "NO (generic ARM64)")"
echo "- PyTorch installed: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "FAILED")"
echo "- CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo "UNKNOWN")"
echo ""
echo "📋 Next steps:"
echo "1. Check CUDA availability: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Test PyTorch: python3 -c 'import torch; x = torch.randn(3,3); print(x)'"
echo "3. Run your training script"
echo ""
if [ "$IS_JETPACK" -eq 1 ]; then
    echo "🔧 Performance tips:"
    echo "- Monitor temperature during operation"
    echo "- Use thermal throttling if needed"
    echo "- Monitor memory usage and adjust batch sizes"
    echo ""
    echo "📊 Resource limits for Jetson Orin Nano:"
    echo "- Max Memory: 6GB"
    echo "- Max Batch Size: 4"
    echo "- Precision: float16 (optimized)"
    echo "- Power Mode: Balanced performance"
else
    echo "⚠️  Important Notes:"
    echo "- This is NOT a JetPack system, CUDA is NOT available"
    echo "- Running in CPU-only mode (slower performance)"
    echo "- For GPU acceleration, deploy on a real Jetson Orin Nano device"
    echo ""
    echo "📊 Resource limits:"
    echo "- Max Memory: System dependent"
    echo "- Max Batch Size: Reduce to 1-2 for CPU mode"
    echo "- Precision: float32 (CPU optimized)"
fi
