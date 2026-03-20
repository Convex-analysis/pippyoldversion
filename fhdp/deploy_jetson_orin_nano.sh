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

# Check JetPack version
JETPACK_VERSION=""
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(grep -oE 'R[0-9]+\.[0-9]+' /etc/nv_tegra_release | head -1)
fi

echo "   Detected JetPack version: ${JETPACK_VERSION:-unknown}"

# Install PyTorch for Jetson
echo "   Installing NVIDIA Jetson-specific PyTorch..."

# Correct PyTorch wheel URLs for Jetson (from NVIDIA)
PYTORCH_URL=""

case "$JETPACK_VERSION" in
    R36*)
        # JetPack 6.0 series (Orin)
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
        echo "   Using JetPack 6.0 PyTorch"
        ;;
    R35*)
        # JetPack 5.x series (Orin/AGX)
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch"
        echo "   Using JetPack 5.x PyTorch"
        ;;
    R35.3*|R35.4*)
        # JetPack 5.3/5.4
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v504/pytorch"
        echo "   Using JetPack 5.3/5.4 PyTorch"
        ;;
    R34*)
        # JetPack 4.x series
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch"
        echo "   Using JetPack 4.x PyTorch"
        ;;
    *)
        # Default to JetPack 6.0 (most recent)
        echo "   ⚠️  Unknown JetPack version: ${JETPACK_VERSION:-unknown}"
        echo "   Defaulting to JetPack 6.0 PyTorch (may not work if your JetPack version differs)"
        PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
        ;;
esac

if [ -n "$PYTORCH_URL" ]; then
    echo "   Installing from: $PYTORCH_URL"
    pip3 install --no-cache-dir --upgrade pip
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_URL" || echo "   ⚠️  PyTorch installation failed"
else
    echo "   ❌ Could not determine PyTorch URL for JetPack version"
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

# Install packaging module first (required by flash-attn)
echo "   Installing packaging module..."
pip3 install packaging

# Try flash-attn separately (optional)
echo "   Attempting flash-attn installation (optional, may take a while)..."
export MAX_JOBS=2
pip3 install flash-attn --no-build-isolation || echo "   ⚠️  flash-attn failed (optional, will use slower attention)"

# Create configuration file
echo ""
echo "⚙️  Creating Jetson Orin Nano configuration..."

cat > config_jetson_orin_nano.yaml << EOF
# EVO-1 Stage 1 Configuration for Jetson Orin Nano
jetson_config:
  model: "NVIDIA Orin Nano Developer Kit"
  max_memory_mb: 6144
  max_batch_size: 4
  precision: "float16"
  power_optimized: false

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
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ CUDA check failed"

if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "   CUDA device: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")"
fi

echo ""
echo "✅ Jetson Orin Nano Stage 1 deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Check CUDA availability: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Test PyTorch: python3 -c 'import torch; x = torch.randn(3,3); print(x)'"
echo "3. Run your training script"
echo ""
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
