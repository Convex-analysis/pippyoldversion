#!/bin/bash

echo "🔍 Jetson AGX Orin Diagnostic Tool"
echo "=================================="
echo ""

# 1. Check JetPack version from multiple sources
echo "📋 Checking JetPack version..."
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ Found /etc/nv_tegra_release:"
    cat /etc/nv_tegra_release
elif [ -f /etc/nv_tegra-release ]; then
    echo "✅ Found /etc/nv_tegra-release:"
    cat /etc/nv_tegra-release
else
    echo "⚠️  Standard JetPack release file not found"
    echo "   Checking dpkg for jetpack packages..."
    dpkg -l | grep -i jetpack | head -5 || echo "   No jetpack packages found via dpkg"
fi
echo ""

# 2. Check CUDA version
echo "🎯 Checking CUDA version..."
if command -v nvcc >/dev/null 2>&1; then
    echo "✅ nvcc found:"
    nvcc --version
else
    echo "⚠️  nvcc not found in PATH"
fi
echo ""

# 3. Check existing PyTorch
echo "🔥 Checking PyTorch installation..."
if python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    echo "✅ PyTorch is already installed:"
    python3 -c "import torch; print(f'   Version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}')"
else
    echo "⚠️  PyTorch not installed"
fi
echo ""

# 4. Check CUDA libraries
echo "📚 Checking CUDA libraries..."
CUDA_PATHS=("/usr/local/cuda" "/usr/local/cuda-12" "/usr/local/cuda-11")
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ Found CUDA at $path"
        echo "   Version: $path | $(basename $path)"
    fi
done
echo ""

# 5. Detect architecture
echo "🏗️  Checking GPU architecture..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU info:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi
echo ""

# 6. Recommend installation approach
echo "💡 Installation Recommendation:"
echo "================================"

# Detect system details
IS_JETSON=false
JETPACK_VER="unknown"
PYTORCH_URL=""

if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=true
    JETPACK_VER=$(grep -oE 'R[0-9]+\.[0-9]+' /etc/nv_tegra_release 2>/dev/null || grep -oE 'R[0-9]+' /etc/nv_tegra_release 2>/dev/null || echo "unknown")
fi

if [ "$IS_JETSON" = true ]; then
    case "$JETPACK_VER" in
        R36*)
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
            echo "   Detected JetPack 6.0 (R36)"
            echo "   Recommended URL: $PYTORCH_URL"
            ;;
        R35*)
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch"
            echo "   Detected JetPack 5.x (R35)"
            echo "   Recommended URL: $PYTORCH_URL"
            ;;
        R34*)
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch"
            echo "   Detected JetPack 4.x (R34)"
            echo "   Recommended URL: $PYTORCH_URL"
            ;;
        *)
            echo "   ⚠️  Unknown JetPack version: $JETPACK_VER"
            echo "   Trying JetPack 6.0 URL (may not work):"
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
            echo "   URL: $PYTORCH_URL"
            echo ""
            echo "   💡 If this doesn't work, try checking:"
            echo "      cat /etc/nv_tegra_release"
            echo "      or install JetPack documentation"
            ;;
    esac
else
    echo "   ⚠️  Not a standard JetPack system"
    echo "   Install PyTorch from PyPI with CUDA support"
fi

echo ""
echo "📝 Quick install command:"
if [ -n "$PYTORCH_URL" ]; then
    echo "   pip3 install --upgrade pip"
    echo "   pip3 install --no-cache-dir torch torchvision torchaudio --index-url $PYTORCH_URL"
fi

echo ""
echo "✅ Diagnostic complete!"
