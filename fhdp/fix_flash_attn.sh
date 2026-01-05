#!/bin/bash

# Quick Fix for Flash-Attention Installation Issue
echo "🔧 Fixing flash-attn installation issue..."

# Check if PyTorch is installed
python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not installed. Installing PyTorch first..."
    
    # Install PyTorch for current system
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "🚀 CUDA detected, installing CUDA version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "💻 No CUDA detected, installing CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    echo "✅ PyTorch installation completed"
fi

# Now try flash-attn with proper environment
echo "⚡ Installing flash-attn with proper configuration..."

# Set environment variables for compilation
export MAX_JOBS=2  # Reduce parallel jobs to avoid memory issues
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"  # Common GPU architectures
export CMAKE_ARGS="-DFFLAGS='-O3 -march=native'"

# Method 1: Try without build isolation (recommended)
echo "🔄 Method 1: Installing without build isolation..."
python3 -m pip install flash-attn --no-build-isolation

if [ $? -eq 0 ]; then
    echo "✅ flash-attn installed successfully!"
    echo "🎉 Installation completed!"
    exit 0
fi

# Method 2: Try with CUDA specific version
echo "🔄 Method 2: Trying CUDA-specific version..."
python3 -m pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases

if [ $? -eq 0 ]; then
    echo "✅ flash-attn installed successfully!"
    echo "🎉 Installation completed!"
    exit 0
fi

# Method 3: Skip flash-attn and continue
echo "🔄 Method 3: Skipping flash-attn (optional dependency)"
echo "   flash-attn is optional - you can continue without it"
echo "   The system will use standard attention instead"

# Install core dependencies without flash-attn
echo "📦 Installing core EVO-1 dependencies without flash-attn..."
python3 -m pip install numpy pandas matplotlib tqdm psutil opencv-python-headless pyyaml

echo "✅ Core dependencies installed!"
echo "⚠️  flash-attn not installed (optional - may use slower attention)"
echo "🎉 Installation completed - you can continue with the simulation!"

echo ""
echo "💡 If you want to try flash-attn later:"
echo "   1. Ensure PyTorch is installed"
echo "   2. Set export MAX_JOBS=2"
echo "   3. Run: pip install flash-attn --no-build-isolation"
echo ""
echo "🚀 Ready to run EVO-1 simulation!"
echo "   python3 examples/autonomous_driving_simulation.py"
echo "   python3 examples/evo1_stage1_federated.py"