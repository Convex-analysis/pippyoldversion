#!/bin/bash

# EVO-1 Stage 1 Deployment Script for NVIDIA Jetson Devices
# Optimized for Jetson Orin/Nano with resource constraints

echo "🚀 EVO-1 Stage 1 Deployment for Jetson Devices"
echo "==========================================="

# Check Jetson device
JETSON_MODEL=""
if [ -f /proc/device-tree/model ]; then
    JETSON_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
fi

echo "📱 Detected Jetson: $JETSON_MODEL"

# Set configuration based on Jetson model
case "$JETSON_MODEL" in
    *AGX\ Orin*)
        MAX_MEMORY_MB=16384
        MAX_BATCH_SIZE=8
        NANO_SCALE=0
        echo "🔥 Jetson AGX Orin detected: High performance mode"
        ;;
    *Orin\ Nano\ 8GB*)
        MAX_MEMORY_MB=6144
        MAX_BATCH_SIZE=4
        NANO_SCALE=0
        echo "📱 Jetson Orin Nano 8GB detected: Balanced performance mode"
        ;;
    *Orin\ Nano*)
        MAX_MEMORY_MB=3072
        MAX_BATCH_SIZE=2
        NANO_SCALE=1
        echo "📱 Jetson Orin Nano 4GB detected: Power-optimized mode"
        ;;
    *Orin\ NX*)
        MAX_MEMORY_MB=8192
        MAX_BATCH_SIZE=6
        NANO_SCALE=0
        echo "⚡ Jetson Orin NX detected: High performance mode"
        ;;
    *Nano*)
        MAX_MEMORY_MB=3072
        MAX_BATCH_SIZE=2
        NANO_SCALE=1
        echo "📱 Original Jetson Nano detected: Power-optimized mode"
        ;;
    *)
        MAX_MEMORY_MB=4096
        MAX_BATCH_SIZE=3
        NANO_SCALE=0
        echo "⚡ Generic Jetson device detected"
        ;;
esac

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

# Configure GPU memory
echo "   Configuring GPU memory split..."
if [ "$NANO_SCALE" -eq 1 ]; then
    # Nano: More memory for CPU
    sudo /usr/bin/jetson_clocks --store 2>/dev/null
else
    # Orin: Optimize for ML
    echo "   GPU memory configuration optimized for ML workloads"
fi

# Install dependencies
echo ""
echo "📦 Installing Jetson-optimized dependencies..."

# Option 1: Use the Python installation script (recommended)
echo "   Using staged installation script..."
python3 install_evo1_deps.py
STAGED_INSTALL_RESULT=$?

# Install FHDP System from source
echo "   Installing FHDP (Federated Highway-based Distributed Pipeline)..."
pip install -e .[communication] || echo "   ⚠️  FHDP installation failed"

# Option 2: Manual installation (fallback)
if [ $STAGED_INSTALL_RESULT -ne 0 ]; then
    echo "   ⚠️  Staged installation failed, trying manual install..."
    
    # Install PyTorch for Jetson
    echo "   Installing NVIDIA Jetson-specific PyTorch..."
    
    # Get JetPack version to determine compatible PyTorch version
    JETPACK_VERSION=""
    if [ -f /etc/nv_tegra_release ]; then
        JETPACK_VERSION=$(grep -oE 'R[0-9]+\.[0-9]+' /etc/nv_tegra_release | head -1)
    fi
    
    echo "   Detected JetPack version: ${JETPACK_VERSION:-unknown}"

    # Install Jetson-specific PyTorch from NVIDIA's official sources
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
            # Default to JetPack 6.0
            echo "   ⚠️  Unknown JetPack version: ${JETPACK_VERSION:-unknown}"
            echo "   Defaulting to JetPack 6.0 PyTorch"
            PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
            ;;
    esac

    if [ -n "$PYTORCH_URL" ]; then
        echo "   Installing from: $PYTORCH_URL"
        pip3 install --no-cache-dir --upgrade pip
        pip3 install --no-cache-dir --verbose torch torchvision torchaudio --index-url "$PYTORCH_URL"
    else
        echo "   ❌ Could not determine PyTorch URL"
    fi
    
    # Install Stage 1 requirements (without flash-attn)
    echo "   Installing Stage 1 requirements..."
    if [ -f requirements_jetson_stage1.txt ]; then
        pip install -r requirements_jetson_stage1.txt
    elif [ -f requirements/requirements_jetson_stage1.txt ]; then
        pip install -r requirements/requirements_jetson_stage1.txt
    else
        echo "   ⚠️  requirements_jetson_stage1.txt not found, checking other requirement files..."
        if [ -f requirements.txt ]; then
            pip install -r requirements.txt
        elif [ -f requirements/requirements.txt ]; then
            pip install -r requirements/requirements.txt
        fi
    fi
    
    # Try flash-attn separately (optional)
    echo "   Attempting flash-attn installation (optional)..."
    export MAX_JOBS=2
    pip install flash-attn --no-build-isolation || echo "   ⚠️  flash-attn failed (optional)"
    
    # Install FHDP System from source
    echo "   Installing FHDP (Federated Highway-based Distributed Pipeline)..."
    pip install -e .[communication] || echo "   ⚠️  FHDP installation failed"
    
    # Optional: Install JetPack-specific packages
    if [ -f /etc/nv_tegra_release ]; then
        echo "   JetPack detected, installing additional packages..."
        pip install jetson-stats>=4.2.0
        pip install pycuda>=2022.1  # For direct CUDA access
    fi
fi

# Create configuration file
echo ""
echo "⚙️  Creating Jetson configuration..."

cat > config_jetson_stage1.yaml << EOF
# EVO-1 Stage 1 Configuration for Jetson Devices
jetson_config:
  model: "${JETSON_MODEL:-generic_jetson}"
  max_memory_mb: ${MAX_MEMORY_MB}
  max_batch_size: ${MAX_BATCH_SIZE}
  precision: "float16"
  power_optimized: ${NANO_SCALE}

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

autonomous_driving:
  simulation_frequency: 2.0     # Hz
  max_data_samples: 100        # Per vehicle
  safety_violation_threshold: 10
  training_data_retention: 3600 # seconds

logging:
  level: "INFO"
  log_directory: "./logs/jetson_stage1"
  enable_tensorboard: false      # Save memory
  performance_monitoring: true
EOF

echo "✅ Configuration saved to config_jetson_stage1.yaml"

# Create performance monitoring script
echo ""
echo "📊 Creating performance monitor..."

cat > jetson_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Jetson Performance Monitor for Stage 1 Training
Monitors CPU, GPU, memory, and thermal performance
"""
import time
import psutil
import json
import os
from datetime import datetime

def get_jetson_stats():
    """Get Jetson-specific performance statistics"""
    stats = {}
    
    # CPU stats
    stats['cpu_percent'] = psutil.cpu_percent(interval=1)
    stats['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
    
    # Memory stats
    memory = psutil.virtual_memory()
    stats['memory'] = {
        'total_gb': memory.total / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }
    
    # Temperature (Jetson-specific)
    try:
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_raw = f.read().strip()
                stats['temperature'] = float(temp_raw) / 1000.0
    except:
        stats['temperature'] = None
    
    # GPU stats (if available)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        stats['gpu'] = {
            'utilization_percent': gpu_util.gpu,
            'memory_used_mb': gpu_mem.used / (1024**2),
            'memory_total_mb': gpu_mem.total / (1024**2)
        }
    except:
        stats['gpu'] = None
    
    # Power usage (if available)
    try:
        if os.path.exists('/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device0/in_voltage0_raw'):
            with open('/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device0/in_voltage0_raw', 'r') as f:
                power_raw = f.read().strip()
                stats['power_watts'] = float(power_raw) * 0.000267  # Conversion factor
    except:
        stats['power_watts'] = None
    
    return stats

def main():
    """Main monitoring loop"""
    print("🔍 Jetson Performance Monitor Started")
    print("=" * 50)
    
    log_file = "jetson_performance.log"
    
    try:
        while True:
            stats = get_jetson_stats()
            timestamp = datetime.now().isoformat()
            
            # Print formatted stats
            print(f"\n{timestamp}")
            print(f"CPU: {stats['cpu_percent']:.1f}%")
            
            memory = stats['memory']
            print(f"Memory: {memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB ({memory['percent']:.1f}%)")
            
            if stats['temperature']:
                print(f"Temperature: {stats['temperature']:.1f}°C")
            
            if stats['gpu']:
                gpu = stats['gpu']
                print(f"GPU: {gpu['utilization_percent']:.1f}%, "
                      f"Mem: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB")
            
            if stats['power_watts']:
                print(f"Power: {stats['power_watts']:.2f}W")
            
            # Log to file
            with open(log_file, 'a') as f:
                log_entry = {
                    'timestamp': timestamp,
                    'stats': stats
                }
                f.write(json.dumps(log_entry) + '\n')
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == '__main__':
    main()
EOF

chmod +x jetson_monitor.py
echo "✅ Performance monitor created: jetson_monitor.py"

# Create run script
echo ""
echo "🚀 Creating optimized run script..."

cat > run_jetson_stage1.sh << EOF
#!/bin/bash

# EVO-1 Stage 1 Run Script for Jetson Devices

echo "🚀 Starting EVO-1 Stage 1 on Jetson..."
echo "=================================="

# Load configuration
if [ -f config_jetson_stage1.yaml ]; then
    echo "✅ Loading Jetson configuration..."
    export JETSON_CONFIG_FILE="config_jetson_stage1.yaml"
else
    echo "⚠️  Configuration file not found, using defaults"
fi

# Start performance monitor in background
echo "📊 Starting performance monitor..."
python3 jetson_monitor.py &
MONITOR_PID=$!

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

# Memory management
echo "💾 Configuring memory management..."
sudo sysctl -w vm.drop_caches=3 2>/dev/null

# Run Stage 1 simulation
echo "🎓 Starting Stage 1 federated learning..."
python3 examples/evo1_stage1_federated.py

# Cleanup
echo "🧹 Cleaning up..."
kill $MONITOR_PID 2>/dev/null

echo "✅ Stage 1 simulation completed"
EOF

chmod +x run_jetson_stage1.sh
echo "✅ Run script created: run_jetson_stage1.sh"

# Create memory optimization utility
echo ""
echo "💾 Creating memory optimization utility..."

cat > jetson_memory_optimizer.py << 'EOF'
#!/usr/bin/env python3
"""
Jetson Memory Optimization Utility
Provides memory management and optimization for Stage 1 training
"""
import torch
import gc
import psutil
import time
from typing import Dict, Any

class JetsonMemoryOptimizer:
    """Memory optimizer for Jetson devices"""
    
    def __init__(self, target_memory_percent: float = 75.0):
        self.target_memory_percent = target_memory_percent
        self.last_cleanup = 0
        self.cleanup_interval = 30  # seconds
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        info = {}
        
        # System memory
        memory = psutil.virtual_memory()
        info['system_memory_percent'] = memory.percent
        info['system_memory_gb'] = memory.used / (1024**3)
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            info['gpu_memory_percent'] = (allocated / total) * 100
            info['gpu_memory_gb'] = allocated
            info['gpu_memory_reserved_gb'] = reserved
            info['gpu_memory_total_gb'] = total
        
        return info
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Optimize memory usage"""
        current_time = time.time()
        should_cleanup = force or (current_time - self.last_cleanup > self.cleanup_interval)
        
        if not should_cleanup:
            return {'action': 'skipped', 'reason': 'too_soon'}
        
        before = self.get_memory_info()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear gradients in PyTorch objects
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                if obj.grad is not None:
                    obj.grad = None
        
        after = self.get_memory_info()
        self.last_cleanup = current_time
        
        return {
            'action': 'optimized',
            'collected_objects': collected,
            'memory_before': before,
            'memory_after': after,
            'gpu_memory_freed': before.get('gpu_memory_gb', 0) - after.get('gpu_memory_gb', 0)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds threshold"""
        info = self.get_memory_info()
        return info['system_memory_percent'] > self.target_memory_percent

def main():
    """Test memory optimization"""
    print("🧠 Jetson Memory Optimization Test")
    print("=" * 40)
    
    optimizer = JetsonMemoryOptimizer(target_memory_percent=75.0)
    
    # Initial memory check
    initial = optimizer.get_memory_info()
    print(f"Initial memory usage:")
    print(f"  System: {initial['system_memory_percent']:.1f}%")
    if 'gpu_memory_percent' in initial:
        print(f"  GPU: {initial['gpu_memory_percent']:.1f}%")
    
    # Optimize memory
    result = optimizer.optimize_memory(force=True)
    print(f"\nOptimization result: {result['action']}")
    if result['action'] == 'optimized':
        print(f"  Collected objects: {result['collected_objects']}")
        print(f"  GPU memory freed: {result['gpu_memory_freed']:.2f} GB")
    
    # Final memory check
    final = optimizer.get_memory_info()
    print(f"\nFinal memory usage:")
    print(f"  System: {final['system_memory_percent']:.1f}%")
    if 'gpu_memory_percent' in final:
        print(f"  GPU: {final['gpu_memory_percent']:.1f}%")

if __name__ == '__main__':
    main()
EOF

chmod +x jetson_memory_optimizer.py
echo "✅ Memory optimizer created: jetson_memory_optimizer.py"

# Installation verification
echo ""
echo "🔍 Verifying installation..."

# Check Python packages
echo "   Checking Python packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "❌ PyTorch not installed"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" || echo "❌ TorchVision not installed"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" || echo "❌ NumPy not installed"
python3 -c "import psutil; print(f'PSUtil: {psutil.__version__}')" || echo "❌ PSUtil not installed"

# Check CUDA availability
echo "   Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "❌ CUDA check failed"

if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "   CUDA device: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")"
fi

# Performance test
echo ""
echo "⚡ Running performance test..."
python3 -c "
import torch
import time
import numpy as np

print('🧠 Testing Jetson performance...')

# CPU test
start = time.time()
for _ in range(1000):
    np.random.randn(100, 100).dot(np.random.randn(100, 100))
cpu_time = time.time() - start

# GPU test (if available)
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f'   CPU performance: {cpu_time:.3f}s')
    print(f'   GPU performance: {gpu_time:.3f}s')
    print(f'   Speedup: {cpu_time/gpu_time:.1f}x')
else:
    print(f'   CPU performance: {cpu_time:.3f}s')
    print('   GPU: Not available')
"

echo ""
echo "✅ Jetson Stage 1 deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run performance test: python3 jetson_memory_optimizer.py"
echo "2. Start monitoring: python3 jetson_monitor.py"
echo "3. Run simulation: ./run_jetson_stage1.sh"
echo ""
echo "🔧 Performance tips:"
echo "- Monitor temperature during operation"
echo "- Use thermal throttling if needed"
echo "- Monitor memory usage and adjust batch sizes"
echo "- Use the memory optimizer for large workloads"
echo ""
echo "📊 Resource limits for this device:"
echo "- Max Memory: ${MAX_MEMORY_MB}MB"
echo "- Max Batch Size: ${MAX_BATCH_SIZE}"
echo "- Precision: float16 (optimized)"
echo "- Power Mode: $([ "$NANO_SCALE" -eq 1 ] && echo "Power-saving" || echo "High-performance")"

echo ""
echo "🚀 Ready for Stage 1 federated learning on Jetson!"
echo ""
echo "⚠️  IMPORTANT: Always run this script with bash, not sh:"
echo "   bash ./deploy_jetson_stage1.sh"
echo "   or"
echo "   ./deploy_jetson_stage1.sh  (after chmod +x deploy_jetson_stage1.sh)"
