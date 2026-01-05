#!/usr/bin/env python3
"""
Test the installation fix for flash-attn issue
"""
import subprocess
import sys

def test_pytorch_installation():
    """Test PyTorch installation"""
    print("🔍 Testing PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_flash_attn_installation():
    """Test flash-attn installation"""
    print("\n⚡ Testing flash-attn installation...")
    try:
        import flash_attn
        print("✅ flash-attn installed successfully")
        return True
    except ImportError:
        print("⚠️  flash-attn not installed (optional)")
        return False

def test_core_dependencies():
    """Test core dependencies"""
    print("\n📦 Testing core dependencies...")
    
    core_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('psutil', 'PSUtil'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'TQDM')
    ]
    
    success_count = 0
    for module_name, display_name in core_packages:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
            success_count += 1
        except ImportError:
            print(f"   ❌ {display_name}")
    
    print(f"\n   Core dependencies: {success_count}/{len(core_packages)} installed")
    return success_count >= len(core_packages) * 0.8  # 80% pass rate

def main():
    """Main test function"""
    print("🧪 EVO-1 Installation Fix Test")
    print("=" * 40)
    
    # Test installations
    pytorch_ok = test_pytorch_installation()
    flash_attn_ok = test_flash_attn_installation()
    core_ok = test_core_dependencies()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Installation Summary")
    print("=" * 40)
    
    print(f"PyTorch: {'✅' if pytorch_ok else '❌'}")
    print(f"flash-attn: {'✅' if flash_attn_ok else '⚠️  (optional)'}")
    print(f"Core dependencies: {'✅' if core_ok else '❌'}")
    
    # Recommendations
    if pytorch_ok and core_ok:
        print(f"\n🎉 Installation successful!")
        print(f"   You can run EVO-1 simulations:")
        print(f"   - python examples/autonomous_driving_simulation.py")
        print(f"   - python examples/evo1_stage1_federated.py")
        
        if not flash_attn_ok:
            print(f"\n💡 Note:")
            print(f"   - flash-attn not installed (optional)")
            print(f"   - Will use standard attention (slower but functional)")
    else:
        print(f"\n⚠️  Installation incomplete:")
        
        if not pytorch_ok:
            print(f"   ❌ PyTorch missing - install first:")
            print(f"      pip install torch torchvision torchaudio")
        
        if not core_ok:
            print(f"   ❌ Core dependencies missing:")
            print(f"      pip install numpy pandas matplotlib tqdm psutil opencv-python-headless pyyaml")
    
    print(f"\n🔧 Quick fix commands:")
    print(f"   # Install PyTorch:")
    print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print(f"")
    print(f"   # Install core dependencies:")
    print(f"   pip install -r requirements_jetson_stage1.txt")
    print(f"")
    print(f"   # Try flash-attn (optional):")
    print(f"   export MAX_JOBS=2")
    print(f"   pip install flash-attn --no-build-isolation")

if __name__ == '__main__':
    main()