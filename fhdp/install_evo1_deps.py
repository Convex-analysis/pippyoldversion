#!/usr/bin/env python3
"""
EVO-1 Dependencies Installation Script
Handles dependency order and flash-attn compilation properly
"""
import subprocess
import sys
import os
import time
from typing import List, Dict

def run_command(cmd: List[str], description: str) -> bool:
    """Run command and handle errors"""
    print(f"🔧 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed:")
        print(f"      Error: {e.stderr}")
        return False

def check_package_installed(package_name: str) -> bool:
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_pytorch():
    """Install PyTorch with CUDA detection"""
    print("🔥 Installing PyTorch...")
    
    # Check CUDA availability
    try:
        import torch
        print(f"   ✅ PyTorch already installed: {torch.__version__}")
        return True
    except ImportError:
        pass
    
    # Detect CUDA version if available
    cuda_available = False
    try:
        import pynvml
        pynvml.nvmlInit()
        cuda_available = True
        print("   🚀 CUDA detected, installing CUDA version")
    except:
        print("   💻 CUDA not detected, installing CPU version")
    
    if cuda_available:
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    return run_command(cmd, "PyTorch installation")

def install_flash_attention():
    """Install flash-attn with proper environment"""
    print("⚡ Installing Flash Attention...")
    
    if check_package_installed("flash_attn"):
        print("   ✅ flash-attn already installed")
        return True
    
    # Set environment variables for compilation
    env = os.environ.copy()
    env['MAX_JOBS'] = '4'  # Limit parallel jobs
    env['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'  # Common GPU architectures
    
    # Try different installation methods
    install_methods = [
        # Method 1: Direct pip install
        [
            sys.executable, "-m", "pip", "install", 
            "flash-attn", "--no-build-isolation"
        ],
        # Method 2: Without isolation
        [
            sys.executable, "-m", "pip", "install", 
            "flash-attn>=2.0.0", "--no-cache-dir"
        ],
        # Method 3: From source (fallback)
        [
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/Dao-AILab/flash-attention.git"
        ]
    ]
    
    for i, cmd in enumerate(install_methods):
        print(f"   🔄 Trying installation method {i+1}/3...")
        
        if run_command(cmd, f"Flash-attn installation (method {i+1})"):
            return True
        
        print(f"   ⚠️  Method {i+1} failed, trying next...")
    
    print("   ⚠️  All flash-attn installation methods failed")
    print("   💡 You can continue without flash-attn (will use slower attention)")
    return False

def install_requirements_file(file_path: str, stage_name: str):
    """Install requirements from a specific file"""
    print(f"📦 Installing {stage_name} dependencies...")
    
    if not os.path.exists(file_path):
        print(f"   ⚠️  Requirements file {file_path} not found")
        return True  # Not critical
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", file_path]
    return run_command(cmd, f"{stage_name} dependencies installation")

def main():
    """Main installation function"""
    print("🚀 EVO-1 Dependencies Installation")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Python version: {python_version}")
    
    if sys.version_info < (3, 8) or sys.version_info > (3, 10):
        print("⚠️  Warning: Python 3.8-3.10 recommended")
    
    # Installation stages
    stages = [
        ("Stage 1: Core Dependencies", "requirements_evo1_stages.txt", install_pytorch),
        ("Stage 2: ML and Advanced Features", "requirements_evo1_stage2.txt", None),
        ("Stage 3: Optional and Specialized", "requirements_evo1_stage3.txt", None),
    ]
    
    success_count = 0
    total_stages = len(stages)
    
    for stage_name, req_file, special_install in stages:
        print(f"\n{'='*20} {stage_name} {'='*20}")
        
        stage_success = True
        
        # Special installation for some stages
        if special_install:
            stage_success = special_install()
        
        # Install from requirements file
        if stage_success:
            stage_success = install_requirements_file(req_file, stage_name)
        
        # Flash-attn special handling (install after PyTorch)
        if stage_name == "Stage 1: Core Dependencies" and stage_success:
            flash_success = install_flash_attention()
            # Don't fail the entire stage if flash-attn fails
        
        if stage_success:
            success_count += 1
            print(f"✅ {stage_name} completed")
        else:
            print(f"❌ {stage_name} failed")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Installation Summary")
    print(f"{'='*50}")
    print(f"Completed stages: {success_count}/{total_stages}")
    
    if success_count == total_stages:
        print("🎉 All dependencies installed successfully!")
        
        # Verification
        print("\n🔍 Verifying installation...")
        critical_packages = [
            "torch", "torchvision", "numpy", "pandas", 
            "psutil", "opencv-python", "matplotlib"
        ]
        
        for package in critical_packages:
            try:
                if package == "opencv-python":
                    __import__("cv2")
                elif package == "psutil":
                    __import__("psutil")
                else:
                    __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package}")
        
        # Flash-attn check (optional)
        try:
            import flash_attn
            print(f"   ✅ flash-attn (optional)")
        except ImportError:
            print(f"   ⚠️  flash-attn (optional, not installed)")
        
        print(f"\n🚀 Installation completed!")
        print(f"You can now run:")
        print(f"   python examples/autonomous_driving_simulation.py")
        print(f"   python examples/evo1_stage1_federated.py")
    
    # Install FHDP System from source
    print("\n📦 Installing FHDP (Federated Highway-based Distributed Pipeline)...")
    try:
        import sys
        import subprocess
        # Install FHDP in editable mode with communication extras
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".[communication]"
        ], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ FHDP installed successfully")
        else:
            print(f"   ⚠️  FHDP installation failed: {result.stderr}")
            print("   💡 You can try installing it manually with: pip install -e .[communication]")
    except Exception as e:
        print(f"   ⚠️  FHDP installation error: {e}")

    if success_count != total_stages:
        print(f"⚠️  {total_stages - success_count} stage(s) failed")
        print(f"Please check the errors above and try again")
        
        # Troubleshooting tips
        print(f"\n💡 Troubleshooting:")
        print(f"1. Ensure you have sufficient disk space (5GB+)")
        print(f"2. Check internet connection")
        print(f"3. Try installing stages individually:")
        for stage_name, req_file, _ in stages:
            print(f"   pip install -r {req_file}")
        print(f"4. For flash-attn issues:")
        print(f"   - Set MAX_JOBS=2 (reduce parallel jobs)")
        print(f"   - Install torch first, then flash-attn")
        print(f"   - Use CPU-only version if GPU issues persist")

if __name__ == '__main__':
    main()