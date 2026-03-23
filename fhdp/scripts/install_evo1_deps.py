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

def run_command(cmd: List[str], description: str, capture_output: bool = True) -> bool:
    """Run command and handle errors"""
    print(f"🔧 {description}...")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True,
            capture_output=capture_output,
            text=True
        )
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed:")
        if e.stderr:
            print(f"      Error: {e.stderr}")
        else:
            print(f"      Return code: {e.returncode}")
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

    # Check if running on Jetson device
    is_jetson = False
    jetpack_version = ""

    # Check multiple possible locations for JetPack version
    for release_file in ['/etc/nv_tegra_release', '/etc/nv_tegra-release']:
        if os.path.exists(release_file):
            is_jetson = True
            print("   📱 Jetson device detected")

            # Get JetPack version
            try:
                with open(release_file, 'r') as f:
                    content = f.read()
                    import re
                    # Try multiple patterns
                    match = re.search(r'R(\d+)\.(\d+)', content)
                    if match:
                        jetpack_version = f"R{match.group(1)}.{match.group(2)}"
                    else:
                        # Try simpler pattern (e.g., R36)
                        match = re.search(r'R(\d+)', content)
                        if match:
                            jetpack_version = f"R{match.group(1)}"
            except:
                pass
            break  # Found first release file, stop checking

    print(f"   JetPack version: {jetpack_version or 'unknown'}")

    if is_jetson:
        # Determine PyTorch URL based on JetPack version
        pytorch_url = ""
        if jetpack_version.startswith('R36'):
            # JetPack 6.0
            pytorch_url = "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"
            print("   📌 Using JetPack 6.0 PyTorch repository")
        elif jetpack_version.startswith('R35'):
            # JetPack 5.x
            pytorch_url = "https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch"
            print("   📌 Using JetPack 5.x PyTorch repository")
        elif jetpack_version.startswith('R34'):
            # JetPack 4.x
            pytorch_url = "https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch"
            print("   📌 Using JetPack 4.x PyTorch repository")
        else:
            # Unknown JetPack version - try to detect from CUDA
            print("   ⚠️  Unknown JetPack version, trying to detect from CUDA...")
            print("   💡 If this fails, run: bash ./jetson_diag.sh")

            # Try different versions starting with newest
            pytorch_urls = [
                ("JetPack 6.0", "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch"),
                ("JetPack 5.05", "https://developer.download.nvidia.com/compute/redist/jp/v505/pytorch"),
                ("JetPack 5.04", "https://developer.download.nvidia.com/compute/redist/jp/v504/pytorch"),
                ("JetPack 4.6", "https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch"),
            ]

            # Use the first URL (JetPack 6.0) as default
            pytorch_url = pytorch_urls[0][1]
            print(f"   📌 Defaulting to JetPack 6.0 URL: {pytorch_url}")
            print(f"   💡 If installation fails, try manually with:")
            for name, url in pytorch_urls[1:]:
                print(f"      pip3 install torch torchvision torchaudio --index-url {url}")

        print(f"   Installing from: {pytorch_url}")
        print(f"   ⏳ This may take 10-30 minutes...")

        # Upgrade pip first
        pip_upgrade_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        if not run_command(pip_upgrade_cmd, "Pip upgrade", capture_output=False):
            print("   ⚠️  Pip upgrade failed, continuing...")

        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", pytorch_url
        ]

        # Don't capture output for PyTorch to see progress
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    else:
        # Regular CUDA detection
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

def install_requirements_file(file_path: str, stage_name: str, skip_upgrade: bool = False):
    """Install requirements from a specific file

    Args:
        file_path: Path to requirements file
        stage_name: Name of the installation stage
        skip_upgrade: If True, use --no-deps and --no-upgrade to prevent PyTorch overwrites
    """
    print(f"📦 Installing {stage_name} dependencies...")

    # Convert relative path to absolute path from fhdp/requirements
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(__file__), "..", "fhdp", "requirements", file_path)

    if not os.path.exists(file_path):
        print(f"   ⚠️  Requirements file {file_path} not found")
        return True  # Not critical

    cmd = [sys.executable, "-m", "pip", "install", "-r", file_path]

    # Prevent PyTorch overwrites on Jetson
    if skip_upgrade:
        cmd.extend(["--no-deps"])
        print("   🛡️  Using --no-deps to prevent PyTorch version conflicts")
        print("   💡 This may require additional manual dependency resolution")

    return run_command(cmd, f"{stage_name} dependencies installation")

def main():
    """Main installation function"""
    print("🚀 EVO-1 Dependencies Installation")
    print("=" * 50)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Python version: {python_version}")

    if sys.version_info < (3, 8) or sys.version_info > (3, 11):
        print("⚠️  Warning: Python 3.8-3.11 recommended")

    # Detect if running on Jetson
    is_jetson = os.path.exists('/etc/nv_tegra_release')
    if is_jetson:
        print("📱 Jetson device detected")
    else:
        print("💻 Standard system detected")

    # Determine which requirements files to use
    if is_jetson:
        # For Jetson, use jetson-optimized requirements
        stages = [
            ("Stage 1: Core Dependencies", "requirements/jetson.txt", install_pytorch),
            ("Stage 2: EVO-1 Model", "requirements/evo1.txt", None, True),  # skip_upgrade=True
            ("Stage 3: Communication", "requirements/ml.txt", None, True),  # skip_upgrade=True
        ]
    else:
        # For standard systems, use modular requirements
        stages = [
            ("Stage 1: Base Dependencies", "requirements/base.txt", install_pytorch),
            ("Stage 2: ML Features", "requirements/ml.txt", None),
            ("Stage 3: EVO-1 Model", "requirements/evo1.txt", None),
        ]

    success_count = 0
    total_stages = len(stages)

    for stage_name, req_file, special_install, skip_upgrade in stages:
        print(f"\n{'='*20} {stage_name} {'='*20}")

        stage_success = True

        # Special installation for some stages
        if special_install:
            stage_success = special_install()

        # Install from requirements file
        if stage_success:
            stage_success = install_requirements_file(req_file, stage_name, skip_upgrade=skip_upgrade)

        # Flash-attn special handling (install after PyTorch)
        if stage_name.startswith("Stage 1") and stage_success:
            # Only try flash-attn on non-Jetson systems
            if not is_jetson:
                flash_success = install_flash_attention()
                # Don't fail the entire stage if flash-attn fails
            else:
                print("   ⏭️  Skipping flash-attn on Jetson (uses optimized attention)")

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