#!/usr/bin/env python3
"""
FHDP Heterogeneous Platform Deployment Script

Deploys FHDP across heterogeneous platforms including Jetson Orin Nano,
x86 PCs, and other computing devices.
"""
import os
import sys
import argparse
import yaml
import json
import subprocess
import socket
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import platform as platform_module

# Add FHDP to path
sys.path.append(str(Path(__file__).parent.parent))

from core.hardware_adapter import HardwareDetector, HardwarePlatform
from core.heterogeneous_resource import AdaptiveResourceMonitor
from core.cross_platform_comm import PlatformBridge

class DeploymentManager:
    """Manages deployment across heterogeneous platforms"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.detector = HardwareDetector()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def detect_platform(self) -> HardwarePlatform:
        """Detect current hardware platform"""
        platform = self.detector.detect_platform()
        self.logger.info(f"Detected platform: {platform.value}")
        return platform
    
    def get_platform_config(self, platform: HardwarePlatform) -> Dict:
        """Get configuration for specific platform"""
        platform_map = {
            HardwarePlatform.JETSON_ORIN: "jetson_orin_nano",
            HardwarePlatform.JETSON_NANO: "jetson_orin_nano",  # Use same config
            HardwarePlatform.JETSON_XAVIER: "jetson_xavier",
            HardwarePlatform.X86_LINUX: "x86_linux",
            HardwarePlatform.X86_WINDOWS: "x86_windows",
            HardwarePlatform.X86_MACOS: "x86_linux",  # Use Linux config for macOS
        }
        
        config_key = platform_map.get(platform)
        if not config_key:
            self.logger.warning(f"No specific config for {platform.value}, using default")
            config_key = "x86_linux"
        
        return self.config["platforms"].get(config_key, {})
    
    def setup_environment(self, platform: HardwarePlatform) -> bool:
        """Setup environment for the platform"""
        self.logger.info(f"Setting up environment for {platform.value}")
        
        if platform in [HardwarePlatform.JETSON_ORIN, HardwarePlatform.JETSON_NANO, HardwarePlatform.JETSON_XAVIER]:
            return self._setup_jetson_environment(platform)
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_MACOS]:
            return self._setup_linux_environment(platform)
        elif platform == HardwarePlatform.X86_WINDOWS:
            return self._setup_windows_environment(platform)
        else:
            self.logger.warning(f"Unknown platform: {platform.value}")
            return False
    
    def _setup_jetson_environment(self, platform: HardwarePlatform) -> bool:
        """Setup Jetson environment"""
        try:
            self.logger.info("Setting up Jetson environment...")
            
            # Check Jetson-specific requirements
            if not self._check_jetson_requirements():
                return False
            
            # Setup power mode
            self._setup_jetson_power_mode(platform)
            
            # Setup GPU environment
            self._setup_jetson_gpu_environment()
            
            # Create necessary directories
            self._create_directories()
            
            self.logger.info("Jetson environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Jetson setup failed: {e}")
            return False
    
    def _check_jetson_requirements(self) -> bool:
        """Check Jetson-specific requirements"""
        try:
            # Check for JetPack
            result = subprocess.run(['cat', '/etc/nv_tegra_release'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("JetPack not detected")
                return False
            
            self.logger.info(f"JetPack: {result.stdout.strip()}")
            
            # Check for CUDA
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("CUDA not detected")
                return False
            
            self.logger.info("CUDA detected")
            
            # Check for sufficient storage
            disk_usage = shutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 5:  # Require at least 5GB free
                self.logger.error(f"Insufficient storage: {free_gb:.1f}GB free, need 5GB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Jetson requirements check failed: {e}")
            return False
    
    def _setup_jetson_power_mode(self, platform: HardwarePlatform):
        """Setup Jetson power mode"""
        try:
            if platform == HardwarePlatform.JETSON_ORIN:
                # Set to max performance mode
                subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=False)
                subprocess.run(['sudo', 'jetson_clocks'], check=False)
                self.logger.info("Set Jetson to maximum performance mode")
            elif platform == HardwarePlatform.JETSON_NANO:
                # Set to 10W mode for balance
                subprocess.run(['sudo', 'nvpmodel', '-m', '1'], check=False)
                self.logger.info("Set Jetson Nano to 10W mode")
        except Exception as e:
            self.logger.warning(f"Failed to set power mode: {e}")
    
    def _setup_jetson_gpu_environment(self):
        """Setup Jetson GPU environment"""
        try:
            # Set CUDA environment variables
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Check GPU memory
            result = subprocess.run(['tegrastats', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("GPU monitoring available")
            else:
                self.logger.warning("GPU monitoring not available")
                
        except Exception as e:
            self.logger.warning(f"GPU setup failed: {e}")
    
    def _setup_linux_environment(self, platform: HardwarePlatform) -> bool:
        """Setup Linux environment"""
        try:
            self.logger.info("Setting up Linux environment...")
            
            # Check system requirements
            if not self._check_linux_requirements():
                return False
            
            # Setup GPU if available
            self._setup_linux_gpu()
            
            # Create directories
            self._create_directories()
            
            self.logger.info("Linux environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Linux setup failed: {e}")
            return False
    
    def _check_linux_requirements(self) -> bool:
        """Check Linux requirements"""
        try:
            # Check Python version
            if sys.version_info < (3, 7):
                self.logger.error("Python 3.7+ required")
                return False
            
            # Check required packages
            required_packages = ['numpy', 'psutil', 'pyyaml']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    self.logger.error(f"Required package missing: {package}")
                    return False
            
            # Check storage
            disk_usage = shutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:  # Require at least 10GB free
                self.logger.error(f"Insufficient storage: {free_gb:.1f}GB free, need 10GB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Linux requirements check failed: {e}")
            return False
    
    def _setup_linux_gpu(self):
        """Setup GPU on Linux"""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("NVIDIA GPU detected")
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            else:
                self.logger.info("No NVIDIA GPU detected, using CPU")
                
        except Exception as e:
            self.logger.warning(f"GPU setup failed: {e}")
    
    def _setup_windows_environment(self, platform: HardwarePlatform) -> bool:
        """Setup Windows environment"""
        try:
            self.logger.info("Setting up Windows environment...")
            
            # Check requirements
            if not self._check_windows_requirements():
                return False
            
            # Setup GPU if available
            self._setup_windows_gpu()
            
            # Create directories
            self._create_directories()
            
            self.logger.info("Windows environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Windows setup failed: {e}")
            return False
    
    def _check_windows_requirements(self) -> bool:
        """Check Windows requirements"""
        try:
            # Check Python version
            if sys.version_info < (3, 7):
                self.logger.error("Python 3.7+ required")
                return False
            
            # Check storage
            disk_usage = shutil.disk_usage('C:')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 15:  # Require at least 15GB free on Windows
                self.logger.error(f"Insufficient storage: {free_gb:.1f}GB free, need 15GB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Windows requirements check failed: {e}")
            return False
    
    def _setup_windows_gpu(self):
        """Setup GPU on Windows"""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi.exe'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("NVIDIA GPU detected")
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            else:
                self.logger.info("No NVIDIA GPU detected, using CPU")
                
        except Exception as e:
            self.logger.warning(f"GPU setup failed: {e}")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            '/var/log/fhdp',
            '/var/lib/fhdp',
            '/var/backups/fhdp',
            '/etc/fhdp/certs',
            '/tmp/fhdp'
        ]
        
        for directory in directories:
            try:
                if platform_module.system() == "Windows":
                    # Convert Unix paths to Windows
                    directory = directory.replace('/var/log/', 'C:\\ProgramData\\fhdp\\logs\\')
                    directory = directory.replace('/var/lib/', 'C:\\ProgramData\\fhdp\\lib\\')
                    directory = directory.replace('/var/backups/', 'C:\\ProgramData\\fhdp\\backups\\')
                    directory = directory.replace('/etc/fhdp/', 'C:\\ProgramData\\fhdp\\etc\\')
                    directory = directory.replace('/tmp/', 'C:\\temp\\')
                
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            except Exception as e:
                self.logger.warning(f"Failed to create directory {directory}: {e}")
    
    def install_dependencies(self, platform: HardwarePlatform) -> bool:
        """Install platform-specific dependencies"""
        self.logger.info(f"Installing dependencies for {platform.value}")
        
        try:
            if platform in [HardwarePlatform.JETSON_ORIN, HardwarePlatform.JETSON_NANO, HardwarePlatform.JETSON_XAVIER]:
                return self._install_jetson_dependencies()
            elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_MACOS]:
                return self._install_linux_dependencies()
            elif platform == HardwarePlatform.X86_WINDOWS:
                return self._install_windows_dependencies()
            else:
                self.logger.warning(f"No specific dependencies for {platform.value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _install_jetson_dependencies(self) -> bool:
        """Install Jetson-specific dependencies"""
        try:
            # Jetson packages are usually pre-installed with JetPack
            required_python_packages = [
                'numpy>=1.19.0',
                'psutil>=5.8.0',
                'pyyaml>=5.4.0',
                'torch>=1.9.0',
                'torchvision>=0.10.0',
                'opencv-python>=4.5.0'
            ]
            
            for package in required_python_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True)
                    self.logger.info(f"Installed: {package}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to install {package}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Jetson dependency installation failed: {e}")
            return False
    
    def _install_linux_dependencies(self) -> bool:
        """Install Linux-specific dependencies"""
        try:
            # System packages (Ubuntu/Debian)
            if platform_module.system() == "Linux":
                try:
                    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                    system_packages = [
                        'python3-dev',
                        'python3-pip',
                        'build-essential',
                        'cmake',
                        'pkg-config'
                    ]
                    
                    for package in system_packages:
                        try:
                            subprocess.run(['sudo', 'apt-get', 'install', '-y', package], 
                                         check=True)
                            self.logger.info(f"Installed: {package}")
                        except subprocess.CalledProcessError as e:
                            self.logger.warning(f"Failed to install {package}: {e}")
                
                except Exception as e:
                    self.logger.warning(f"System package installation failed: {e}")
            
            # Python packages
            required_python_packages = [
                'numpy>=1.19.0',
                'psutil>=5.8.0',
                'pyyaml>=5.4.0',
                'torch>=1.9.0',
                'torchvision>=0.10.0',
                'opencv-python>=4.5.0',
                'scipy>=1.7.0',
                'scikit-learn>=1.0.0'
            ]
            
            for package in required_python_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True)
                    self.logger.info(f"Installed: {package}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to install {package}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Linux dependency installation failed: {e}")
            return False
    
    def _install_windows_dependencies(self) -> bool:
        """Install Windows-specific dependencies"""
        try:
            # Python packages for Windows
            required_python_packages = [
                'numpy>=1.19.0',
                'psutil>=5.8.0',
                'pyyaml>=5.4.0',
                'torch>=1.9.0',
                'torchvision>=0.10.0',
                'opencv-python>=4.5.0',
                'scipy>=1.7.0',
                'scikit-learn>=1.0.0',
                'pywin32>=227'  # For Windows system monitoring
            ]
            
            for package in required_python_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True)
                    self.logger.info(f"Installed: {package}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to install {package}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Windows dependency installation failed: {e}")
            return False
    
    def generate_startup_script(self, platform: HardwarePlatform) -> str:
        """Generate platform-specific startup script"""
        script_content = ""
        
        if platform in [HardwarePlatform.JETSON_ORIN, HardwarePlatform.JETSON_NANO, HardwarePlatform.JETSON_XAVIER]:
            script_content = self._generate_jetson_startup_script()
        elif platform in [HardwarePlatform.X86_LINUX, HardwarePlatform.X86_MACOS]:
            script_content = self._generate_linux_startup_script()
        elif platform == HardwarePlatform.X86_WINDOWS:
            script_content = self._generate_windows_startup_script()
        
        return script_content
    
    def _generate_jetson_startup_script(self) -> str:
        """Generate Jetson startup script"""
        return '''#!/bin/bash
# FHDP Jetson Startup Script

# Set maximum performance mode
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set FHDP environment
export FHDP_CONFIG_PATH="/etc/fhdp/heterogeneous_config.yaml"
export FHDP_LOG_PATH="/var/log/fhdp"
export FHDP_DATA_PATH="/var/lib/fhdp"

# Start FHDP
cd /opt/fhdp
python -m fhdp --platform jetson --config "$FHDP_CONFIG_PATH"
'''
    
    def _generate_linux_startup_script(self) -> str:
        """Generate Linux startup script"""
        return '''#!/bin/bash
# FHDP Linux Startup Script

# Set CUDA environment if available
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

# Set FHDP environment
export FHDP_CONFIG_PATH="/etc/fhdp/heterogeneous_config.yaml"
export FHDP_LOG_PATH="/var/log/fhdp"
export FHDP_DATA_PATH="/var/lib/fhdp"

# Start FHDP
cd /opt/fhdp
python -m fhdp --platform x86 --config "$FHDP_CONFIG_PATH"
'''
    
    def _generate_windows_startup_script(self) -> str:
        """Generate Windows startup script"""
        return '''@echo off
REM FHDP Windows Startup Script

REM Set CUDA environment if available
nvidia-smi.exe >nul 2>&1
if %ERRORLEVEL% == 0 (
    set CUDA_VISIBLE_DEVICES=0
    set CUDA_DEVICE_ORDER=PCI_BUS_ID
)

REM Set FHDP environment
set FHDP_CONFIG_PATH=C:\\ProgramData\\fhdp\\etc\\heterogeneous_config.yaml
set FHDP_LOG_PATH=C:\\ProgramData\\fhdp\\logs
set FHDP_DATA_PATH=C:\\ProgramData\\fhdp\\lib

REM Start FHDP
cd /d C:\\Program Files\\FHDP
python -m fhdp --platform windows --config "%FHDP_CONFIG_PATH%"
'''
    
    def deploy(self, platform: Optional[HardwarePlatform] = None) -> bool:
        """Deploy FHDP on detected or specified platform"""
        if platform is None:
            platform = self.detect_platform()
        
        self.logger.info(f"Deploying FHDP on {platform.value}")
        
        # Get platform configuration
        platform_config = self.get_platform_config(platform)
        self.logger.info(f"Using platform config: {platform_config.get('hardware', {}).get('platform', 'unknown')}")
        
        # Setup environment
        if not self.setup_environment(platform):
            self.logger.error("Environment setup failed")
            return False
        
        # Install dependencies
        if not self.install_dependencies(platform):
            self.logger.error("Dependency installation failed")
            return False
        
        # Generate startup script
        startup_script = self.generate_startup_script(platform)
        script_path = f"/tmp/fhdp_startup.sh" if platform != HardwarePlatform.X86_WINDOWS else "C:\\temp\\fhdp_startup.bat"
        
        try:
            with open(script_path, 'w') as f:
                f.write(startup_script)
            
            # Make script executable on Unix systems
            if platform != HardwarePlatform.X86_WINDOWS:
                os.chmod(script_path, 0o755)
            
            self.logger.info(f"Generated startup script: {script_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate startup script: {e}")
            return False
        
        self.logger.info(f"FHDP deployment complete for {platform.value}")
        self.logger.info(f"To start FHDP, run: {script_path}")
        
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='FHDP Heterogeneous Platform Deployment')
    parser.add_argument('--config', '-c', required=True,
                       help='Path to heterogeneous configuration file')
    parser.add_argument('--platform', '-p', choices=['jetson', 'x86', 'windows', 'auto'], default='auto',
                       help='Target platform (auto-detect if not specified)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run deployment without making changes')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Create deployment manager
    deployer = DeploymentManager(args.config)
    
    # Determine target platform
    if args.platform == 'auto':
        platform = deployer.detect_platform()
    else:
        platform_map = {
            'jetson': HardwarePlatform.JETSON_ORIN,
            'x86': HardwarePlatform.X86_LINUX,
            'windows': HardwarePlatform.X86_WINDOWS
        }
        platform = platform_map.get(args.platform)
        if not platform:
            print(f"Error: Unknown platform: {args.platform}")
            sys.exit(1)
    
    if args.dry_run:
        print(f"Dry run: Would deploy on {platform.value}")
        platform_config = deployer.get_platform_config(platform)
        print(f"Platform config: {json.dumps(platform_config, indent=2)}")
        return
    
    # Deploy
    success = deployer.deploy(platform)
    
    if success:
        print(f"✅ FHDP successfully deployed on {platform.value}")
        sys.exit(0)
    else:
        print(f"❌ FHDP deployment failed on {platform.value}")
        sys.exit(1)

if __name__ == "__main__":
    main()