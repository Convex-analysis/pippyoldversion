#!/usr/bin/env python3
"""
Intelligent FHDP Requirements Installer
Automatically detects platform and installs appropriate dependencies
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class FHDPInstaller:
    def __init__(self):
        self.requirements_dir = Path(__file__).parent.parent / "requirements"
        self.system_info = self._detect_system()
        
    def _detect_system(self) -> Dict[str, str]:
        """Detect system information for platform-specific installation"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'is_jetson': self._is_jetson(),
            'cuda_available': self._check_cuda()
        }
        
        # Check for Jetson-specific files
        if info['is_jetson']:
            info['platform'] = 'jetson'
            
        return info
    
    def _is_jetson(self) -> bool:
        """Check if running on Jetson device"""
        try:
            # Check for Jetson-specific files
            jetson_files = [
                '/etc/nv_tegra_release',
                '/sys/module/tegra_fuse',
                '/proc/device-tree/compatible'
            ]
            
            for file_path in jetson_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ['tegra', 'jetson', 'nvidia']):
                            return True
                            
            # Check for jetson-tools
            try:
                subprocess.run(['jetson_clocks'], capture_output=True, check=False)
                return True
            except FileNotFoundError:
                pass
                
        except Exception:
            pass
            
        return False
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _get_recommended_requirements(self) -> List[str]:
        """Get recommended requirement files based on system"""
        recommendations = []
        
        if self.system_info['platform'] == 'jetson':
            recommendations.extend(['jetson.txt'])
        else:
            recommendations.extend(['base.txt'])
            
            # Check if ML components are needed
            if self.system_info['cuda_available']:
                recommendations.append('ml.txt')
                
            # Add EVO-1 if user wants autonomous driving
            recommendations.append('evo1.txt')
            
        return recommendations
    
    def _install_requirements_file(self, req_file: str, upgrade: bool = False) -> bool:
        """Install a single requirements file"""
        req_path = self.requirements_dir / req_file
        
        if not req_path.exists():
            print(f"❌ Requirements file not found: {req_path}")
            return False
            
        print(f"📦 Installing {req_file}...")
        
        cmd = ['pip', 'install']
        if upgrade:
            cmd.append('--upgrade')
        cmd.extend(['-r', str(req_path)])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully installed {req_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req_file}: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error output: {e.stderr}")
            return False
    
    def install_complete(self, upgrade: bool = False) -> bool:
        """Install complete FHDP system"""
        return self._install_requirements_file('complete.txt', upgrade)
    
    def install_minimal(self, upgrade: bool = False) -> bool:
        """Install minimal FHDP system"""
        return self._install_requirements_file('minimal.txt', upgrade)
    
    def install_platform_specific(self, upgrade: bool = False) -> bool:
        """Install platform-specific requirements"""
        if self.system_info['platform'] == 'jetson':
            return self._install_requirements_file('jetson.txt', upgrade)
        else:
            # Install base + ML + EVO-1 for standard systems
            success = True
            for req_file in ['base.txt', 'ml.txt', 'evo1.txt']:
                success &= self._install_requirements_file(req_file, upgrade)
            return success
    
    def install_recommended(self, upgrade: bool = False) -> bool:
        """Install recommended requirements based on system"""
        recommendations = self._get_recommended_requirements()
        
        print(f"🎯 Installing recommended requirements: {', '.join(recommendations)}")
        
        success = True
        for req_file in recommendations:
            success &= self._install_requirements_file(req_file, upgrade)
            
        return success
    
    def install_custom(self, req_files: List[str], upgrade: bool = False) -> bool:
        """Install custom requirement files"""
        success = True
        for req_file in req_files:
            # Auto-add .txt extension if not present
            if not req_file.endswith('.txt'):
                req_file += '.txt'
            success &= self._install_requirements_file(req_file, upgrade)
        return success
    
    def list_available(self):
        """List all available requirement files"""
        print("📋 Available requirement files:")
        
        for req_file in sorted(self.requirements_dir.glob('*.txt')):
            size_kb = req_file.stat().st_size / 1024
            print(f"  📄 {req_file.name} ({size_kb:.1f}KB)")
    
    def show_system_info(self):
        """Display system information"""
        print("🖥️  System Information:")
        print(f"  Platform: {self.system_info['platform']}")
        print(f"  Machine: {self.system_info['machine']}")
        print(f"  Python: {self.system_info['python_version']}")
        print(f"  Jetson Device: {'Yes' if self.system_info['is_jetson'] else 'No'}")
        print(f"  CUDA Available: {'Yes' if self.system_info['cuda_available'] else 'No'}")
        
        recommendations = self._get_recommended_requirements()
        print(f"  Recommended: {', '.join(recommendations)}")

def main():
    parser = argparse.ArgumentParser(description='FHDP Requirements Installer')
    parser.add_argument('--minimal', action='store_true', 
                       help='Install minimal requirements')
    parser.add_argument('--complete', action='store_true', 
                       help='Install complete requirements')
    parser.add_argument('--platform-specific', action='store_true', 
                       help='Install platform-specific requirements')
    parser.add_argument('--recommended', action='store_true', 
                       help='Install recommended requirements')
    parser.add_argument('--list', action='store_true', 
                       help='List available requirement files')
    parser.add_argument('--info', action='store_true', 
                       help='Show system information')
    parser.add_argument('--custom', nargs='+', 
                       help='Install custom requirement files')
    parser.add_argument('--upgrade', action='store_true', 
                       help='Upgrade existing packages')
    parser.add_argument('--detect-platform', action='store_true', 
                       help='Detect platform and install appropriate requirements')
    
    args = parser.parse_args()
    
    installer = FHDPInstaller()
    
    if args.info:
        installer.show_system_info()
        return
    
    if args.list:
        installer.list_available()
        return
    
    if args.detect_platform:
        installer.show_system_info()
        success = installer.install_platform_specific(args.upgrade)
        sys.exit(0 if success else 1)
    
    # Installation commands
    if args.minimal:
        success = installer.install_minimal(args.upgrade)
    elif args.complete:
        success = installer.install_complete(args.upgrade)
    elif args.platform_specific:
        success = installer.install_platform_specific(args.upgrade)
    elif args.recommended:
        success = installer.install_recommended(args.upgrade)
    elif args.custom:
        success = installer.install_custom(args.custom, args.upgrade)
    else:
        # Default: show help
        parser.print_help()
        print("\n🎯 Quick start recommendations:")
        print("  python scripts/install_requirements.py --detect-platform")
        print("  python scripts/install_requirements.py --minimal")
        print("  python scripts/install_requirements.py --complete")
        return
    
    if success:
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n❌ Installation failed. Check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()