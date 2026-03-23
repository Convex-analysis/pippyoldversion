#!/usr/bin/env python3
"""
Pipeline Test Validation Script

This script validates that the FHDP pipeline training test environment
is properly configured before running the actual test.

Run this on each machine before starting the pipeline test.
"""

import sys
import os
import platform
import importlib
import socket
from typing import Tuple, List

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro}"

def check_module(module_name: str) -> Tuple[bool, str]:
    """Check if a Python module is installed"""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, f"{module_name} {version}"
    except ImportError:
        return False, f"{module_name} not installed"

def check_pytorch() -> Tuple[bool, List[Tuple[bool, str]]]:
    """Check PyTorch installation and CUDA availability"""
    results = []

    # Check if PyTorch is installed
    try:
        import torch
        version = torch.__version__
        results.append((True, f"PyTorch {version}"))

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            results.append((True, f"CUDA {cuda_version} available ({device_count} device(s))"))
            results.append((True, f"Primary GPU: {device_name}"))
        else:
            results.append((False, "CUDA not available"))

    except ImportError:
        results.append((False, "PyTorch not installed"))

    all_passed = all(passed for passed, _ in results)
    return all_passed, results

def check_network_port(port: int) -> Tuple[bool, str]:
    """Check if a network port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True, f"Port {port} is available"
    except OSError:
        return False, f"Port {port} is already in use"

def check_disk_space(min_gb: int = 5) -> Tuple[bool, str]:
    """Check if there's sufficient disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb >= min_gb:
            return True, f"{free_gb} GB free disk space"
        else:
            return False, f"Only {free_gb} GB free (need {min_gb} GB)"
    except Exception:
        return True, "Disk space check skipped"

def check_memory(min_gb: int = 4) -> Tuple[bool, str]:
    """Check if there's sufficient memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        if available_gb >= min_gb:
            return True, f"{available_gb:.1f} GB available memory"
        else:
            return False, f"Only {available_gb:.1f} GB available (need {min_gb} GB)"
    except ImportError:
        return True, "Memory check skipped (psutil not installed)"

def check_fhdp_files() -> Tuple[bool, List[Tuple[bool, str]]]:
    """Check if required FHDP files exist"""
    results = []

    required_files = [
        'test_pipeline_training.py',
        'core/types.py',
        'core/fhdp_system.py',
        'edge_server/server.py',
        'vehicle_layer/vehicle.py'
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            results.append((True, f"Found: {file_path}"))
        else:
            results.append((False, f"Missing: {file_path}"))

    all_passed = all(passed for passed, _ in results)
    return all_passed, results

def check_environment() -> Tuple[bool, List[Tuple[bool, str]]]:
    """Check environment variables and configuration"""
    results = []

    # Check for FHDP_PORT
    fhdp_port = os.environ.get('FHDP_PORT')
    if fhdp_port:
        results.append((True, f"FHDP_PORT set to {fhdp_port}"))
    else:
        results.append((True, "FHDP_PORT not set (will use default 5000)"))

    # Check for FHDP_CONFIG
    fhdp_config = os.environ.get('FHDP_CONFIG')
    if fhdp_config:
        if os.path.exists(fhdp_config):
            results.append((True, f"FHDP_CONFIG: {fhdp_config}"))
        else:
            results.append((False, f"FHDP_CONFIG file not found: {fhdp_config}"))

    all_passed = all(passed for passed, _ in results)
    return all_passed, results

def main():
    print_header("FHDP Pipeline Test Environment Validation")

    all_passed = True

    # 1. Check Python version
    print(f"Python Version Check:")
    passed, msg = check_python_version()
    if passed:
        print_success(msg)
    else:
        print_error(msg)
        print_warning("Pipeline test requires Python 3.8 or higher")
        all_passed = False
    print()

    # 2. Check required modules
    print("Required Python Modules:")
    modules = ['yaml', 'numpy', 'torch', 'torch.nn', 'torch.optim']
    for module in modules:
        passed, msg = check_module(module)
        if passed:
            print_success(msg)
        else:
            print_error(msg)
            all_passed = False
    print()

    # 3. Check PyTorch and CUDA
    print("PyTorch and CUDA:")
    all_torch_passed, torch_results = check_pytorch()
    for passed, msg in torch_results:
        if passed:
            print_success(msg)
        else:
            print_error(msg)
            all_passed = False

    if not all_torch_passed:
        print()
        print_warning("PyTorch or CUDA issues detected")
        if platform.machine() == 'aarch64':
            print_warning("You're on ARM64 (likely Jetson)")
            print_warning("Install PyTorch for Jetson:")
            print("  https://developer.nvidia.com/embedded/downloads")
    print()

    # 4. Check network port
    print("Network Port Check:")
    port = int(os.environ.get('FHDP_PORT', 5000))
    passed, msg = check_network_port(port)
    if passed:
        print_success(msg)
    else:
        print_error(msg)
        print_warning("Try using a different port:")
        print("  export FHDP_PORT=6000")
        all_passed = False
    print()

    # 5. Check disk space
    print("Disk Space Check:")
    passed, msg = check_disk_space(min_gb=5)
    if passed:
        print_success(msg)
    else:
        print_error(msg)
        all_passed = False
    print()

    # 6. Check memory
    print("Memory Check:")
    passed, msg = check_memory(min_gb=4)
    if passed:
        print_success(msg)
    else:
        print_error(msg)
        all_passed = False
    print()

    # 7. Check FHDP files
    print("FHDP Files Check:")
    all_files_passed, file_results = check_fhdp_files()
    for passed, msg in file_results:
        if passed:
            print_success(msg)
        else:
            print_error(msg)
            all_passed = False
    print()

    # 8. Check environment
    print("Environment Check:")
    all_env_passed, env_results = check_environment()
    for passed, msg in env_results:
        if passed:
            print_success(msg)
        else:
            print_error(msg)
            all_passed = False
    print()

    # 9. System information
    print_header("System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Node: {platform.node()}")
    print()

    # 10. Summary
    print_header("Validation Summary")
    if all_passed:
        print_success("All checks passed! Your environment is ready for pipeline testing.")
        print()
        print("Next steps:")
        print("  - For server: ./run_pipeline_test.sh server")
        print("  - For vehicle: ./run_pipeline_test.sh <agx|nano> <server-ip>")
        print()
        return 0
    else:
        print_error("Some checks failed. Please fix the issues above before running the test.")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
