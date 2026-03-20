"""
Setup script for FHDP System
"""
import os
import platform
import re
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Check if running on Jetson device
def is_jetson_device():
    """Detect if running on NVIDIA Jetson platform"""
    try:
        # Check for Jetson-specific files
        if os.path.exists('/etc/nv_tegra_release'):
            return True
        # Check for Jetson in /proc/device-tree/model
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'jetson' in model.lower():
                    return True
        except:
            pass
    except:
        pass
    return False

IS_JETSON = is_jetson_device()

# Read requirements, handling -r references
requirements = []
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                # Handle -r reference
                ref_file = line[3:].strip()
                try:
                    with open(ref_file, "r", encoding="utf-8") as ref_fh:
                        for ref_line in ref_fh:
                            ref_line = ref_line.strip()
                            # Skip torch-related packages on Jetson devices
                            if IS_JETSON and any(pkg in ref_line.lower() for pkg in ['torch', 'torchvision', 'torchaudio']):
                                continue
                            if ref_line and not ref_line.startswith("#"):
                                requirements.append(ref_line)
                except FileNotFoundError:
                    continue  # Skip if referenced file doesn't exist
            else:
                requirements.append(line)
except FileNotFoundError:
    # If requirements.txt doesn't exist, use minimal requirements
    requirements = []

setup(
    name="fhdp",
    version="1.0.0",
    author="FHDP Development Team",
    author_email="fhdp@example.com",
    description="Federated Highway-based Distributed Pipeline for Vehicular Federated Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/fhdp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
        ],
        "performance": [
            "numba>=0.56.0",
        ],
        "security": [
            "pycryptodome>=3.15.0",
        ],
        "communication": [
            "websockets>=10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fhdp=fhdp.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fhdp": [
            "config/*.yaml",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    zip_safe=False,
)