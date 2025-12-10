"""
Setup script for FHDP System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

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