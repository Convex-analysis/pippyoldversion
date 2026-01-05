#!/bin/bash

# EVO-1 and FHDP Autonomous Driving Setup Script
# This script sets up the environment for EVO-1 integration with FHDP

echo "🚗 Setting up EVO-1 + FHDP Autonomous Driving Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$python_version" != "$required_version" ]; then
    echo "⚠️  Python $required_version is recommended. Current version: $python_version"
    echo "   Please install Python $required_version for best compatibility"
fi

# Create conda environment (optional)
read -p "Do you want to create a new conda environment? (y/n): " create_env
if [ "$create_env" = "y" ]; then
    env_name="fhdp_evo1"
    echo "📦 Creating conda environment: $env_name"
    conda create -n $env_name python=3.10 -y
    echo "✅ Environment created. Activate with: conda activate $env_name"
    
    # Activate environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $env_name
fi

# Install PyTorch with CUDA support (if available)
read -p "Do you have CUDA support? (y/n): " has_cuda
if [ "$has_cuda" = "y" ]; then
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
else
    echo "💻 Installing PyTorch for CPU..."
    pip install torch torchvision torchaudio
fi

# Install EVO-1 dependencies
echo "📚 Installing EVO-1 dependencies..."
pip install -r requirements_evo1.txt

# Install flash-attn (may need compilation)
echo "⚡ Installing flash-attn for efficient attention..."
read -p "Do you want to install flash-attn? (may require compilation) (y/n): " install_flash
if [ "$install_flash" = "y" ]; then
    # Set MAX_JOBS to limit parallel compilation
    export MAX_JOBS=4
    pip install flash-attn --no-build-isolation
fi

# Clone EVO-1 repository (optional)
read -p "Do you want to clone the EVO-1 repository? (y/n): " clone_evo
if [ "$clone_evo" = "y" ]; then
    echo "📥 Cloning EVO-1 repository..."
    git clone https://github.com/MINT-SJTU/Evo-1.git evo1_repo
    echo "✅ EVO-1 repository cloned to evo1_repo/"
    echo "   You can start the EVO-1 server with:"
    echo "   cd evo1_repo && python scripts/Evo1_server.py"
fi

# Create directories for data
echo "📁 Creating data directories..."
mkdir -p data/nuscenes
mkdir -p models/evo1
mkdir -p logs/evo1

# Download nuScenes mini dataset (optional)
read -p "Do you want to download nuScenes mini dataset? (~4GB) (y/n): " download_nuscenes
if [ "$download_nuscenes" = "y" ]; then
    echo "📥 Downloading nuScenes mini dataset..."
    mkdir -p data/nuscenes
    
    # Note: Users need to register at https://www.nuscenes.org/download
    echo "   Please register at https://www.nuscenes.org/download"
    echo "   After registration, download the mini dataset to data/nuscenes/"
fi

# Create configuration file
echo "⚙️  Creating configuration file..."
cat > config_evo1.yaml << 'EOF'
# EVO-1 + FHDP Configuration
evo1:
  server_url: "ws://localhost:8765"
  model_name: "OpenGVLab/InternVL3-1B"
  image_size: 448
  horizon: 50
  dropout: 0.2
  weight_decay: 1e-3

nuscenes:
  data_root: "./data/nuscenes"
  version: "v1.0-mini"
  camera_names: ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                  "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

federated_learning:
  aggregation_interval: 45  # seconds
  training_interval: 30     # seconds
  num_training_epochs: 3
  batch_size: 8
  
simulation:
  num_vehicles: 4
  simulation_duration: 120  # seconds
  scenario_types: ["highway", "urban", "night"]
  
logging:
  log_level: "INFO"
  save_models: true
  log_dir: "./logs/evo1"
EOF

echo "✅ Configuration saved to config_evo1.yaml"

# Create run script
echo "🚀 Creating run script..."
cat > run_autonomous_driving.sh << 'EOF'
#!/bin/bash

echo "🚗 Starting EVO-1 + FHDP Autonomous Driving Simulation..."

# Check if EVO-1 server is running
if ! curl -s http://localhost:8765 > /dev/null; then
    echo "⚠️  EVO-1 server not detected on port 8765"
    echo "   Please start the EVO-1 server first:"
    echo "   cd evo1_repo && python scripts/Evo1_server.py"
    echo ""
    read -p "Continue anyway? (fallback policy will be used) (y/n): " continue
    if [ "$continue" != "y" ]; then
        exit 1
    fi
fi

# Run the autonomous driving simulation
echo "🧠 Starting autonomous driving simulation..."
python examples/autonomous_driving_simulation.py

echo "✅ Simulation completed!"
EOF

chmod +x run_autonomous_driving.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. (Optional) Start EVO-1 server: cd evo1_repo && python scripts/Evo1_server.py"
echo "   2. Run simulation: ./run_autonomous_driving.sh"
echo "   3. Check configuration: config_evo1.yaml"
echo ""
echo "📚 For more information:"
echo "   - EVO-1 Repository: https://github.com/MINT-SJTU/Evo-1"
echo "   - nuScenes Dataset: https://www.nuscenes.org"
echo "   - FHDP Documentation: ./README.md"
echo ""
echo "🔧 If you encounter issues:"
echo "   - Check Python version (requires 3.10)"
echo "   - Verify CUDA installation if using GPU"
echo "   - Check flash-attn compilation"
echo "   - Ensure all dependencies are installed"