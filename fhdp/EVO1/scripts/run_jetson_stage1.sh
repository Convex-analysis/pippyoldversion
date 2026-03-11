#!/bin/bash

# EVO-1 Stage 1 Simulation Script for NVIDIA Jetson Devices
# This script runs the Stage 1 simulation on Jetson devices

echo "🚀 EVO-1 Stage 1 Simulation for Jetson Devices"
echo "==========================================="

# Check Jetson device
JETSON_MODEL=""
if [ -f /proc/device-tree/model ]; then
    JETSON_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
fi

echo "📱 Detected Jetson: $JETSON_MODEL"

# Set configuration based on Jetson model
if [[ "$JETSON_MODEL" == *"Orin"* ]]; then
    MAX_MEMORY_MB=6144
    MAX_BATCH_SIZE=4
    SIMULATION_DURATION=120
    echo "🔥 Jetson Orin detected: High performance mode"
elif [[ "$JETSON_MODEL" == *"Nano"* ]]; then
    MAX_MEMORY_MB=3072
    MAX_BATCH_SIZE=2
    SIMULATION_DURATION=60
    echo "📱 Jetson Nano detected: Power-optimized mode"
else
    MAX_MEMORY_MB=4096
    MAX_BATCH_SIZE=3
    SIMULATION_DURATION=90
    echo "⚡ Generic Jetson device detected"
fi

# Simulation parameters
DURATION=${1:-$SIMULATION_DURATION}  # Default duration based on device
EPOCHS=${2:-3}  # Default training epochs: 3
NUM_VEHICLES=${3:-1}  # Default number of vehicles: 1 (Jetson-optimized)

echo ""
echo "📊 Simulation Parameters:"
echo "   Duration: $DURATION seconds"
echo "   Training epochs: $EPOCHS"
echo "   Number of vehicles: $NUM_VEHICLES"
echo "   Max memory: ${MAX_MEMORY_MB}MB"
echo "   Max batch size: ${MAX_BATCH_SIZE}"
echo "==========================================="

# Check if requirements are installed
if ! python3 -c "import torch, numpy, pydantic" > /dev/null 2>&1; then
    echo "📦 Installing dependencies..."
    pip install -r requirements_jetson_stage1.txt
fi

# Run the Stage 1 simulation
echo ""
echo "🔥 Starting Stage 1 simulation..."
python3 -c "from examples.evo1_stage1_federated import main; import asyncio; asyncio.run(main())"

echo ""
echo "✅ Stage 1 simulation completed successfully!"
echo "📈 Check the outputs/ directory for results."
echo "==========================================="
