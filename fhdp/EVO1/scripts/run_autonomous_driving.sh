#!/bin/bash

# EVO-1 Autonomous Driving Simulation Script
# This script runs the autonomous driving simulation using EVO-1 and FHDP

echo "🚗 Starting EVO-1 Autonomous Driving Simulation..."
echo "=" * 60

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$python_version" != "$required_version" ]; then
    echo "⚠️  Python $required_version is recommended. Current version: $python_version"
fi

# Check if requirements are installed
if ! python3 -c "import torch, numpy, pydantic" > /dev/null 2>&1; then
    echo "📦 Installing dependencies..."
    pip install -r requirements_evo1.txt
fi

# Simulation parameters
DURATION=${1:-60}  # Default duration: 60 seconds
NUM_VEHICLES=${2:-3}  # Default number of vehicles: 3
SIMULATION_TYPE=${3:-"simple"}  # Default simulation type: simple

echo "📊 Simulation Parameters:"
echo "   Duration: $DURATION seconds"
echo "   Number of vehicles: $NUM_VEHICLES"
echo "   Simulation type: $SIMULATION_TYPE"
echo "=" * 60

# Run the simulation
echo "🔥 Starting simulation..."
python3 -c "from examples.autonomous_driving_simulation import run_simulation; run_simulation(duration=$DURATION, num_vehicles=$NUM_VEHICLES, simulation_type='$SIMULATION_TYPE')"

echo "\n✅ Simulation completed successfully!"
echo "📈 Check the outputs/ directory for results."
