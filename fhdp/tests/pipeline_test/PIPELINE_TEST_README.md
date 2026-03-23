# FHDP Pipeline Training Test

This directory contains a comprehensive test script for validating FHDP's pipeline training capability across heterogeneous devices (Jetson AGX Orin, Jetson Orin Nano, and a 4090 Linux server).

## Overview

The test script simulates a realistic federated learning scenario where:

1. **Server (4090 Linux)**: Acts as the edge server that coordinates pipeline formation and model aggregation
2. **Vehicle 1 (Jetson AGX Orin)**: High-resource vehicle with more computational power
3. **Vehicle 2 (Jetson Orin Nano)**: Medium-resource vehicle with limited computational power

The system demonstrates FHDP's ability to:
- **Form pipelines dynamically based on vehicle capabilities** ✨ (Now using real FHDP logic)
  - Automatic vehicle resource classification (HIGH/MEDIUM/LOW)
  - Intelligent template matching
  - Greedy stage allocation algorithm
- Coordinate training across heterogeneous devices
- Aggregate model updates asynchronously
- Handle communication between server and vehicles
- Manage resource allocation and fairness

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     4090 Linux Server                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Edge Server  │  │   FHDP       │  │   Network    │      │
│  │  Coordinator │  │   System     │  │   Server     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                        │                                      │
└────────────────────────┼──────────────────────────────────────┘
                         │ Network (TCP/IP)
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐  ┌───▼──────────┐  ┌──▼────────────┐
│  Jetson AGX    │  │  Jetson      │  │  More...      │
│  Orin (High)   │  │  Orin Nano   │  │  Vehicles     │
│                │  │  (Medium)    │  │               │
│ ┌────────────┐ │  │ ┌──────────┐ │  │              │
│ │ Vehicle    │ │  │ │ Vehicle  │ │  │              │
│ │ Agent      │ │  │ │ Agent    │ │  │              │
│ └────────────┘ │  │ └──────────┘ │  │              │
│               │  │              │  │              │
└───────────────┘  └──────────────┘  └──────────────┘
```

## Requirements

### Common Requirements
- Python 3.8+
- PyTorch 1.10+
- Network connectivity between all devices
- Sufficient permissions to bind to network ports

### Server Requirements (4090 Linux)
- Linux OS with CUDA support
- NVIDIA 4090 GPU
- PyTorch with CUDA support
- Network port 5000 available (or custom port)

### Vehicle Requirements (Jetson)
- JetPack 5.0+ recommended
- PyTorch for Jetson (refer to NVIDIA's installation guide)
- WiFi or Ethernet connection to server network
- At least 2GB RAM available

## Installation

### On Server (4090 Linux)

```bash
# Clone or navigate to fhdp directory
cd /path/to/fhdp

# Install dependencies
pip install -r requirements.txt

# Make test scripts executable
chmod +x test_pipeline_training.py
chmod +x run_pipeline_test.sh
```

### On Jetson AGX Orin

```bash
# Install PyTorch for Jetson (if not already installed)
# Refer to: https://developer.nvidia.com/embedded/downloads

# Install other dependencies
pip install pyyaml

# Transfer test scripts from server
scp user@server:/path/to/fhdp/test_pipeline_training.py ~/
scp user@server:/path/to/fhdp/run_pipeline_test.sh ~/
chmod +x run_pipeline_test.sh
```

### On Jetson Orin Nano

```bash
# Same installation as AGX Orin
pip install pyyaml
# Transfer test scripts from server
```

## Usage

### Quick Start with Shell Script

The `run_pipeline_test.sh` script simplifies launching the test:

#### 1. Start Server (on 4090 Linux)

```bash
cd /path/to/fhdp
./run_pipeline_test.sh server
```

#### 2. Start Vehicle on Jetson AGX Orin

```bash
# Replace <server-ip> with actual server IP
./run_pipeline_test.sh agx <server-ip>
```

#### 3. Start Vehicle on Jetson Orin Nano

```bash
# Replace <server-ip> with actual server IP
./run_pipeline_test.sh nano <server-ip>
```

### Direct Python Execution

Alternatively, run the Python script directly:

#### Server Mode

```bash
python test_pipeline_training.py --mode server --host 0.0.0.0 --port 5000
```

#### Vehicle Mode (AGX Orin)

```bash
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id agx_orin_001 \
    --server-host <server-ip> \
    --server-port 5000 \
    --resource-level high
```

#### Vehicle Mode (Orin Nano)

```bash
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id orin_nano_001 \
    --server-host <server-ip> \
    --server-port 5000 \
    --resource-level medium
```

### Custom Configuration

You can use a custom configuration file:

```bash
# Server with custom config
python test_pipeline_training.py --mode server --config config/custom_config.yaml

# Vehicle with custom config
python test_pipeline_training.py --mode vehicle --vehicle-id v1 --server-host 192.168.1.100 --config config/custom_config.yaml
```

### Environment Variables

```bash
# Set custom port
export FHDP_PORT=6000

# Set custom config path
export FHDP_CONFIG=/path/to/config.yaml

# Use specific Python interpreter
export PYTHON=/usr/bin/python3.9
```

## Expected Output

### Server Output

```
╔════════════════════════════════════════════════════════════╗
║     FHDP Pipeline Training Test Launcher                  ║
║     Testing pipeline training across Jetson devices       ║
╚════════════════════════════════════════════════════════════╝
✓ Using Python: Python 3.9.0
✓ PyTorch installed, CUDA available: True

==================================
Starting Server Mode
==================================
Host: 0.0.0.0
Port: 5000

Executing: python test_pipeline_training.py --mode server --host 0.0.0.0 --port 5000

============================================================
Starting FHDP Pipeline Training Test Server
============================================================
✓ FHDP system started
✓ Edge server started
✓ Network server started

Server is ready to accept vehicle connections...
Listen on: 0.0.0.0:5000

✓ Vehicle agx_orin_001 registered
  Position: (0.0, 0.0)
  Resources: {...}

✓ Vehicle orin_nano_001 registered
  Position: (0.0, 0.0)
  Resources: {...}

============================================================
Attempting to form pipeline using FHDP stage partitioning...
============================================================

Step 1: Classifying vehicles by resource capability...
  agx_orin_001: high
  orin_nano_001: medium

Step 2: Finding best matching pipeline template...
  Template ID: template_42
  Resource requirements: ['high', 'medium']
  Expected duration: 60.0s

Step 3: Performing greedy stage selection...

✓ Pipeline formed successfully!
  Pipeline ID: pipeline_1711234567890
  Template: template_42
  Vehicles in pipeline: ['agx_orin_001', 'orin_nano_001']
  Stages: ['stage_0', 'stage_1']
  Vehicle-Stage mapping:
    agx_orin_001 -> stage_0 (required: high, actual: high)
    orin_nano_001 -> stage_1 (required: medium, actual: medium)

→ Sent pipeline invitation to agx_orin_001
→ Sent pipeline invitation to orin_nano_001

==================== Round 1 ====================
✓ Broadcast global model for round 1

[Round 1] Received update from agx_orin_001
[Round 1] Received update from orin_nano_001
→ Aggregating updates for round 1...
✓ Aggregation completed for round 1
  Aggregated 2 model updates
  Global model accuracy (round 1): 75.00%

==================== Round 2 ====================
✓ Broadcast global model for round 2
...

============================================================
Pipeline training completed!
============================================================

Training Summary
============================================================
Vehicles registered: 2
Pipelines formed: 1
Training rounds: 3
Aggregations performed: 3
Total updates received: 6
============================================================
```

### Vehicle Output (AGX Orin)

```
============================================================
Starting Vehicle: agx_orin_001
============================================================
Server: 192.168.1.100:5000
Resources: {'cpu': 0.9, 'memory': 0.8, ...}
Connected to server 192.168.1.100:5000
✓ Registration sent to server
✓ Vehicle started and registered

Received pipeline invitation from server
  Pipeline ID: pipeline_1711234567890
  Template ID: template_42
  Vehicles in pipeline: ['agx_orin_001', 'orin_nano_001']
  Stages: ['stage_0', 'stage_1']
  Assigned stage: stage_0 (requires: high)
✓ Accepted pipeline invitation

Received global model for round 1
→ Starting local training for round 1...
  Epochs: 2
  Batch size: 32
  Learning rate: 0.001
  Epoch 1/2, Batch 1/7, Loss: 2.4567
  Epoch 1/2, Batch 6/7, Loss: 2.1234
  Epoch 1 completed, Avg Loss: 2.2891
  Epoch 2/2, Batch 1/7, Loss: 2.1123
  Epoch 2/2, Batch 6/7, Loss: 1.9876
  Epoch 2 completed, Avg Loss: 2.0456
✓ Local training completed for round 1
  Average loss: 2.1674
→ Sending model update for round 1...
✓ Model update sent for round 1

Received global model for round 2
...
```

## Troubleshooting

### Connection Issues

**Problem**: Vehicle cannot connect to server

**Solutions**:
- Verify server IP is correct
- Check firewall settings on server: `sudo ufw allow 5000/tcp`
- Ensure devices are on the same network
- Ping server from vehicle: `ping <server-ip>`

### PyTorch Issues

**Problem**: PyTorch not found on Jetson

**Solutions**:
- Install PyTorch for Jetson from NVIDIA's official repository
- Follow guide at: https://developer.nvidia.com/embedded/downloads
- Use JetPack SDK to install dependencies

### Port Already in Use

**Problem**: Port 5000 already in use

**Solutions**:
```bash
# Use different port
export FHDP_PORT=6000
./run_pipeline_test.sh server

# Or kill process using port 5000
lsof -ti:5000 | xargs kill -9
```

### Memory Issues

**Problem**: Out of memory on Jetson Nano

**Solutions**:
- Reduce batch size in training config
- Reduce number of epochs
- Close other applications
- Use model with fewer parameters

## Advanced Configuration

### Custom Training Configuration

Create a custom YAML config file:

```yaml
# custom_config.yaml
system:
  max_vehicles_per_region: 10
  aggregation_interval: 2.0
  enable_pipeline_training: true

training:
  learning_rate: 0.001
  batch_size: 16  # Smaller for Orin Nano
  optimizer: "sgd"
  loss_function: "cross_entropy"
```

Then use it:

```bash
python test_pipeline_training.py --mode server --config custom_config.yaml
```

### Modifying Model Architecture

Edit `SimpleCNN` class in `test_pipeline_training.py` to use different architectures:

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Testing Checklist

Before running the full test, verify:

- [ ] All devices have Python 3.8+ installed
- [ ] PyTorch is installed on all devices
- [ ] Network connectivity between all devices
- [ ] Server can bind to port 5000
- [ ] Firewalls allow TCP traffic on port 5000
- [ ] Jetson devices have sufficient memory
- [ ] Server has sufficient GPU memory

## Performance Expectations

Based on FHDP design goals:

- **Pipeline Formation**: < 1.5 seconds
- **Template Lookup**: < 5 milliseconds
- **Aggregation Latency**: < 2 seconds
- **Training per Round**: 1-2 epochs, < 10 seconds per vehicle
- **Memory Usage**: < 50MB for FHDP components per vehicle
- **Network Traffic**: Optimized with 64KB bundles

## Extending the Test

### Adding More Vehicles

To test with more than 2 vehicles:

```bash
# Terminal 3 - Vehicle 3
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id vehicle_003 \
    --server-host <server-ip> \
    --resource-level medium

# Terminal 4 - Vehicle 4
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id vehicle_004 \
    --server-host <server-ip> \
    --resource-level low
```

### Testing Pipeline Reformation

To test dynamic pipeline reformation:

1. Start server and 3+ vehicles
2. After training starts, terminate one vehicle (Ctrl+C)
3. Observe if server reforms pipeline with remaining vehicles

## Stage Partitioning Implementation ✨

The test script now implements authentic FHDP stage partitioning logic instead of hardcoded placeholders.

### Key Components

#### 1. ResourceClassifier
**Location**: `fhdp/edge_server/resource_classifier.py`

Classifies vehicles based on CPU, memory, and battery:
```python
classifier = ResourceClassifier()
resource_class = classifier.classify_vehicle(vehicle_info)
# Returns: ResourceClass.HIGH / MEDIUM / LOW
```

#### 2. TemplateManager
**Location**: `fhdp/edge_server/template_manager.py`

Finds optimal pipeline templates for vehicle groups:
```python
template_manager = TemplateManager()
template = template_manager.find_template_for_vehicles(vehicles)
# Returns: PipelineTemplate with resource requirements
```

#### 3. PipelineFormation
**Location**: `fhdp/vehicle_layer/pipeline_formation.py`

Performs greedy stage allocation based on multiple scores:
- Resource match score (40%)
- Mobility stability score (30%)
- Communication quality score (30%)

### Stage Partitioning Flow

```
┌───────────────────────────────────────────────────┐
│ Step 1: Vehicle Registration & Classification     │
├───────────────────────────────────────────────────┤
│ Vehicle → ResourceClassifier.classify_vehicle()   │
│ Result: vehicle_id → ResourceClass               │
└───────────────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│ Step 2: Find Matching Pipeline Template         │
├───────────────────────────────────────────────────┤
│ TemplateManager.find_template_for_vehicles()      │
│ Input: List[VehicleInfo]                        │
│ Output: PipelineTemplate                        │
│ Example: [HIGH, MEDIUM] → 2-stage template   │
└───────────────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│ Step 3: Greedy Stage Allocation                │
├───────────────────────────────────────────────────┤
│ PipelineFormation.initiate_pipeline_formation()  │
│ • Calculate vehicle scores for each stage       │
│ • Select best match for stage_0                │
│ • Select best remaining for stage_1            │
│ • Output: Pipeline with vehicle-stage mapping   │
└───────────────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│ Step 4: Send Invitation with Assignment         │
├───────────────────────────────────────────────────┤
│ Server → Vehicle: PIPELINE_INVITE              │
│ {                                              │
│   'pipeline_id': 'pipeline_123',              │
│   'vehicles': ['v1', 'v2'],                  │
│   'stages': ['stage_0', 'stage_1'],           │
│   'resource_requirements': ['high', 'medium']   │
│ }                                              │
│                                                │
│ Vehicle → Server: PIPELINE_RESPONSE             │
│ { 'accepted': True, 'stage': 'stage_0' }      │
└───────────────────────────────────────────────────┘
```

### Greedy Allocation Algorithm

**Scoring Factors**:
| Factor | Weight | Description |
|--------|---------|-------------|
| Resource Match | 40% | Vehicle resources vs stage requirements |
| Mobility | 30% | Velocity stability, direction consistency |
| Communication | 30% | Network bandwidth, signal strength |

**Position-Specific Weights**:
- First/Last stages: Resource weight 0.5 (more critical)
- Middle stages: Balanced weights (0.4 each)

### Validation

Run the validation script:
```bash
python3 validate_stage_partitioning.py
```

Expected output:
```
============================================================
FHDP Stage Partitioning Validation
============================================================
✓ All imports successful
✓ Resource classification working correctly
✓ Template manager working
✓ Pipeline formation working
All validation tests passed!
```

## References

- FHDP Architecture: `/fhdp/docs/ARCHITECTURE.md`
- Core Types: `/fhdp/core/types.py`
- System Implementation: `/fhdp/core/fhdp_system.py`
- Edge Server: `/fhdp/edge_server/server.py`
- Vehicle Layer: `/fhdp/vehicle_layer/vehicle.py`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review FHDP documentation in `/fhdp/docs/`
3. Examine logs in the output for error messages
4. Verify network connectivity and configuration

## License

This test script follows the same license as the FHDP project.
