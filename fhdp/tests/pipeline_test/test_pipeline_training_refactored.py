#!/usr/bin/env python3
"""
FHDP Pipeline Training Test Script (Refactored)

This script tests the FHDP system's capability to orchestrate pipeline training
across two Jetson devices (AGX Orin and Orin Nano) with a 4090 Linux server.

Refactored to use FHDP's built-in cross_platform_comm.py for reliable network
communication with length-prefix protocol and compression support.

Architecture:
- Server: Edge Server (4090 Linux) - Coordinates pipeline formation and aggregation
- Client 1: Jetson AGX Orin - High-resource vehicle
- Client 2: Jetson Orin Nano - Medium-resource vehicle

Usage:
    # On the 4090 server:
    python test_pipeline_training_refactored.py --mode server --host 0.0.0.0 --port 5000

    # On Jetson AGX Orin:
    python test_pipeline_training_refactored.py --mode vehicle --vehicle-id agx_orin_001 --server-host <server-ip> --server-port 5000 --resource-level high

    # On Jetson Orin Nano:
    python test_pipeline_training_refactored.py --mode vehicle --vehicle-id orin_nano_001 --server-host <server-ip> --server-port 5000 --resource-level medium
"""

import sys
import os
import argparse
import time
import socket
import threading
import signal
import yaml
import torch
import torch.nn as nn
import numpy as np
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable

# Add project root to path - support both local and remote deployment
script_dir = os.path.dirname(os.path.abspath(__file__))

# Debug: print current working directory and script location
print(f"[DEBUG] Script dir: {script_dir}")
print(f"[DEBUG] CWD: {os.getcwd()}")

# Try multiple possible project root locations
# Order matters: try fhdp as root first (remote case), then parent (local case)
possible_roots = [
    os.path.abspath(os.path.join(script_dir, '..')),    # fhdp/tests/pipeline_test -> fhdp/ (if fhdp is project root)
    os.path.abspath(os.path.join(script_dir, '../..')),  # fhdp/tests/pipeline_test -> project root (if nested under fhdp/)
    os.path.abspath(os.path.join(script_dir, '../../..')), # deeper nesting
    os.getcwd()  # Current directory as fallback
]

project_root = None
for root in possible_roots:
    print(f"[DEBUG] Checking root: {root}")
    # Check if fhdp module exists at this root level OR if root contains core/ directory
    check_path1 = os.path.join(root, 'fhdp')
    check_path2 = os.path.join(root, 'core')  # Check if root IS fhdp
    if os.path.exists(check_path1):
        project_root = root
        print(f"[DEBUG] Found fhdp/ at {check_path1}")
        break
    elif os.path.exists(check_path2):
        # root is the fhdp package directory itself, use its parent
        project_root = os.path.dirname(root)
        print(f"[DEBUG] Found core/ at {check_path2}, root is fhdp, using parent: {project_root}")
        break

if project_root is not None:
    sys.path.insert(0, project_root)
    print(f"[INFO] Using project root: {project_root}")
    # Verify fhdp import works
    try:
        import fhdp
        print(f"[INFO] Successfully imported fhdp from {fhdp.__file__}")
    except ImportError as e:
        print(f"[ERROR] Cannot import fhdp despite adding root: {e}")
else:
    print("[WARNING] Could not determine project root, trying direct import...")
    # Try importing without modifying path
    try:
        import fhdp
        print(f"[INFO] Direct import successful: {fhdp.__file__}")
    except ImportError as e:
        print(f"[ERROR] Cannot import fhdp: {e}")
        print(f"[HINT] Please run this script from the project root directory")
        print(f"[HINT] Or set PYTHONPATH to include the project root")
        sys.exit(1)

from fhdp.core.types import (
    VehicleInfo, VehicleState, TrainingMode, Pipeline, PipelineTemplate,
    ModelUpdate, AggregationResult, ResourceClass, TrainingConfig
)
from fhdp.vehicle_layer.training_engine import TrainingTask
from fhdp.core.fhdp_system import FHDPSystem, SystemConfiguration
from fhdp.edge_server.server import EdgeServer
from fhdp.vehicle_layer.vehicle import Vehicle

# Use FHDP's built-in cross-platform communication
from fhdp.core.cross_platform_comm import (
    NetworkEndpoint, CrossPlatformMessage, PlatformBridge,
    TransportProtocol, CompressionType, HardwareCapabilities, HardwarePlatform
)


# ==================== Simple Model for Testing ====================

class SimpleCNN(nn.Module):
    """Simple CNN model for testing"""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate flattened size after convolutions
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def _serialize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert torch.Tensor values to lists for JSON serialization."""
    return {k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in state_dict.items()}


def _deserialize_state_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Restore lists back to torch.Tensor after JSON deserialization."""
    return {k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in raw.items()}


def create_mock_data_loader(num_samples: int = 100, batch_size: int = 32):
    """Create mock data loader for testing"""

    class MockDataLoader:
        def __init__(self, num_samples, batch_size):
            self.num_samples = num_samples
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, self.num_samples, self.batch_size):
                batch_size = min(self.batch_size, self.num_samples - i)

                # Mock images (MNIST-like: 1x28x28)
                images = torch.randn(batch_size, 1, 28, 28)
                labels = torch.randint(0, 10, (batch_size,))

                yield images, labels

        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    return MockDataLoader(num_samples, batch_size)


# ==================== Server Mode ====================

class PipelineTestServer:
    """Server mode for pipeline training test using FHDP's cross_platform_comm"""

    def __init__(self, host: str, port: int, config_path: Optional[str] = None):
        self.host = host
        self.port = port

        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            config = SystemConfiguration(**config_data.get('system', {}))
        else:
            config = SystemConfiguration()

        # Initialize FHDP system and edge server
        self.fhdp_system = FHDPSystem(config)
        self.edge_server = EdgeServer(config_path)

        # Initialize FHDP components for pipeline formation
        from fhdp.edge_server.resource_classifier import ResourceClassifier
        from fhdp.edge_server.template_manager import TemplateManager
        from fhdp.vehicle_layer.pipeline_formation import PipelineFormation

        self.resource_classifier = ResourceClassifier()
        self.template_manager = TemplateManager()

        # Create dummy vehicle info for PipelineFormation initialization
        dummy_vehicle_info = VehicleInfo(
            vehicle_id="dummy",
            position=(0.0, 0.0),
            velocity=0.0,
            direction=0.0,
            resources={"cpu": 4, "memory": 16, "gpu": 1},
            state=VehicleState.IDLE
        )
        self.pipeline_formation = PipelineFormation(dummy_vehicle_info)

        # Initialize cross-platform communication
        self.server_endpoint = NetworkEndpoint(
            host=host,
            port=port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.ZLIB
        )

        # Create platform bridge with x86 Linux capabilities
        from fhdp.core.hardware_adapter import ComputeCapability

        self.local_capabilities = HardwareCapabilities(
            platform=HardwarePlatform.X86_LINUX,
            compute_capability=ComputeCapability.SERVER_CLASS,
            cpu_cores=32,
            cpu_freq=3.5,
            memory_total=64.0,
            gpu_memory=24.0,
            npu_memory=0.0,
            storage_speed='ssd',
            network_speed=1000.0,
            power_profile='high_performance',
            thermal_limit=95.0,
            accelerated_compute=True
        )
        self.platform_bridge = PlatformBridge(self.local_capabilities)

        # Pipeline management
        self.active_pipeline: Optional[Pipeline] = None
        self.pipeline_formed = False
        self.global_model = SimpleCNN()

        # Statistics
        self.stats = {
            'vehicles_registered': 0,
            'pipelines_formed': 0,
            'aggregations_performed': 0,
            'total_updates_received': 0,
            'training_rounds': 0
        }

        # Lock for thread safety
        self.lock = threading.Lock()

        # Pending updates for aggregation
        self.pending_updates: Dict[str, Dict] = {}
        self.round_num = 0

    def start(self):
        """Start server"""
        print("=" * 60)
        print("Starting FHDP Pipeline Training Test Server")
        print("=" * 60)

        # Start FHDP system
        self.fhdp_system.start_system()
        print("✓ FHDP system started")

        # Start edge server
        self.edge_server.start_server()
        print("✓ Edge server started")

        # Register message handlers
        self._register_message_handlers()

        # Start TCP listener using cross_platform_comm
        self.platform_bridge.message_router.start_message_listener(self.server_endpoint)

        print(f"✓ Network server listening on {self.host}:{self.port}")
        print("\nServer is ready to accept vehicle connections...")
        print(f"Listen on: {self.host}:{self.port}\n")

    def stop(self):
        """Stop server"""
        print("\nStopping server...")

        self.edge_server.stop_server()
        self.fhdp_system.stop_system()

        print("✓ Server stopped")

    def _register_message_handlers(self):
        """Register message handlers for cross-platform communication"""

        self.platform_bridge.message_router.register_handler(
            'register',
            self._handle_register
        )

        self.platform_bridge.message_router.register_handler(
            'model_update',
            self._handle_model_update
        )

        self.platform_bridge.message_router.register_handler(
            'heartbeat',
            self._handle_heartbeat
        )

        self.platform_bridge.message_router.register_handler(
            'pipeline_response',
            self._handle_pipeline_response
        )

        self.platform_bridge.message_router.register_handler(
            'status',
            self._handle_status
        )

    def _handle_register(self, message: CrossPlatformMessage):
        """Handle vehicle registration"""
        vehicle_data = message.payload

        # Create vehicle info
        vehicle_info = VehicleInfo(
            vehicle_id=message.source_id,
            position=vehicle_data.get('position', (0.0, 0.0)),
            velocity=vehicle_data.get('velocity', 0.0),
            direction=vehicle_data.get('direction', 0.0),
            resources=vehicle_data.get('resources', {}),
            state=VehicleState.IDLE
        )

        # Register with FHDP system
        success = self.fhdp_system.register_vehicle(vehicle_info)
        if success:
            # Register with edge server
            success = self.edge_server.register_vehicle(vehicle_info)

        if success:
            # Classify vehicle resources
            resource_class = self.resource_classifier.classify_vehicle(vehicle_info)
            print(f"✓ Vehicle {message.source_id} registered (Resource Class: {resource_class.value})")
            print(f"  Position: {vehicle_info.position}")
            print(f"  Resources: {vehicle_info.resources}")

            # Create route to this vehicle
            # Note: client_socket is already stored in message_router.client_connections
            # by the _handle_tcp_connection method, so we don't need to specify client_socket here
            vehicle_endpoint = NetworkEndpoint(
                host=message.metadata.get('client_host', 'unknown'),
                port=message.metadata.get('client_port', 0),
                protocol=TransportProtocol.TCP,
                compression=CompressionType.ZLIB
            )
            self.platform_bridge.message_router.add_route(message.source_id, vehicle_endpoint)

            with self.lock:
                self.stats['vehicles_registered'] += 1

            # Check if we have enough vehicles for pipeline
            if self.stats['vehicles_registered'] >= 2 and not self.pipeline_formed:
                self._try_form_pipeline()

    def _handle_model_update(self, message: CrossPlatformMessage):
        """Handle model update from vehicle"""
        with self.lock:
            self.stats['total_updates_received'] += 1

        update_data = message.payload
        source_id = message.source_id
        round_num = update_data.get('round', 0)

        print(f"[Round {round_num}] Received update from {source_id}")

        # Store update
        self.pending_updates[source_id] = update_data

        # Check if we have updates from all pipeline vehicles
        if self.active_pipeline:
            pipeline_vehicles = set(self.active_pipeline.vehicles)
            received_updates = set(self.pending_updates.keys())

            if received_updates == pipeline_vehicles:
                # Aggregate updates
                updates = [self.pending_updates[vid] for vid in self.active_pipeline.vehicles]
                self._aggregate_model_updates(round_num, updates)
                self.pending_updates.clear()

    def _handle_heartbeat(self, message: CrossPlatformMessage):
        """Handle heartbeat from vehicle"""
        # Just acknowledge - keep connection alive
        pass

    def _handle_pipeline_response(self, message: CrossPlatformMessage):
        """Handle pipeline invitation response"""
        response = message.payload
        accepted = response.get('accepted', False)

        if accepted:
            print(f"✓ Vehicle {message.source_id} accepted pipeline invitation")

            # Check if all vehicles have accepted
            if self.active_pipeline:
                if len(self.active_pipeline.vehicles) >= 2:
                    self._start_pipeline_training()
        else:
            print(f"✗ Vehicle {message.source_id} declined pipeline invitation")

    def _handle_status(self, message: CrossPlatformMessage):
        """Handle status request"""
        status = self.get_status()

        # Send response using cross-platform communication
        response_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=message.source_id,
            message_type="status_response",
            payload=status
        )

        target_endpoint = self.platform_bridge.message_router.routing_table.get(message.source_id)
        if target_endpoint:
            self.platform_bridge.send_cross_platform_message(response_msg)

    def _try_form_pipeline(self):
        """Try to form a pipeline with registered vehicles using FHDP stage partitioning"""
        registered_vehicles = list(self.fhdp_system.registered_vehicles.values())

        if len(registered_vehicles) >= 2:
            print("\n" + "=" * 60)
            print("Attempting to form pipeline using FHDP stage partitioning...")
            print("=" * 60)

            # Get registered vehicle info
            candidate_vehicles = registered_vehicles[:2]

            # Step 1: Classify vehicles by resource capability
            print("\nStep 1: Classifying vehicles by resource capability...")
            vehicle_classes = {}
            for vehicle in candidate_vehicles:
                resource_class = self.resource_classifier.classify_vehicle(vehicle)
                vehicle_classes[vehicle.vehicle_id] = resource_class
                print(f"  {vehicle.vehicle_id}: {resource_class.value}")

            # Step 2: Find best matching template
            print("\nStep 2: Finding best matching pipeline template...")
            template = self.template_manager.find_template_for_vehicles(candidate_vehicles)

            if template is None:
                print("✗ No suitable template found for current vehicles")
                print("  Using default 2-stage pipeline template")
                # Create a default template
                template = PipelineTemplate(
                    template_id="default_2stage",
                    resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM],
                    expected_duration=60.0,
                    communication_pattern=[(0, 1)],
                    training_config=TrainingConfig(epochs=2, batch_size=32, learning_rate=0.001)
                )

            print(f"  Template ID: {template.template_id}")
            print(f"  Resource requirements: {[r.value for r in template.resource_requirements]}")
            print(f"  Expected duration: {template.expected_duration}s")

            # Step 3: Perform greedy stage selection
            print("\nStep 3: Performing greedy stage selection...")
            pipeline_id = self.pipeline_formation.initiate_pipeline_formation(
                template=template,
                candidate_vehicles=candidate_vehicles
            )

            if pipeline_id is None:
                print("✗ Failed to form pipeline")
                return

            # Get the formed pipeline
            self.active_pipeline = self.pipeline_formation.active_pipelines[pipeline_id]

            print("\n✓ Pipeline formed successfully!")
            print(f"  Pipeline ID: {pipeline_id}")
            print(f"  Template: {template.template_id}")
            print(f"  Vehicles in pipeline: {self.active_pipeline.vehicles}")
            print(f"  Stages: {self.active_pipeline.stages}")
            print(f"  Vehicle-Stage mapping:")
            for i, (vid, stage) in enumerate(zip(self.active_pipeline.vehicles, self.active_pipeline.stages)):
                rclass = vehicle_classes.get(vid, ResourceClass.MEDIUM)
                print(f"    {vid} -> {stage} (required: {template.resource_requirements[i].value}, actual: {rclass.value})")

            with self.lock:
                self.stats['pipelines_formed'] += 1
                self.pipeline_formed = True

            # Send invitations to vehicles with stage assignment
            invitation = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id="server",
                target_id="",  # Will be set per vehicle
                message_type="pipeline_invite",
                payload={
                    'pipeline_id': pipeline_id,
                    'template_id': template.template_id,
                    'vehicles': self.active_pipeline.vehicles,
                    'stages': self.active_pipeline.stages,
                    'resource_requirements': [r.value for r in template.resource_requirements]
                }
            )

            for vehicle_id in self.active_pipeline.vehicles:
                invitation.target_id = vehicle_id
                target_endpoint = self.platform_bridge.message_router.routing_table.get(vehicle_id)
                if target_endpoint:
                    self.platform_bridge.send_cross_platform_message(invitation)
                    print(f"→ Sent pipeline invitation to {vehicle_id}")

    def _start_pipeline_training(self):
        """Start pipeline training"""
        print("\n" + "=" * 60)
        print("Pipeline formed! Starting training...")
        print("=" * 60)
        print(f"Pipeline ID: {self.active_pipeline.pipeline_id}")
        print(f"Vehicles: {self.active_pipeline.vehicles}")
        print(f"Stages: {self.active_pipeline.stages}")
        print("=" * 60 + "\n")

        # Start training rounds
        self._start_training_round(1)

    def _start_training_round(self, round_num: int):
        """Start a training round"""
        print(f"\n{'='*20} Round {round_num} {'='*20}")

        # Broadcast global model
        global_model_state = _serialize_state_dict(self.global_model.state_dict())

        broadcast_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id="",  # Broadcast to all
            message_type="global_model",
            payload={
                'round': round_num,
                'model_state': global_model_state,
                'training_config': {
                    'epochs': 2,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            }
        )

        # Broadcast to all vehicles in pipeline
        for vehicle_id in self.active_pipeline.vehicles:
            broadcast_msg.target_id = vehicle_id
            target_endpoint = self.platform_bridge.message_router.routing_table.get(vehicle_id)
            if target_endpoint:
                self.platform_bridge.send_cross_platform_message(broadcast_msg)

        print(f"✓ Broadcast global model for round {round_num}")

    def _aggregate_model_updates(self, round_num: int, updates: List[Dict]):
        """Aggregate model updates from all vehicles"""
        print(f"\n→ Aggregating updates for round {round_num}...")

        # Simple averaging aggregation
        aggregated_state = {}
        num_updates = len(updates)

        for key in updates[0]['model_state'].keys():
            # Average parameters (deserialize list→Tensor if needed)
            tensors = [
                torch.tensor(update['model_state'][key])
                if isinstance(update['model_state'][key], list)
                else update['model_state'][key]
                for update in updates
            ]
            aggregated_state[key] = torch.mean(
                torch.stack(tensors),
                dim=0
            )

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

        with self.lock:
            self.stats['aggregations_performed'] += 1
            self.stats['training_rounds'] += 1

        print(f"✓ Aggregation completed for round {round_num}")
        print(f"  Aggregated {num_updates} model updates")

        # Evaluate global model
        self._evaluate_global_model(round_num)

        # Start next round if not done
        if round_num < 3:  # Test with 3 rounds
            time.sleep(2.0)
            self._start_training_round(round_num + 1)
        else:
            print("\n" + "=" * 60)
            print("Pipeline training completed!")
            print("=" * 60)
            self._print_summary()

    def _evaluate_global_model(self, round_num: int):
        """Evaluate global model (simplified)"""
        mock_accuracy = 0.7 + (round_num * 0.05)
        print(f"  Global model accuracy (round {round_num}): {mock_accuracy:.2%}")

    def _print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Vehicles registered: {self.stats['vehicles_registered']}")
        print(f"Pipelines formed: {self.stats['pipelines_formed']}")
        print(f"Training rounds: {self.stats['training_rounds']}")
        print(f"Aggregations performed: {self.stats['aggregations_performed']}")
        print(f"Total updates received: {self.stats['total_updates_received']}")
        print("=" * 60)

    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            'pipeline_formed': self.pipeline_formed,
            'active_pipeline': {
                'pipeline_id': self.active_pipeline.pipeline_id,
                'vehicles': self.active_pipeline.vehicles,
                'stages': self.active_pipeline.stages
            } if self.active_pipeline else None,
            'stats': self.stats.copy(),
            'system_status': self.fhdp_system.get_system_status()
        }


# ==================== Vehicle Mode ====================

class PipelineTestVehicle:
    """Vehicle mode for pipeline training test using FHDP's cross_platform_comm"""

    def __init__(self, vehicle_id: str, server_host: str, server_port: int,
                 resource_level: str = "medium", position: Tuple[float, float] = (0.0, 0.0)):
        self.vehicle_id = vehicle_id
        self.server_host = server_host
        self.server_port = server_port
        self.position = position

        # Determine resource capabilities based on level
        if resource_level == "high":
            # AGX Orin
            self.resources = {
                'cpu': 0.9,
                'memory': 0.8,
                'battery': 0.9,
                'network_quality': 0.8,
                'thermal_state': 0.3,
                'cpu_capacity': 12.0,  # 12 cores
                'memory_capacity': 32.0,  # 32GB
                'battery_capacity': 50.0,
                'bandwidth': 100.0
            }
        else:
            # Orin Nano
            self.resources = {
                'cpu': 0.6,
                'memory': 0.5,
                'battery': 0.7,
                'network_quality': 0.7,
                'thermal_state': 0.4,
                'cpu_capacity': 6.0,  # 6 cores
                'memory_capacity': 8.0,  # 8GB
                'battery_capacity': 20.0,
                'bandwidth': 50.0
            }

        # Initialize components
        self.model = SimpleCNN()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize cross-platform communication
        self.server_endpoint = NetworkEndpoint(
            host=server_host,
            port=server_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.ZLIB
        )

        # Create platform bridge with Jetson capabilities
        from fhdp.core.hardware_adapter import ComputeCapability

        local_platform = HardwarePlatform.JETSON_ORIN if resource_level == "high" else HardwarePlatform.JETSON_NANO
        capability = ComputeCapability.EDGE_AI if resource_level == "high" else ComputeCapability.EDGE_AI

        gpu_mem = 8.0 if local_platform == HardwarePlatform.JETSON_ORIN else 2.0

        self.local_capabilities = HardwareCapabilities(
            platform=local_platform,
            compute_capability=capability,
            cpu_cores=int(self.resources['cpu_capacity']),
            cpu_freq=2.0,
            memory_total=float(self.resources['memory_capacity']),
            gpu_memory=gpu_mem,
            npu_memory=0.0,
            storage_speed='emmc',
            network_speed=1000.0,
            power_profile='balanced',
            thermal_limit=85.0,
            accelerated_compute=True
        )
        self.platform_bridge = PlatformBridge(self.local_capabilities,
                                               node_id=self.vehicle_id)

        # Training state
        self.current_pipeline_id: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.training_active = False
        self.current_round = 0

        # Statistics
        self.stats = {
            'training_sessions': 0,
            'epochs_completed': 0,
            'batches_processed': 0,
            'updates_sent': 0
        }

        # Lock for thread safety
        self.lock = threading.Lock()

    def start(self):
        """Start vehicle"""
        print("=" * 60)
        print(f"Starting Vehicle: {self.vehicle_id}")
        print("=" * 60)
        print(f"Server: {self.server_host}:{self.server_port}")
        print(f"Resources: {self.resources}")

        # Connect to server
        print("Connecting to server...")
        self._connect_to_server()

        # Register message handlers
        print("Registering network handlers...")
        self._register_message_handlers()

        # Register with server
        print("Registering with server...")
        self._register_with_server()

        print("✓ Vehicle started and registered")
        return True

    def stop(self):
        """Stop vehicle"""
        print(f"\nStopping vehicle {self.vehicle_id}...")

        self.training_active = False

        print("✓ Vehicle stopped")

    def _connect_to_server(self):
        """Connect to server using cross-platform communication"""
        # Add route to server
        from fhdp.core.hardware_adapter import ComputeCapability

        remote_capabilities = HardwareCapabilities(
            platform=HardwarePlatform.X86_LINUX,
            compute_capability=ComputeCapability.SERVER_CLASS,
            cpu_cores=32,
            cpu_freq=3.5,
            memory_total=64.0,
            gpu_memory=24.0,
            npu_memory=0.0,
            storage_speed='ssd',
            network_speed=1000.0,
            power_profile='high_performance',
            thermal_limit=95.0,
            accelerated_compute=True
        )

        success = self.platform_bridge.connect_to_platform(
            remote_node_id="server",
            remote_capabilities=remote_capabilities,
            network_endpoint=self.server_endpoint
        )

        if success:
            print(f"✓ Connected to server {self.server_host}:{self.server_port}")
        else:
            print(f"✗ Failed to connect to server")
            raise ConnectionError("Cannot connect to server")

    def _register_message_handlers(self):
        """Register message handlers for cross-platform communication"""

        self.platform_bridge.message_router.register_handler(
            'pipeline_invite',
            self._handle_pipeline_invitation
        )

        self.platform_bridge.message_router.register_handler(
            'global_model',
            self._handle_global_model
        )

        self.platform_bridge.message_router.register_handler(
            'status_response',
            self._handle_status
        )

    def _register_with_server(self):
        """Register vehicle with server"""
        print(f"Sending registration message...")

        registration_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="register",
            payload={
                'position': self.position,
                'velocity': 0.0,
                'direction': 0.0,
                'resources': self.resources
            },
            metadata={
                'client_host': socket.gethostbyname(socket.gethostname()),
                'client_port': 0
            }
        )

        success = self.platform_bridge.send_cross_platform_message(registration_msg)
        if success:
            print("✓ Registration sent to server")
        else:
            print("✗ Failed to send registration message")

    def _handle_pipeline_invitation(self, message: CrossPlatformMessage):
        """Handle pipeline invitation from server"""
        pipeline_data = message.payload

        print(f"\nReceived pipeline invitation from server")
        print(f"  Pipeline ID: {pipeline_data['pipeline_id']}")
        print(f"  Template ID: {pipeline_data.get('template_id', 'N/A')}")
        print(f"  Vehicles in pipeline: {pipeline_data['vehicles']}")
        print(f"  Stages: {pipeline_data['stages']}")

        # Find which stage this vehicle is assigned to
        if self.vehicle_id in pipeline_data['vehicles']:
            stage_index = pipeline_data['vehicles'].index(self.vehicle_id)
            self.current_pipeline_id = pipeline_data['pipeline_id']
            self.current_stage = pipeline_data['stages'][stage_index]

            # Get resource requirement for this stage
            resource_requirements = pipeline_data.get('resource_requirements', [])
            if resource_requirements and stage_index < len(resource_requirements):
                required_class = resource_requirements[stage_index]
                print(f"  Assigned stage: {self.current_stage} (requires: {required_class})")
            else:
                print(f"  Assigned stage: {self.current_stage}")

            # Accept invitation
            response = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id=self.vehicle_id,
                target_id="server",
                message_type="pipeline_response",
                payload={
                    'pipeline_id': pipeline_data['pipeline_id'],
                    'accepted': True,
                    'stage': self.current_stage,
                    'stage_index': stage_index
                }
            )

            success = self.platform_bridge.send_cross_platform_message(response)
            if success:
                print(f"✓ Accepted pipeline invitation")
        else:
            print(f"✗ Vehicle {self.vehicle_id} not in selected pipeline vehicles")
            response = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id=self.vehicle_id,
                target_id="server",
                message_type="pipeline_response",
                payload={
                    'pipeline_id': pipeline_data['pipeline_id'],
                    'accepted': False,
                    'reason': 'Not in vehicle list'
                }
            )
            self.platform_bridge.send_cross_platform_message(response)

    def _handle_global_model(self, message: CrossPlatformMessage):
        """Handle global model broadcast from server"""
        model_data = message.payload
        round_num = model_data['round']
        training_config = model_data['training_config']

        print(f"\nReceived global model for round {round_num}")

        # Update local model (deserialize list→Tensor after JSON transport)
        self.model.load_state_dict(_deserialize_state_dict(model_data['model_state']))

        # Start training
        self._train_locally(round_num, training_config)

    def _handle_status(self, message: CrossPlatformMessage):
        """Handle status response"""
        status = message.payload
        print("\nServer Status:")
        print(f"  Pipeline formed: {status.get('pipeline_formed')}")
        print(f"  Active pipeline: {status.get('active_pipeline')}")
        print(f"  Statistics: {status.get('stats')}")

    def _train_locally(self, round_num: int, config: Dict[str, Any]):
        """Train model locally"""
        print(f"→ Starting local training for round {round_num}...")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config['learning_rate']

        # Create data loader
        train_loader = create_mock_data_loader(
            num_samples=200,
            batch_size=config['batch_size']
        )

        # Training loop
        self.model.train()
        epoch_losses = []

        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                with self.lock:
                    self.stats['batches_processed'] += 1

                # Progress indicator
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}/{config['epochs']}, "
                          f"Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_losses.append(avg_epoch_loss)

            print(f"  Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")

            with self.lock:
                self.stats['epochs_completed'] += 1

        with self.lock:
            self.stats['training_sessions'] += 1

        print(f"✓ Local training completed for round {round_num}")
        print(f"  Average loss: {np.mean(epoch_losses):.4f}")

        # Send model update to server
        self._send_model_update(round_num, epoch_losses)

    def _send_model_update(self, round_num: int, epoch_losses: List[float]):
        """Send model update to server"""
        print(f"→ Sending model update for round {round_num}...")

        # Get model state dict (serialize Tensor→list for JSON transport)
        model_state = _serialize_state_dict(self.model.state_dict())

        update_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="model_update",
            payload={
                'round': round_num,
                'model_state': model_state,
                'loss': float(np.mean(epoch_losses)),
                'timestamp': time.time()
            }
        )

        success = self.platform_bridge.send_cross_platform_message(update_msg)

        if success:
            with self.lock:
                self.stats['updates_sent'] += 1
            print(f"✓ Model update sent for round {round_num}")
        else:
            print(f"✗ Failed to send model update for round {round_num}")

    def send_heartbeat(self):
        """Send heartbeat to server"""
        heartbeat_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="heartbeat",
            payload={'timestamp': time.time()}
        )
        self.platform_bridge.send_cross_platform_message(heartbeat_msg)

    def request_status(self):
        """Request server status"""
        status_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="status",
            payload={}
        )
        self.platform_bridge.send_cross_platform_message(status_msg)


# ==================== Main ====================

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n\nReceived interrupt signal, shutting down...")
    global server_instance, vehicle_instance

    if 'server_instance' in globals() and server_instance:
        server_instance.stop()
    if 'vehicle_instance' in globals() and vehicle_instance:
        vehicle_instance.stop()

    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='FHDP Pipeline Training Test (Refactored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on 4090 Linux machine:
  python test_pipeline_training_refactored.py --mode server --host 0.0.0.0 --port 5000

  # Start vehicle on Jetson AGX Orin:
  python test_pipeline_training_refactored.py --mode vehicle --vehicle-id agx_orin_001 \\
      --server-host <server-ip> --server-port 5000 --resource-level high

  # Start vehicle on Jetson Orin Nano:
  python test_pipeline_training_refactored.py --mode vehicle --vehicle-id orin_nano_001 \\
      --server-host <server-ip> --server-port 5000 --resource-level medium
        """
    )

    parser.add_argument('--mode', required=True, choices=['server', 'vehicle'],
                        help='Operation mode: server or vehicle')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host address for server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number for server')
    parser.add_argument('--vehicle-id', default='vehicle_001',
                        help='Vehicle ID (vehicle mode only)')
    parser.add_argument('--server-host', default='localhost',
                        help='Server host address (vehicle mode only)')
    parser.add_argument('--server-port', type=int, default=5000,
                        help='Server port (vehicle mode only)')
    parser.add_argument('--resource-level', default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Resource level (vehicle mode only)')
    parser.add_argument('--config', help='Path to configuration file')

    args = parser.parse_args()

    # Set signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global server_instance, vehicle_instance

    try:
        if args.mode == 'server':
            # Server mode
            server_instance = PipelineTestServer(
                host=args.host,
                port=args.port,
                config_path=args.config
            )
            server_instance.start()

            # Keep server running
            while True:
                time.sleep(1)

        else:
            # Vehicle mode
            vehicle_instance = PipelineTestVehicle(
                vehicle_id=args.vehicle_id,
                server_host=args.server_host,
                server_port=args.server_port,
                resource_level=args.resource_level
            )

            if vehicle_instance.start():
                # Send periodic heartbeats
                try:
                    while True:
                        time.sleep(10.0)
                        vehicle_instance.send_heartbeat()
                except KeyboardInterrupt:
                    pass

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
