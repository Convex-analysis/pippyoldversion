#!/usr/bin/env python3
"""
FHDP Pipeline Training Test Script

This script tests the FHDP system's capability to orchestrate pipeline training
across two Jetson devices (AGX Orin and Orin Nano) with a 4090 Linux server.

Architecture:
- Server: Edge Server (4090 Linux) - Coordinates pipeline formation and aggregation
- Client 1: Jetson AGX Orin - High-resource vehicle
- Client 2: Jetson Orin Nano - Medium-resource vehicle

Usage:
    # On the 4090 server:
    python test_pipeline_training.py --mode server --host 0.0.0.0 --port 5000

    # On Jetson AGX Orin:
    python test_pipeline_training.py --mode vehicle --vehicle-id agx_orin_001 --server-host <server-ip> --server-port 5000 --resource-level high

    # On Jetson Orin Nano:
    python test_pipeline_training.py --mode vehicle --vehicle-id orin_nano_001 --server-host <server-ip> --server-port 5000 --resource-level medium
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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
# Script location: pippyoldversion/fhdp/tests/pipeline_test/test_pipeline_training.py
# Project root should be: pippyoldversion/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add both project root and its parent to path for development mode
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)

from fhdp.core.types import (
    VehicleInfo, VehicleState, TrainingMode, Pipeline, PipelineTemplate,
    ModelUpdate, AggregationResult, ResourceClass, TrainingConfig
)
from fhdp.vehicle_layer.training_engine import TrainingTask
from fhdp.core.fhdp_system import FHDPSystem, SystemConfiguration
from fhdp.edge_server.server import EdgeServer
from fhdp.vehicle_layer.vehicle import Vehicle


# ==================== Network Communication ====================

class MessageType(Enum):
    """Message types for network communication"""
    REGISTER = "register"
    UNREGISTER = "unregister"
    MODEL_UPDATE = "model_update"
    GLOBAL_MODEL = "global_model"
    PIPELINE_INVITE = "pipeline_invite"
    PIPELINE_RESPONSE = "pipeline_response"
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    SHUTDOWN = "shutdown"


@dataclass
class NetworkMessage:
    """Network message structure"""
    msg_type: MessageType
    sender_id: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class NetworkServer:
    """Network server for edge server"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_connections: Dict[str, socket.socket] = {}
        self.running = False
        self.message_handlers: Dict[MessageType, callable] = {}

    def start(self):
        """Start network server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            print(f"✓ Network server listening on {self.host}:{self.port}")

            # Verify binding
            actual_address = self.server_socket.getsockname()
            print(f"  Actual bound address: {actual_address}")

        except Exception as e:
            print(f"✗ Failed to bind socket: {e}")
            raise

        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()

    def stop(self):
        """Stop network server"""
        self.running = False

        # Close all client connections
        for conn in self.client_connections.values():
            try:
                conn.close()
            except:
                pass

        # Close server socket
        if self.server_socket:
            self.server_socket.close()

    def _accept_connections(self):
        """Accept incoming connections"""
        print("Accept thread started, waiting for connections...")
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"✓ Accepted connection from {address}")

                    # Start client handler thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue

            except Exception as e:
                if self.running:
                    print(f"✗ Accept error: {e}")

        print("Accept thread stopped")

    def _handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        vehicle_id = None

        try:
            while self.running:
                # Receive message
                data = client_socket.recv(4096 * 10)  # 40KB buffer
                if not data:
                    break

                # Deserialize message
                import pickle
                try:
                    message = pickle.loads(data)
                    vehicle_id = message.sender_id

                    # Store connection
                    if vehicle_id:
                        self.client_connections[vehicle_id] = client_socket

                    # Handle message
                    handler = self.message_handlers.get(message.msg_type)
                    if handler:
                        handler(message, client_socket)

                except Exception as e:
                    print(f"Message handling error: {e}")

        except Exception as e:
            print(f"Client handler error: {e}")

        finally:
            # Cleanup
            if vehicle_id and vehicle_id in self.client_connections:
                del self.client_connections[vehicle_id]
            client_socket.close()
            print(f"Client {vehicle_id} disconnected")

    def register_handler(self, msg_type: MessageType, handler: callable):
        """Register message handler"""
        self.message_handlers[msg_type] = handler

    def send_message(self, vehicle_id: str, message: NetworkMessage):
        """Send message to specific vehicle"""
        if vehicle_id in self.client_connections:
            conn = self.client_connections[vehicle_id]
            import pickle
            try:
                conn.sendall(pickle.dumps(message))
            except Exception as e:
                print(f"Send error to {vehicle_id}: {e}")

    def broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected vehicles"""
        import pickle
        data = pickle.dumps(message)
        for vehicle_id, conn in self.client_connections.items():
            try:
                conn.sendall(data)
            except Exception as e:
                print(f"Broadcast error to {vehicle_id}: {e}")


class NetworkClient:
    """Network client for vehicles"""

    def __init__(self, server_host: str, server_port: int, vehicle_id: str):
        self.server_host = server_host
        self.server_port = server_port
        self.vehicle_id = vehicle_id
        self.server_socket = None
        self.connected = False
        self.running = False
        self.message_handlers: Dict[MessageType, callable] = {}
        self.lock = threading.Lock()

    def connect(self, max_retries: int = 10, retry_interval: float = 2.0):
        """Connect to server"""
        print(f"Attempting to connect to {self.server_host}:{self.server_port}...")
        for attempt in range(max_retries):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # Set a shorter timeout for individual connection attempts
                self.server_socket.settimeout(5.0)

                print(f"  Attempt {attempt + 1}/{max_retries}...", flush=True)
                self.server_socket.connect((self.server_host, self.server_port))
                self.connected = True

                # Reset timeout to blocking after connection
                self.server_socket.settimeout(None)

                print(f"✓ Connected to server {self.server_host}:{self.server_port}")
                return True

            except socket.timeout:
                print(f"  Connection timeout on attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                print(f"  Waiting {retry_interval}s before retry...", flush=True)
                time.sleep(retry_interval)

        print(f"✗ Failed to connect to server after {max_retries} attempts")
        return False

    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        self.connected = False

        if self.server_socket:
            self.server_socket.close()

    def start_receiving(self):
        """Start receiving messages from server"""
        self.running = True
        receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
        receive_thread.start()

    def _receive_messages(self):
        """Receive messages from server"""
        while self.running and self.connected:
            try:
                self.server_socket.settimeout(1.0)
                data = self.server_socket.recv(4096 * 10)
                if not data:
                    break

                import pickle
                message = pickle.loads(data)

                # Handle message
                handler = self.message_handlers.get(message.msg_type)
                if handler:
                    handler(message)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break

    def send_message(self, message: NetworkMessage):
        """Send message to server"""
        if not self.connected:
            return False

        try:
            with self.lock:
                import pickle
                self.server_socket.sendall(pickle.dumps(message))
            return True

        except Exception as e:
            print(f"Send error: {e}")
            return False

    def register_handler(self, msg_type: MessageType, handler: callable):
        """Register message handler"""
        self.message_handlers[msg_type] = handler


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
        # Assuming 28x28 input: -> 14x14 -> 7x7
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
    """Server mode for pipeline training test"""

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
        self.network_server = NetworkServer(host, port)

        # Initialize FHDP components for pipeline formation
        from fhdp.edge_server.resource_classifier import ResourceClassifier
        from fhdp.edge_server.template_manager import TemplateManager
        from fhdp.vehicle_layer.pipeline_formation import PipelineFormation
        from fhdp.core.types import VehicleInfo, ResourceClass, TrainingConfig

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

        # Start network server
        self.network_server.start()
        print("✓ Network server started")

        # Register network handlers
        self._register_network_handlers()

        print("\nServer is ready to accept vehicle connections...")
        print(f"Listen on: {self.host}:{self.port}\n")

    def stop(self):
        """Stop server"""
        print("\nStopping server...")

        self.network_server.stop()
        self.edge_server.stop_server()
        self.fhdp_system.stop_system()

        print("✓ Server stopped")

    def _register_network_handlers(self):
        """Register network message handlers"""

        # Vehicle registration
        self.network_server.register_handler(
            MessageType.REGISTER,
            self._handle_register
        )

        # Model update
        self.network_server.register_handler(
            MessageType.MODEL_UPDATE,
            self._handle_model_update
        )

        # Heartbeat
        self.network_server.register_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )

        # Pipeline response
        self.network_server.register_handler(
            MessageType.PIPELINE_RESPONSE,
            self._handle_pipeline_response
        )

        # Status request
        self.network_server.register_handler(
            MessageType.STATUS,
            self._handle_status
        )

    def _handle_register(self, message: NetworkMessage, client_socket: socket.socket):
        """Handle vehicle registration"""
        vehicle_data = message.data

        # Create vehicle info
        vehicle_info = VehicleInfo(
            vehicle_id=message.sender_id,
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
            print(f"✓ Vehicle {message.sender_id} registered (Resource Class: {resource_class.value})")
            print(f"  Position: {vehicle_info.position}")
            print(f"  Resources: {vehicle_info.resources}")

            with self.lock:
                self.stats['vehicles_registered'] += 1

            # Check if we have enough vehicles for pipeline
            if self.stats['vehicles_registered'] >= 2 and not self.pipeline_formed:
                self._try_form_pipeline()

    def _handle_model_update(self, message: NetworkMessage, client_socket: socket.socket):
        """Handle model update from vehicle"""
        with self.lock:
            self.stats['total_updates_received'] += 1

        update_data = message.data
        source_id = message.sender_id
        round_num = update_data.get('round', 0)

        print(f"[Round {round_num}] Received update from {source_id}")

        # Check if we have updates from all pipeline vehicles
        if self.active_pipeline:
            pipeline_vehicles = set(self.active_pipeline.vehicles)
            updates_in_round = update_data.get('updates_in_round', [])

            if set(updates_in_round) == pipeline_vehicles:
                # Aggregate updates
                self._aggregate_model_updates(round_num, update_data.get('updates'))

    def _handle_heartbeat(self, message: NetworkMessage, client_socket: socket.socket):
        """Handle heartbeat from vehicle"""
        # Just acknowledge - keep connection alive
        pass

    def _handle_pipeline_response(self, message: NetworkMessage, client_socket: socket.socket):
        """Handle pipeline invitation response"""
        response = message.data
        accepted = response.get('accepted', False)

        if accepted:
            print(f"✓ Vehicle {message.sender_id} accepted pipeline invitation")

            # Check if all vehicles have accepted
            if self.active_pipeline:
                if len(self.active_pipeline.vehicles) >= 2:
                    self._start_pipeline_training()
        else:
            print(f"✗ Vehicle {message.sender_id} declined pipeline invitation")

    def _handle_status(self, message: NetworkMessage, client_socket: socket.socket):
        """Handle status request"""
        status = self.get_status()
        response_msg = NetworkMessage(
            msg_type=MessageType.STATUS,
            sender_id="server",
            data=status
        )
        self.network_server.send_message(message.sender_id, response_msg)

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
            invitation = NetworkMessage(
                msg_type=MessageType.PIPELINE_INVITE,
                sender_id="server",
                data={
                    'pipeline_id': pipeline_id,
                    'template_id': template.template_id,
                    'vehicles': self.active_pipeline.vehicles,
                    'stages': self.active_pipeline.stages,
                    'resource_requirements': [r.value for r in template.resource_requirements]
                }
            )

            for vehicle_id in self.active_pipeline.vehicles:
                self.network_server.send_message(vehicle_id, invitation)
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
        global_model_state = self.global_model.state_dict()

        broadcast_msg = NetworkMessage(
            msg_type=MessageType.GLOBAL_MODEL,
            sender_id="server",
            data={
                'round': round_num,
                'model_state': global_model_state,
                'training_config': {
                    'epochs': 2,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            }
        )

        self.network_server.broadcast_message(broadcast_msg)
        print(f"✓ Broadcast global model for round {round_num}")

    def _aggregate_model_updates(self, round_num: int, updates: List[Dict]):
        """Aggregate model updates from all vehicles"""
        print(f"\n→ Aggregating updates for round {round_num}...")

        # Simple averaging aggregation
        aggregated_state = {}
        num_updates = len(updates)

        for key in updates[0].keys():
            # Average parameters
            aggregated_state[key] = torch.mean(
                torch.stack([update[key] for update in updates]),
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
        # In real implementation, this would evaluate on test data
        # For testing, we just print a mock accuracy
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
    """Vehicle mode for pipeline training test"""

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

        # Network client
        self.network_client = NetworkClient(server_host, server_port, vehicle_id)

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
        if not self.network_client.connect():
            print("✗ Failed to connect to server")
            return False

        # Start receiving messages
        print("Starting message receiver...")
        self.network_client.start_receiving()

        # Register network handlers
        print("Registering network handlers...")
        self._register_network_handlers()

        # Register with server
        print("Registering with server...")
        self._register_with_server()

        print("✓ Vehicle started and registered")
        return True

    def stop(self):
        """Stop vehicle"""
        print(f"\nStopping vehicle {self.vehicle_id}...")

        self.training_active = False
        self.network_client.disconnect()

        print("✓ Vehicle stopped")

    def _register_network_handlers(self):
        """Register network message handlers"""

        # Pipeline invitation
        self.network_client.register_handler(
            MessageType.PIPELINE_INVITE,
            self._handle_pipeline_invitation
        )

        # Global model
        self.network_client.register_handler(
            MessageType.GLOBAL_MODEL,
            self._handle_global_model
        )

        # Status
        self.network_client.register_handler(
            MessageType.STATUS,
            self._handle_status
        )

    def _register_with_server(self):
        """Register vehicle with server"""
        print(f"Sending registration message...")
        registration_msg = NetworkMessage(
            msg_type=MessageType.REGISTER,
            sender_id=self.vehicle_id,
            data={
                'position': self.position,
                'velocity': 0.0,
                'direction': 0.0,
                'resources': self.resources
            }
        )

        success = self.network_client.send_message(registration_msg)
        if success:
            print("✓ Registration sent to server")
        else:
            print("✗ Failed to send registration message")

    def _handle_pipeline_invitation(self, message: NetworkMessage):
        """Handle pipeline invitation from server"""
        pipeline_data = message.data

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
            response = NetworkMessage(
                msg_type=MessageType.PIPELINE_RESPONSE,
                sender_id=self.vehicle_id,
                data={
                    'pipeline_id': pipeline_data['pipeline_id'],
                    'accepted': True,
                    'stage': self.current_stage,
                    'stage_index': stage_index
                }
            )

            success = self.network_client.send_message(response)
            if success:
                print(f"✓ Accepted pipeline invitation")
        else:
            print(f"✗ Vehicle {self.vehicle_id} not in selected pipeline vehicles")
            response = NetworkMessage(
                msg_type=MessageType.PIPELINE_RESPONSE,
                sender_id=self.vehicle_id,
                data={
                    'pipeline_id': pipeline_data['pipeline_id'],
                    'accepted': False,
                    'reason': 'Not in vehicle list'
                }
            )
            self.network_client.send_message(response)

    def _handle_global_model(self, message: NetworkMessage):
        """Handle global model broadcast from server"""
        model_data = message.data
        round_num = model_data['round']
        training_config = model_data['training_config']

        print(f"\nReceived global model for round {round_num}")

        # Update local model
        self.model.load_state_dict(model_data['model_state'])

        # Start training
        self._train_locally(round_num, training_config)

    def _handle_status(self, message: NetworkMessage):
        """Handle status response"""
        status = message.data
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

        # Get model state dict
        model_state = self.model.state_dict()

        update_msg = NetworkMessage(
            msg_type=MessageType.MODEL_UPDATE,
            sender_id=self.vehicle_id,
            data={
                'round': round_num,
                'model_state': model_state,
                'loss': np.mean(epoch_losses),
                'timestamp': time.time()
            }
        )

        success = self.network_client.send_message(update_msg)

        if success:
            with self.lock:
                self.stats['updates_sent'] += 1
            print(f"✓ Model update sent for round {round_num}")
        else:
            print(f"✗ Failed to send model update for round {round_num}")

    def send_heartbeat(self):
        """Send heartbeat to server"""
        heartbeat_msg = NetworkMessage(
            msg_type=MessageType.HEARTBEAT,
            sender_id=self.vehicle_id,
            data={'timestamp': time.time()}
        )
        self.network_client.send_message(heartbeat_msg)

    def request_status(self):
        """Request server status"""
        status_msg = NetworkMessage(
            msg_type=MessageType.STATUS,
            sender_id=self.vehicle_id,
            data={}
        )
        self.network_client.send_message(status_msg)


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
        description='FHDP Pipeline Training Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on 4090 Linux machine:
  python test_pipeline_training.py --mode server --host 0.0.0.0 --port 5000

  # Start vehicle on Jetson AGX Orin:
  python test_pipeline_training.py --mode vehicle --vehicle-id agx_orin_001 \\
      --server-host <server-ip> --server-port 5000 --resource-level high

  # Start vehicle on Jetson Orin Nano:
  python test_pipeline_training.py --mode vehicle --vehicle-id orin_nano_001 \\
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
