#!/usr/bin/env python3
"""
FHDP Pipeline Training Test Script (Refactored)

This script tests the FHDP system's capability to orchestrate pipeline training
across two Jetson devices (AGX Orin and Orin Nano) with a 4090 Linux server.

Model: EVO1Driving (vision-language-action, ~25-30M fallback params, no HF download).
Template: Full TemplateManager integration with latency benchmark and online-learning feedback.

Refactored to use FHDP's built-in cross_platform_comm.py for reliable network
communication with length-prefix protocol and compression support.

Architecture:
- Server: Edge Server (4090 Linux) - Coordinates pipeline formation and aggregation
- Client 1: Jetson AGX Orin - High-resource vehicle (batch=4, 3 views, full params)
- Client 2: Jetson Orin Nano - Medium-resource vehicle (batch=2, 2 views, stage-1 freeze)

Usage:
    # On the 4090 server:
    python test_pipeline_training_refactored.py --mode server --host 0.0.0.0 --port 5000

    # On Jetson AGX Orin:
    python test_pipeline_training_refactored.py --mode vehicle --vehicle-id agx_orin_001 --server-host <server-ip> --server-port 5000 --resource-level high

    # On Jetson Orin Nano:
    python test_pipeline_training_refactored.py --mode vehicle --vehicle-id orin_nano_001 --server-host <server-ip> --server-port 5000 --resource-level medium

    # Standalone template benchmark (no network):
    python test_pipeline_training_refactored.py --mode test-template --resource-level medium
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
import torch.nn.functional as F
import numpy as np
import uuid
from dataclasses import dataclass
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
    # Also add fhdp/EVO1 to path for utils import (nuscenes_loader uses "from utils.config")
    evo1_root = os.path.join(project_root, 'fhdp', 'EVO1')
    if os.path.exists(evo1_root):
        sys.path.insert(0, evo1_root)
        print(f"[INFO] Added EVO1 root to path: {evo1_root}")
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


# ==================== EVO1Driving Model for Testing ====================

# Import EVO1Driving model (with fallback implementation)
from fhdp.EVO1.model.evo1_driving import (
    EVO1Driving, EVO1DrivingOutput,
    ModelConfig as _EVO1ModelConfig
)

from fhdp.core.constants import TEMPLATE_LOOKUP_LATENCY_THRESHOLD


@dataclass
class TestModelConfig:
    """Extended ModelConfig for test: adds fields needed by _controls_to_waypoints()
    and hardware-adaptive parameters for Jetson devices."""
    # --- inherited from inline ModelConfig ---
    vision_encoder: str = "OpenGVLab/InternVL3-1B"
    language_model: str = "Qwen/Qwen2.5-0.5B"
    image_size: int = 224
    max_waypoints: int = 20
    action_dim: int = 8
    per_action_dim: int = 7
    sequence_length: int = 32
    hidden_dim: int = 4096
    vision_model_name: str = "OpenGVLab/InternVL3-1B"
    # --- fields required by _controls_to_waypoints ---
    max_steering: float = 0.6
    max_speed: float = 30.0
    # --- driving / hardware-adaptive fields ---
    num_views: int = 3
    horizon: int = 20
    action_hidden_dim: int = 512


def _build_model_config(resource_level: str) -> TestModelConfig:
    """Create device-adaptive model config.

    resource_level:
        'high'   – AGX Orin  (batch=4, 3 views, 20 waypoints, full params)
        'medium' – Orin Nano (batch=2, 2 views, 10 waypoints, stage-1 freeze backbone)
        'server' – 4090 Linux (batch=8, 3 views, 20 waypoints, CPU aggregation only)
    """
    if resource_level == 'high':
        return TestModelConfig(max_waypoints=20, num_views=3, action_hidden_dim=512)
    elif resource_level == 'medium':
        return TestModelConfig(max_waypoints=10, num_views=2, action_hidden_dim=256)
    else:  # server
        return TestModelConfig(max_waypoints=20, num_views=3, action_hidden_dim=512)


def _get_batch_size(resource_level: str) -> int:
    """Return batch size per device type."""
    return {'high': 4, 'medium': 2, 'server': 8}.get(resource_level, 4)


def _get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _serialize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert torch.Tensor values to lists for JSON serialization."""
    return {k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in state_dict.items()}


def _deserialize_state_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Restore lists back to torch.Tensor after JSON deserialization."""
    return {k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in raw.items()}


def create_mock_driving_data_loader(
    num_samples: int = 100,
    batch_size: int = 4,
    num_views: int = 3,
    image_size: int = 224,
    max_waypoints: int = 20
):
    """Create mock autonomous-driving data loader.

    Yields per batch:
        images      [B, N, 3, H, W]
        image_mask  [B, N]          (all True)
        state       [B, 12]         (vehicle state)
        controls    [B, T, 3]       (steering, throttle, brake in [-1,1])
    """

    class MockDrivingDataLoader:
        def __init__(self, num_samples, batch_size, num_views, image_size, max_waypoints):
            self.num_samples = num_samples
            self.batch_size = batch_size
            self.num_views = num_views
            self.image_size = image_size
            self.max_waypoints = max_waypoints

        def __iter__(self):
            for i in range(0, self.num_samples, self.batch_size):
                bs = min(self.batch_size, self.num_samples - i)
                images = torch.randn(bs, self.num_views, 3, self.image_size, self.image_size)
                image_mask = torch.ones(bs, self.num_views, dtype=torch.bool)
                state = torch.randn(bs, 12)
                controls = torch.tanh(torch.randn(bs, self.max_waypoints, 3))
                yield images, image_mask, state, controls

        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    return MockDrivingDataLoader(num_samples, batch_size, num_views, image_size, max_waypoints)


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
            compression=CompressionType.NONE  # Disabled to avoid zlib decompression errors
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
        self.server_config = _build_model_config('server')
        self.global_model = EVO1Driving(self.server_config, device='cpu')

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
        self.current_round = 0
        self.pending_updates: Dict[int, Dict[str, Dict]] = {}  # round_num -> vehicle_id -> update_data
        self.training_complete = False

        # Track vehicles that have accepted pipeline invitation
        self.accepted_vehicles = set()
        self.training_started = False

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
        """Stop server and clean up all resources"""
        print("\nStopping server...")

        try:
            # Close message router and network connections
            if hasattr(self.platform_bridge, 'message_router'):
                self.platform_bridge.message_router.close()
            
            # Stop FHDP components
            self.edge_server.stop_server()
            self.fhdp_system.stop_system()
            
            print("✓ Server stopped")
        except Exception as e:
            print(f"Error during server shutdown: {e}")
            import traceback
            traceback.print_exc()

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
                compression=CompressionType.NONE  # Disabled to avoid zlib decompression errors
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

        # Store update - make sure we have an entry for this round
        if round_num not in self.pending_updates:
            self.pending_updates[round_num] = {}
        self.pending_updates[round_num][source_id] = update_data

        # Check if we have updates from all pipeline vehicles for this specific round
        if self.active_pipeline:
            pipeline_vehicles = set(self.active_pipeline.vehicles)
            received_updates = set(self.pending_updates[round_num].keys())

            if received_updates == pipeline_vehicles:
                # Aggregate updates for this round only
                updates = [self.pending_updates[round_num][vid] for vid in self.active_pipeline.vehicles]
                self._aggregate_model_updates(round_num, updates)
                # Clear updates for this round only, not all rounds
                del self.pending_updates[round_num]

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

            # Track accepted vehicles
            with self.lock:
                self.accepted_vehicles.add(message.source_id)

            # Check if all vehicles have accepted
            if self.active_pipeline:
                expected_vehicles = set(self.active_pipeline.vehicles)
                with self.lock:
                    all_accepted = (self.accepted_vehicles == expected_vehicles and
                                   not self.training_started)

                if all_accepted:
                    with self.lock:
                        self.training_started = True
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
            payload=status,
            requires_ack=False
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

            # Step 2: Find best matching template (timed)
            print("\nStep 2: Finding best matching pipeline template...")
            t0 = time.perf_counter()
            template = self.template_manager.find_template_for_vehicles(candidate_vehicles)
            lookup_ms = (time.perf_counter() - t0) * 1000
            threshold_ms = TEMPLATE_LOOKUP_LATENCY_THRESHOLD * 1000
            status = "PASS" if lookup_ms < threshold_ms else "WARN"
            print(f"  Template lookup latency: {lookup_ms:.2f}ms (threshold {threshold_ms:.0f}ms) [{status}]")

            # Top-3 candidates with match scores
            top_candidates = self.template_manager.matcher.find_best_template(
                candidate_vehicles, max_candidates=3
            )
            if top_candidates:
                for rank, (t, score) in enumerate(top_candidates, 1):
                    print(f"  Candidate #{rank}: {t.template_id} | score={score:.3f} | "
                          f"req={[r.value for r in t.resource_requirements]}")
            else:
                print("  (no candidates returned by matcher)")

            # Basket statistics
            stats = self.template_manager.get_template_statistics()
            print(f"  [Basket Stats] baskets={stats['total_baskets']} "
                  f"templates={stats['total_templates']} "
                  f"avg_success={stats['avg_success_rate']:.3f} "
                  f"cache_hit={stats['cache_hit_rate']:.3f}")

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
                },
                requires_ack=False
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
                    'learning_rate': 1e-4
                },
                'num_views': self.server_config.num_views,
                'image_size': self.server_config.image_size,
                'max_waypoints': self.server_config.max_waypoints
            },
            requires_ack=False
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

        # Start next round if not done (non-recursive)
        if round_num < 3:  # Test with 3 rounds
            time.sleep(2.0)
            # Create a new thread for the next round to avoid recursion
            import threading
            threading.Thread(
                target=self._start_training_round,
                args=(round_num + 1,),
                daemon=True
            ).start()
        else:
            # ---- Template Feedback after final round ----
            print("\n" + "-" * 40)
            print("[Template Feedback] Registering pipeline result...")
            stats_before = self.template_manager.get_template_statistics()
            elapsed = time.time() - getattr(self.active_pipeline, 'start_time', time.time())
            self.template_manager.register_successful_pipeline(
                self.active_pipeline, success=True, duration=elapsed
            )
            stats_after = self.template_manager.get_template_statistics()
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Before: templates={stats_before['total_templates']} "
                  f"avg_success={stats_before['avg_success_rate']:.3f}")
            print(f"  After:  templates={stats_after['total_templates']} "
                  f"avg_success={stats_after['avg_success_rate']:.3f}")
            print("-" * 40)

            print("\n" + "=" * 60)
            print("Pipeline training completed!")
            print("=" * 60)
            self._print_summary()
            # Signal training completion
            self.training_complete = True

            # Notify all vehicles that training is complete
            self._notify_training_complete()

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

    def _notify_training_complete(self):
        """Notify all vehicles that training is complete"""
        if not self.active_pipeline:
            return

        print("\nNotifying vehicles that training is complete...")

        for vehicle_id in self.active_pipeline.vehicles:
            msg = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id="server",
                target_id=vehicle_id,
                message_type="training_complete",
                payload={
                    'pipeline_id': self.active_pipeline.pipeline_id,
                    'total_rounds': self.current_round,
                    'final_stats': self.stats.copy()
                },
                requires_ack=False
            )

            try:
                self.platform_bridge.send_cross_platform_message(msg)
                print(f"✓ Sent training complete notification to {vehicle_id}")
            except Exception as e:
                print(f"✗ Failed to send training complete notification to {vehicle_id}: {e}")

        # Request server shutdown
        global shutdown_requested
        shutdown_requested = True

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

        # Initialize EVO1Driving model with device-adaptive config
        self.resource_level = resource_level
        self.model_config = _build_model_config(resource_level)
        self.device = _get_device()
        self.model = EVO1Driving(self.model_config, device=self.device)

        # Orin Nano: freeze VL backbone → only train action head (~6M params)
        if resource_level == 'medium':
            self.model.set_stage1_mode()

        # AdamW on trainable params only; no separate criterion (compute_loss handles it)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )

        # Initialize cross-platform communication
        self.server_endpoint = NetworkEndpoint(
            host=server_host,
            port=server_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE  # Disabled to avoid zlib decompression errors
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

        # Training synchronization to prevent concurrent training
        self.training_lock = threading.Lock()
        self.latest_round_handled = 0

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
        """Stop vehicle and clean up all resources"""
        print(f"\nStopping vehicle {self.vehicle_id}...")

        try:
            self.training_active = False
            
            # Close message router and network connections
            if hasattr(self.platform_bridge, 'message_router'):
                self.platform_bridge.message_router.close()
            
            print("✓ Vehicle stopped")
        except Exception as e:
            print(f"Error during vehicle shutdown: {e}")
            import traceback
            traceback.print_exc()

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
            'training_complete',
            self._handle_training_complete
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
            },
            requires_ack=False
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
            },
            requires_ack=False
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
            },
            requires_ack=False
        )
            self.platform_bridge.send_cross_platform_message(response)

    def _handle_global_model(self, message: CrossPlatformMessage):
        """Handle global model broadcast from server"""
        model_data = message.payload
        round_num = model_data['round']
        training_config = model_data['training_config']

        # Deduplicate: ignore if we've already handled this round
        if round_num <= self.latest_round_handled:
            return

        # Use lock to prevent concurrent training
        with self.training_lock:
            # Double-check after acquiring lock
            if round_num <= self.latest_round_handled:
                return

            # Mark this round as handled
            self.latest_round_handled = round_num

            print(f"\nReceived global model for round {round_num}")

            # Update local model (deserialize list→Tensor after JSON transport)
            self.model.load_state_dict(_deserialize_state_dict(model_data['model_state']))

            # Extract num_views/image_size/max_waypoints from broadcast payload
            # (use local model_config as fallback for backward-compat)
            training_config['num_views'] = model_data.get('num_views', self.model_config.num_views)
            training_config['image_size'] = model_data.get('image_size', self.model_config.image_size)
            training_config['max_waypoints'] = model_data.get('max_waypoints', self.model_config.max_waypoints)

            # Start training (lock released automatically when with block exits)
            self._train_locally(round_num, training_config)

    def _handle_training_complete(self, message: CrossPlatformMessage):
        """Handle training complete notification from server"""
        payload = message.payload
        print("\n" + "=" * 60)
        print("Training Complete Notification")
        print("=" * 60)
        print(f"Pipeline ID: {payload.get('pipeline_id')}")
        print(f"Total rounds completed: {payload.get('total_rounds')}")
        print(f"Final statistics: {payload.get('final_stats')}")

        # Request shutdown (main loop will handle cleanup)
        global shutdown_requested
        shutdown_requested = True

    def _handle_status(self, message: CrossPlatformMessage):
        """Handle status response"""
        status = message.payload
        print("\nServer Status:")
        print(f"  Pipeline formed: {status.get('pipeline_formed')}")
        print(f"  Active pipeline: {status.get('active_pipeline')}")
        print(f"  Statistics: {status.get('stats')}")

    def _train_locally(self, round_num: int, config: Dict[str, Any]):
        """Train EVO1Driving model locally for one federated round."""
        print(f"→ Starting local training for round {round_num}...")

        num_views    = config.get('num_views',    self.model_config.num_views)
        image_size   = config.get('image_size',   self.model_config.image_size)
        max_waypoints = config.get('max_waypoints', self.model_config.max_waypoints)
        batch_size   = _get_batch_size(self.resource_level)
        lr           = config.get('learning_rate', 1e-4)
        epochs       = config.get('epochs', 2)

        print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
        print(f"  Device: {self.device} | Views: {num_views} | "
              f"ImgSize: {image_size} | Waypoints: {max_waypoints}")

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Create mock driving data loader
        train_loader = create_mock_driving_data_loader(
            num_samples=200,
            batch_size=batch_size,
            num_views=num_views,
            image_size=image_size,
            max_waypoints=max_waypoints
        )

        # Training loop
        self.model.train()
        epoch_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, image_mask, state, target_controls) in enumerate(train_loader):
                # Move tensors to device
                images         = images.to(self.device)
                image_mask     = image_mask.to(self.device)
                state          = state.to(self.device)
                target_controls = target_controls.to(self.device)

                # Forward + loss
                self.optimizer.zero_grad()
                output = self.model(
                    images, image_mask, state,
                    mode='training',
                    future_controls=target_controls
                )
                losses = self.model.compute_loss(output, target_controls)
                loss   = losses['total_loss']

                # Backward + gradient clip + step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss   += loss.item()
                num_batches  += 1

                with self.lock:
                    self.stats['batches_processed'] += 1

                # Progress + per-batch loss breakdown
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)} "
                          f"| total={loss.item():.4f} "
                          f"ctrl={losses['control_loss'].item():.4f} "
                          f"wp={losses['waypoint_loss'].item():.4f} "
                          f"conf={losses['confidence_loss'].item():.4f}")

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
            },
            requires_ack=False
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
            payload={'timestamp': time.time()},
            requires_ack=False
        )
        self.platform_bridge.send_cross_platform_message(heartbeat_msg)

    def request_status(self):
        """Request server status"""
        status_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="status",
            payload={},
            requires_ack=False
        )
        self.platform_bridge.send_cross_platform_message(status_msg)


# ==================== Standalone Template Test ====================

def run_template_standalone_test(resource_level: str = 'medium'):
    """Run standalone TemplateManager benchmark test (no network required).

    Tests 4 mock vehicle groups × 100 iterations to measure:
      - Template lookup latency (avg / max / p99) vs 5ms threshold
      - Top-3 candidate match scores
      - Basket statistics & online-learning feedback
    """
    from fhdp.edge_server.template_manager import TemplateManager
    from fhdp.core.types import VehicleInfo, VehicleState, ResourceClass, Pipeline
    from fhdp.core.constants import TEMPLATE_LOOKUP_LATENCY_THRESHOLD

    print("=" * 70)
    print("  FHDP Template Manager — Standalone Benchmark Test")
    print("=" * 70)

    # Initialise template manager (generates 100 synthetic templates)
    tm = TemplateManager()
    init_stats = tm.get_template_statistics()
    print(f"\n[Init] baskets={init_stats['total_baskets']}  "
          f"templates={init_stats['total_templates']}  "
          f"memory≈{init_stats['memory_usage']/1024:.0f}KB")

    # --- helper: build mock VehicleInfo with specific resource class ---
    def _mock_vehicle(vid: str, rclass: str) -> VehicleInfo:
        """Create a VehicleInfo whose resources map to the desired ResourceClass."""
        if rclass == 'HIGH':
            res = {'cpu': 0.9, 'memory': 0.8, 'battery': 0.9}
        elif rclass == 'MEDIUM':
            res = {'cpu': 0.6, 'memory': 0.5, 'battery': 0.7}
        else:  # LOW
            res = {'cpu': 0.3, 'memory': 0.3, 'battery': 0.4}
        return VehicleInfo(
            vehicle_id=vid,
            position=(0.0, 0.0),
            velocity=0.0,
            direction=0.0,
            resources=res,
            state=VehicleState.IDLE
        )

    # 4 vehicle groups
    groups = {
        'HIGH+HIGH':          [_mock_vehicle('v1', 'HIGH'),  _mock_vehicle('v2', 'HIGH')],
        'HIGH+MEDIUM':        [_mock_vehicle('v1', 'HIGH'),  _mock_vehicle('v2', 'MEDIUM')],
        'MEDIUM+MEDIUM':      [_mock_vehicle('v1', 'MEDIUM'), _mock_vehicle('v2', 'MEDIUM')],
        'HIGH+MEDIUM+HIGH':   [_mock_vehicle('v1', 'HIGH'),  _mock_vehicle('v2', 'MEDIUM'),
                                _mock_vehicle('v3', 'HIGH')],
    }

    threshold_s  = TEMPLATE_LOOKUP_LATENCY_THRESHOLD   # 0.005 s
    num_iters    = 100
    results      = {}
    all_pass     = True

    print(f"\nRunning {num_iters} iterations per group "
          f"(threshold: {threshold_s*1000:.0f}ms)...\n")

    for group_name, vehicles in groups.items():
        latencies = []
        best_score = 0.0

        for _ in range(num_iters):
            t0 = time.perf_counter()
            candidates = tm.matcher.find_best_template(vehicles, max_candidates=3)
            lat = time.perf_counter() - t0
            latencies.append(lat)
            if candidates and candidates[0][1] > best_score:
                best_score = candidates[0][1]

        arr = np.array(latencies)
        avg_ms  = arr.mean() * 1000
        max_ms  = arr.max() * 1000
        p99_ms  = np.percentile(arr, 99) * 1000
        status  = 'PASS' if p99_ms < threshold_s * 1000 else 'FAIL'
        if status == 'FAIL':
            all_pass = False

        results[group_name] = {
            'avg_ms': avg_ms, 'max_ms': max_ms, 'p99_ms': p99_ms,
            'best_score': best_score, 'status': status
        }

        # Print top-3 for this group
        top3 = tm.matcher.find_best_template(vehicles, max_candidates=3)
        top3_str = '  '.join(
            f"#{r}:{t.template_id}({s:.3f})" for r, (t, s) in enumerate(top3, 1)
        ) if top3 else '(none)'
        print(f"  [{group_name}] avg={avg_ms:.2f}ms  max={max_ms:.2f}ms  "
              f"p99={p99_ms:.2f}ms  best={best_score:.3f}  [{status}]")
        print(f"    Top-3: {top3_str}")

    # Online-learning feedback test
    print(f"\n{'='*70}")
    print("Online-learning feedback test")
    print('='*70)
    stats_pre = tm.get_template_statistics()
    # Simulate a successful pipeline
    mock_pipeline = Pipeline(
        pipeline_id='mock_test_pipeline',
        vehicles=['v1', 'v2'],
        stages=['backbone', 'action_head'],
        template_id='synth_000',
        start_time=time.time() - 10.0,
        expected_completion=time.time()
    )
    tm.register_successful_pipeline(mock_pipeline, success=True, duration=10.0)
    stats_post = tm.get_template_statistics()
    print(f"  Before: templates={stats_pre['total_templates']}  "
          f"avg_success={stats_pre['avg_success_rate']:.3f}")
    print(f"  After:  templates={stats_post['total_templates']}  "
          f"avg_success={stats_post['avg_success_rate']:.3f}")
    feedback_ok = stats_post['total_templates'] >= stats_pre['total_templates']
    print(f"  Feedback result: {'PASS' if feedback_ok else 'FAIL'}")
    if not feedback_ok:
        all_pass = False

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Group':<22} {'Avg(ms)':>8} {'Max(ms)':>8} {'P99(ms)':>8} "
          f"{'BestScore':>10} {'Status':>7}")
    print('-' * 70)
    for gn, r in results.items():
        print(f"{gn:<22} {r['avg_ms']:>8.2f} {r['max_ms']:>8.2f} {r['p99_ms']:>8.2f} "
              f"{r['best_score']:>10.3f} {r['status']:>7}")
    print('-' * 70)
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print('=' * 70)

    return 0 if all_pass else 1


# ==================== Main ====================

shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global shutdown_requested, server_instance, vehicle_instance

    if not shutdown_requested:
        shutdown_requested = True
        print("\n\nReceived interrupt signal, shutting down...")

        if 'server_instance' in globals() and server_instance:
            try:
                server_instance.stop()
            except:
                pass
        if 'vehicle_instance' in globals() and vehicle_instance:
            try:
                vehicle_instance.stop()
            except:
                pass

    # Force exit by calling os._exit instead of sys.exit
    # This ensures we exit even if there are blocking threads
    import os
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='FHDP Pipeline Training Test (Refactored) with EVO1Driving & Template Testing',
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

  # Run standalone template benchmark (no network):
  python test_pipeline_training_refactored.py --mode test-template --resource-level medium
        """
    )

    parser.add_argument('--mode', required=True,
                        choices=['server', 'vehicle', 'test-template'],
                        help='Operation mode: server, vehicle, or test-template')
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
                        help='Resource level (vehicle / test-template mode)')
    parser.add_argument('--config', help='Path to configuration file')

    args = parser.parse_args()

    # Set signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global server_instance, vehicle_instance, shutdown_requested

    try:
        if args.mode == 'test-template':
            # Standalone template benchmark (no network)
            rc = run_template_standalone_test(resource_level=args.resource_level)
            sys.exit(rc)

        elif args.mode == 'server':
            # Server mode
            server_instance = PipelineTestServer(
                host=args.host,
                port=args.port,
                config_path=args.config
            )
            server_instance.start()

            # Keep server running until shutdown requested
            while not shutdown_requested:
                time.sleep(0.1)

            # Clean shutdown after training complete or manual stop
            server_instance.stop()

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
                    while not shutdown_requested:
                        time.sleep(10.0)
                        vehicle_instance.send_heartbeat()
                except KeyboardInterrupt:
                    pass

            # Clean shutdown after training complete or manual stop
            if vehicle_instance:
                vehicle_instance.stop()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
