#!/usr/bin/env python3
"""
FHDP Pipeline Prototype (ResNet-18, single micro-batch)

Fixed topology:
- Stage0 (AGX)  : forward first half, send activations
- Stage1 (Orin) : forward second half, compute loss, send gradients
- Server        : coordinate only (no compute)

This script is intentionally decoupled and minimal for later extraction into FHDP core.

Usage:
  # Server (coordination only)
  python pipeline_proto.py --mode server --host 0.0.0.0 --port 5000

  # Stage0 (AGX)
  python pipeline_proto.py --mode vehicle --role stage0 --vehicle-id agx --server-host <server-ip> --server-port 5000

  # Stage1 (Orin)
  python pipeline_proto.py --mode vehicle --role stage1 --vehicle-id orin --server-host <server-ip> --server-port 5000
"""

import os
import sys
import time
import uuid
import argparse
import threading
import signal
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import timm

# ---- Resolve project root for imports ----
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_roots = [
    os.path.abspath(os.path.join(script_dir, '..')),      # fhdp/tests/pipeline_test -> fhdp/
    os.path.abspath(os.path.join(script_dir, '../..')),   # fhdp/tests/pipeline_test -> project root
    os.path.abspath(os.path.join(script_dir, '../../..'))
]
project_root = None
for root in possible_roots:
    if os.path.exists(os.path.join(root, 'fhdp')):
        project_root = root
        break
    if os.path.exists(os.path.join(root, 'core')):  # root is fhdp package dir
        project_root = os.path.dirname(root)
        break
if project_root:
    sys.path.insert(0, project_root)

from fhdp.core.cross_platform_comm import (
    NetworkEndpoint, CrossPlatformMessage, PlatformBridge,
    TransportProtocol, CompressionType, SerializationFormat,
    HardwareCapabilities, HardwarePlatform, PipelineMessage
)
from fhdp.core.hardware_adapter import ComputeCapability


PIPELINE_ID = "proto_resnet18"
DEFAULT_STAGE0_ID = "agx"
DEFAULT_STAGE1_ID = "orin"
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 4
DEFAULT_ROUND = 1
DEFAULT_MICRO_BATCH = 0


# ----------------- Utilities -----------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_sequence_id(pipeline_id: str, round_num: int, kind: str, micro_idx: int) -> str:
    return f"{pipeline_id}|r{round_num}|{kind}|m{micro_idx}"


def _build_fake_loader(batch_size: int, image_size: int, num_classes: int, num_batches: int = 1) -> DataLoader:
    dataset = FakeData(
        size=batch_size * num_batches,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=ToTensor()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _split_resnet18(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    # Stage0: stem + layer1 + layer2
    stage0 = nn.Sequential(
        model.conv1, model.bn1, model.act1, model.maxpool,
        model.layer1, model.layer2
    )
    # Stage1: layer3 + layer4 + global_pool + fc
    stage1 = nn.Sequential(
        model.layer3, model.layer4, model.global_pool, model.fc
    )
    return stage0, stage1


def _build_capabilities(role: str) -> HardwareCapabilities:
    if role == "server":
        return HardwareCapabilities(
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
    if role == "stage0":
        return HardwareCapabilities(
            platform=HardwarePlatform.JETSON_ORIN,
            compute_capability=ComputeCapability.EDGE_AI,
            cpu_cores=12,
            cpu_freq=2.0,
            memory_total=32.0,
            gpu_memory=8.0,
            npu_memory=0.0,
            storage_speed='emmc',
            network_speed=1000.0,
            power_profile='balanced',
            thermal_limit=85.0,
            accelerated_compute=True
        )
    # stage1
    return HardwareCapabilities(
        platform=HardwarePlatform.JETSON_NANO,
        compute_capability=ComputeCapability.EDGE_AI,
        cpu_cores=6,
        cpu_freq=2.0,
        memory_total=8.0,
        gpu_memory=2.0,
        npu_memory=0.0,
        storage_speed='emmc',
        network_speed=1000.0,
        power_profile='balanced',
        thermal_limit=85.0,
        accelerated_compute=True
    )


# ----------------- Server -----------------

class PipelineProtoServer:
    def __init__(self, host: str, port: int, stage0_id: str, stage1_id: str):
        self.host = host
        self.port = port
        self.stage0_id = stage0_id
        self.stage1_id = stage1_id
        self.registered: Dict[str, Dict[str, Any]] = {}
        self.accepted = set()

        self.endpoint = NetworkEndpoint(
            host=host,
            port=port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE
        )
        self.bridge = PlatformBridge(_build_capabilities("server"), node_id="server")
        self.shutdown = False

    def start(self):
        self._register_handlers()
        self.bridge.message_router.start_message_listener(self.endpoint)
        print(f"[Server] Listening on {self.host}:{self.port}")

    def stop(self):
        if hasattr(self.bridge, 'message_router'):
            self.bridge.message_router.close()
        print("[Server] Stopped")

    def _register_handlers(self):
        self.bridge.message_router.register_handler("register", self._handle_register)
        self.bridge.message_router.register_handler("pipeline_response", self._handle_pipeline_response)
        self.bridge.message_router.register_handler("heartbeat", self._handle_heartbeat)
        self.bridge.message_router.register_handler("status", self._handle_status)

    def _handle_register(self, message: CrossPlatformMessage):
        vehicle_id = message.source_id
        metadata = message.metadata or {}
        client_host = metadata.get("client_host", "unknown")
        client_port = metadata.get("client_port", 0)

        self.registered[vehicle_id] = {
            "resources": message.payload.get("resources", {}),
            "host": client_host,
            "port": client_port
        }

        endpoint = NetworkEndpoint(
            host=client_host,
            port=client_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE
        )
        self.bridge.message_router.add_route(vehicle_id, endpoint)
        print(f"[Server] Registered {vehicle_id} @ {client_host}:{client_port}")

        if self.stage0_id in self.registered and self.stage1_id in self.registered:
            self._send_pipeline_invite()

    def _handle_pipeline_response(self, message: CrossPlatformMessage):
        payload = message.payload or {}
        if payload.get("accepted"):
            self.accepted.add(message.source_id)
            print(f"[Server] {message.source_id} accepted pipeline invite")
        else:
            print(f"[Server] {message.source_id} declined pipeline invite")

        if self.stage0_id in self.accepted and self.stage1_id in self.accepted:
            self._start_round(DEFAULT_ROUND)

    def _handle_heartbeat(self, message: CrossPlatformMessage):
        pass

    def _handle_status(self, message: CrossPlatformMessage):
        response = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=message.source_id,
            message_type="status_response",
            payload={
                "pipeline_id": PIPELINE_ID,
                "registered": list(self.registered.keys()),
                "accepted": list(self.accepted)
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(response)

    def _send_pipeline_invite(self):
        endpoints = {}
        for vehicle_id in [self.stage0_id, self.stage1_id]:
            info = self.registered.get(vehicle_id, {})
            host = info.get("host")
            port = info.get("port")
            if host and port:
                endpoints[vehicle_id] = {"host": host, "port": port}

        invite = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id="",
            message_type="pipeline_invite",
            payload={
                "pipeline_id": PIPELINE_ID,
                "template_id": "fixed_agx_orin_v1",
                "vehicles": [self.stage0_id, self.stage1_id],
                "stages": ["stage0", "stage1"],
                "resource_requirements": ["HIGH", "MEDIUM"],
                "endpoints": endpoints
            },
            requires_ack=False
        )

        for vehicle_id in [self.stage0_id, self.stage1_id]:
            invite.target_id = vehicle_id
            self.bridge.send_cross_platform_message(invite)
            print(f"[Server] Sent pipeline invite to {vehicle_id}")

    def _start_round(self, round_num: int):
        control = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=self.stage0_id,
            message_type="pipeline_control",
            payload={
                "pipeline_id": PIPELINE_ID,
                "round": round_num,
                "micro_batch": DEFAULT_MICRO_BATCH
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(control)
        print(f"[Server] Sent start_round to {self.stage0_id} (round {round_num})")


# ----------------- Vehicle -----------------

class PipelineProtoVehicle:
    def __init__(self, vehicle_id: str, role: str, server_host: str, server_port: int,
                 stage0_id: str, stage1_id: str, listen_host: str, listen_port: int,
                 advertise_host: Optional[str] = None):
        self.vehicle_id = vehicle_id
        self.role = role
        self.server_host = server_host
        self.server_port = server_port
        self.stage0_id = stage0_id
        self.stage1_id = stage1_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.advertise_host = advertise_host
        self.device = _get_device()

        self.current_pipeline_id: Optional[str] = None
        self.current_stage: Optional[str] = None

        self.stage0, self.stage1 = _split_resnet18(DEFAULT_NUM_CLASSES)
        if role == "stage0":
            self.model = self.stage0.to(self.device)
        else:
            self.model = self.stage1.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        self.endpoint = NetworkEndpoint(
            host=server_host,
            port=server_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE
        )
        self.bridge = PlatformBridge(_build_capabilities(role), node_id=vehicle_id)

        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def start(self):
        self._register_handlers()
        self._start_peer_listener()
        self._connect_to_server()
        self._register_with_server()
        print(f"[{self.role}] Ready on {self.vehicle_id} ({self.device})")

    def _start_peer_listener(self):
        if self.listen_port <= 0:
            return
        endpoint = NetworkEndpoint(
            host=self.listen_host,
            port=self.listen_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE
        )
        self.bridge.message_router.start_message_listener(endpoint)
        advertise_host = self.advertise_host or self._get_local_ip(self.server_host)
        print(f"[{self.role}] Peer listener on {self.listen_host}:{self.listen_port} (advertise {advertise_host})")

    def stop(self):
        if hasattr(self.bridge, 'message_router'):
            self.bridge.message_router.close()
        print(f"[{self.role}] Stopped")

    def _connect_to_server(self):
        remote_capabilities = _build_capabilities("server")
        ok = self.bridge.connect_to_platform(
            remote_node_id="server",
            remote_capabilities=remote_capabilities,
            network_endpoint=self.endpoint
        )
        if not ok:
            raise ConnectionError("Failed to connect to server")

    def _register_handlers(self):
        self.bridge.message_router.register_handler("pipeline_invite", self._handle_pipeline_invite)
        self.bridge.message_router.register_handler("pipeline_control", self._handle_pipeline_control)
        self.bridge.message_router.register_handler("status_response", self._handle_status)

    def _register_with_server(self):
        advertise_host = self.advertise_host or self._get_local_ip(self.server_host)
        msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="register",
            payload={
                "resources": {
                    "role": self.role,
                    "gpu": 1,
                    "memory": 1
                }
            },
            metadata={
                "client_host": advertise_host,
                "client_port": self.listen_port
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(msg)

    def _handle_pipeline_invite(self, message: CrossPlatformMessage):
        payload = message.payload
        self.current_pipeline_id = payload.get("pipeline_id")

        vehicles = payload.get("vehicles", [])
        stages = payload.get("stages", [])
        stage_map = {stages[i]: vehicles[i] for i in range(min(len(vehicles), len(stages)))}
        self.stage0_id = stage_map.get("stage0", self.stage0_id)
        self.stage1_id = stage_map.get("stage1", self.stage1_id)

        endpoints = payload.get("endpoints", {})
        for vehicle_id, endpoint_info in endpoints.items():
            if vehicle_id == self.vehicle_id:
                continue
            host = endpoint_info.get("host")
            port = endpoint_info.get("port")
            if host and port:
                self.bridge.message_router.add_route(
                    vehicle_id,
                    NetworkEndpoint(
                        host=host,
                        port=port,
                        protocol=TransportProtocol.TCP,
                        compression=CompressionType.NONE
                    )
                )

        if self.vehicle_id in vehicles:
            stage_index = vehicles.index(self.vehicle_id)
            self.current_stage = stages[stage_index]

        accept = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="pipeline_response",
            payload={
                "pipeline_id": self.current_pipeline_id,
                "accepted": True,
                "stage": self.current_stage
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(accept)
        print(f"[{self.role}] Accepted invite: stage={self.current_stage}")

        if self.role == "stage1":
            seq_id = _make_sequence_id(PIPELINE_ID, DEFAULT_ROUND, "activation", DEFAULT_MICRO_BATCH)
            self.bridge.message_router.pipeline_comm_manager.register_sequence_handler(
                seq_id, self._handle_activation_sequence
            )
        if self.role == "stage0":
            seq_id = _make_sequence_id(PIPELINE_ID, DEFAULT_ROUND, "gradient", DEFAULT_MICRO_BATCH)
            self.bridge.message_router.pipeline_comm_manager.register_sequence_handler(
                seq_id, self._handle_gradient_sequence
            )

    def _handle_pipeline_control(self, message: CrossPlatformMessage):
        if self.role != "stage0":
            return
        payload = message.payload or {}
        round_num = payload.get("round", DEFAULT_ROUND)
        micro_batch = payload.get("micro_batch", DEFAULT_MICRO_BATCH)
        threading.Thread(
            target=self._run_stage0_round,
            args=(round_num, micro_batch),
            daemon=True
        ).start()

    def _handle_status(self, message: CrossPlatformMessage):
        print(f"[{self.role}] Server status: {message.payload}")

    @staticmethod
    def _unwrap_tensor_payload(value):
        if isinstance(value, dict):
            if value.get("__tensor_metadata__") and "data" in value:
                return value["data"]
            if "data" in value:
                return value["data"]
        return value

    def _run_stage0_round(self, round_num: int, micro_batch: int):
        loader = _build_fake_loader(DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_NUM_CLASSES, num_batches=1)
        images, labels = next(iter(loader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        activation = self.model(images)
        seq_id = _make_sequence_id(PIPELINE_ID, round_num, "activation", micro_batch)

        with self._lock:
            self._activation_cache[seq_id] = activation

        pipeline_msg = PipelineMessage(
            pipeline_id=PIPELINE_ID,
            stage_id="stage0",
            source_id=self.vehicle_id,
            target_id=self.stage1_id,
            data={
                "activation": activation.detach(),
                "labels": labels.detach(),
                "round": round_num,
                "micro_batch": micro_batch
            },
            data_type="activation",
            sequence_id=seq_id,
            sequence_index=0,
            total_sequence_length=1,
            requires_ack=False,
            compression_type=CompressionType.NONE,
            serialization_format=SerializationFormat.PICKLE
        )

        endpoint = self.bridge.message_router.routing_table.get(self.stage1_id)
        if endpoint:
            self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(pipeline_msg, endpoint)
            print(f"[stage0] Sent activation (round {round_num})")

    def _handle_activation_sequence(self, messages):
        pipeline_msg = messages[0]
        data = pipeline_msg.data

        activation_data = self._unwrap_tensor_payload(data["activation"])
        labels_data = self._unwrap_tensor_payload(data["labels"])
        activation = torch.tensor(activation_data, device=self.device, dtype=torch.float32)
        labels = torch.tensor(labels_data, device=self.device, dtype=torch.long)
        activation.requires_grad_(True)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(activation)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        grad = activation.grad.detach()
        round_num = data.get("round", DEFAULT_ROUND)
        micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
        seq_id = _make_sequence_id(PIPELINE_ID, round_num, "gradient", micro_batch)

        grad_msg = PipelineMessage(
            pipeline_id=PIPELINE_ID,
            stage_id="stage1",
            source_id=self.vehicle_id,
            target_id=self.stage0_id,
            data={
                "grad": grad,
                "round": round_num,
                "micro_batch": micro_batch
            },
            data_type="gradient",
            sequence_id=seq_id,
            sequence_index=0,
            total_sequence_length=1,
            requires_ack=False,
            compression_type=CompressionType.NONE,
            serialization_format=SerializationFormat.PICKLE
        )

        endpoint = self.bridge.message_router.routing_table.get(self.stage0_id)
        if endpoint:
            self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(grad_msg, endpoint)
            print(f"[stage1] Sent gradient (loss={loss.item():.4f})")

        self.bridge.message_router.pipeline_comm_manager.unregister_sequence_handler(pipeline_msg.sequence_id)

    def _handle_gradient_sequence(self, messages):
        pipeline_msg = messages[0]
        data = pipeline_msg.data
        grad_data = self._unwrap_tensor_payload(data["grad"])
        grad = torch.tensor(grad_data, device=self.device, dtype=torch.float32)

        round_num = data.get("round", DEFAULT_ROUND)
        micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
        activation_seq_id = _make_sequence_id(PIPELINE_ID, round_num, "activation", micro_batch)
        with self._lock:
            activation = self._activation_cache.pop(activation_seq_id, None)

        if activation is None:
            print("[stage0] Missing activation for gradient")
            return

        activation.backward(grad)
        self.optimizer.step()
        print("[stage0] Applied gradient and updated weights")

        self.bridge.message_router.pipeline_comm_manager.unregister_sequence_handler(pipeline_msg.sequence_id)

    @staticmethod
    def _get_local_ip(remote_host: Optional[str] = None) -> str:
        import socket
        try:
            if remote_host:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.connect((remote_host, 80))
                ip = sock.getsockname()[0]
                sock.close()
                return ip
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


# ----------------- Main -----------------

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(description="FHDP Pipeline Prototype (ResNet-18)")
    parser.add_argument("--mode", required=True, choices=["server", "vehicle"], help="server or vehicle")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--port", type=int, default=5000, help="server port")
    parser.add_argument("--server-host", default="localhost", help="server host (vehicle mode)")
    parser.add_argument("--server-port", type=int, default=5000, help="server port (vehicle mode)")
    parser.add_argument("--vehicle-id", default="vehicle_001", help="vehicle id")
    parser.add_argument("--role", choices=["stage0", "stage1"], help="vehicle role")
    parser.add_argument("--stage0-id", default=DEFAULT_STAGE0_ID, help="stage0 vehicle id")
    parser.add_argument("--stage1-id", default=DEFAULT_STAGE1_ID, help="stage1 vehicle id")
    parser.add_argument("--listen-host", default="0.0.0.0", help="vehicle listen host for peer pipeline data")
    parser.add_argument("--listen-port", type=int, default=0, help="vehicle listen port for peer pipeline data")
    parser.add_argument("--advertise-host", default=None, help="host/IP to advertise to peers (default: auto-detect)")

    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.mode == "server":
        server = PipelineProtoServer(args.host, args.port, args.stage0_id, args.stage1_id)
        server.start()
        while not shutdown_requested:
            time.sleep(0.1)
        server.stop()
        return

    if args.mode == "vehicle":
        if not args.role:
            raise ValueError("--role is required in vehicle mode")
        if args.listen_port == 0:
            args.listen_port = 6000 if args.role == "stage0" else 6001

        vehicle = PipelineProtoVehicle(
            args.vehicle_id,
            args.role,
            args.server_host,
            args.server_port,
            args.stage0_id,
            args.stage1_id,
            args.listen_host,
            args.listen_port,
            args.advertise_host
        )
        vehicle.start()
        try:
            while not shutdown_requested:
                time.sleep(5.0)
        finally:
            vehicle.stop()


if __name__ == "__main__":
    main()
