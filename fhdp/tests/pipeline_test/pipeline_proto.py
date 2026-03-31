#!/usr/bin/env python3
"""
FHDP Pipeline Prototype (ResNet-18, micro-batching + 1F1B schedule)

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
from queue import Queue, Empty
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
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
from fhdp.core.pipeline_runtime import (
    SequenceIdFactory,
    SequenceHandlerRegistry,
    OneFOneBSchedule,
    get_micro_batch_phase,
)


PIPELINE_ID = "proto_resnet18"
DEFAULT_STAGE0_ID = "agx"
DEFAULT_STAGE1_ID = "orin"
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 4
DEFAULT_ROUND = 1
DEFAULT_ROUNDS = 1
DEFAULT_MICRO_BATCH = 0
DEFAULT_MICRO_BATCHES = 1
DEFAULT_DATA_DIR = os.path.join(script_dir, "data")


# ----------------- Utilities -----------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_advertise_host(listen_host: str, advertise_host: Optional[str], server_host: str) -> str:
    if advertise_host:
        return advertise_host
    if listen_host and listen_host not in {"0.0.0.0", "::"}:
        return listen_host
    return PipelineProtoVehicle._get_local_ip(server_host)


def _probe_endpoint(host: str, port: int, timeout: float = 2.0, label: str = "peer") -> bool:
    import socket
    if not host or port <= 0:
        print(f"[Probe] Skip {label} probe: invalid host/port {host}:{port}")
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"[Probe] {label} reachable: {host}:{port}")
            return True
    except Exception as exc:
        print(f"[Probe] {label} unreachable: {host}:{port} ({exc})")
        return False


def _validate_pipeline_utils(rounds: int, micro_batches: int) -> None:
    """Quick smoke test for 1F1B scheduling + sequence id generation."""
    schedule = OneFOneBSchedule(micro_batches)
    seq_factory = SequenceIdFactory(PIPELINE_ID)

    phases = [schedule.phase(idx).value for idx in schedule.iter_micro_batches()]
    activation_ids = [seq_factory.make(1, "activation", idx) for idx in schedule.iter_micro_batches()]
    gradient_ids = [seq_factory.make(1, "gradient", idx) for idx in schedule.iter_micro_batches()]

    unique_ids = len(set(activation_ids + gradient_ids)) == len(activation_ids + gradient_ids)

    print("[Validate] 1F1B phases:", phases)
    print("[Validate] activation sequence ids:", activation_ids)
    print("[Validate] gradient sequence ids:", gradient_ids)
    print(f"[Validate] unique sequence ids: {unique_ids}")
    print(f"[Validate] rounds={rounds}, micro_batches={micro_batches}")


def _build_cifar10_loader(
    batch_size: int,
    image_size: int,
    num_classes: int,
    num_batches: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    download: bool = False
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    total_samples = batch_size * num_batches
    if total_samples < len(dataset):
        dataset = Subset(dataset, list(range(total_samples)))
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
    def __init__(self, host: str, port: int, stage0_id: str, stage1_id: str,
                 rounds: int, auto_exit: bool, micro_batches: int):
        self.host = host
        self.port = port
        self.stage0_id = stage0_id
        self.stage1_id = stage1_id
        self.rounds = max(1, int(rounds))
        self.auto_exit = auto_exit
        self.micro_batches = max(1, int(micro_batches))
        self.registered: Dict[str, Dict[str, Any]] = {}
        self.accepted = set()
        self.current_round = 0
        self.completed_rounds = set()

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
        self.bridge.message_router.register_handler("round_done", self._handle_round_done)
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
            self._start_next_round()

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
                "accepted": list(self.accepted),
                "current_round": self.current_round,
                "total_rounds": self.rounds
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(response)

    def _handle_round_done(self, message: CrossPlatformMessage):
        payload = message.payload or {}
        round_num = payload.get("round")
        if message.source_id != self.stage0_id:
            return
        if not isinstance(round_num, int):
            return
        if round_num in self.completed_rounds:
            return
        if round_num != self.current_round:
            return

        self.completed_rounds.add(round_num)
        print(f"[Server] round {round_num} completed by {message.source_id}")

        if self.current_round >= self.rounds:
            if self.auto_exit:
                _request_shutdown()
            return

        self._start_next_round()

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

    def _start_next_round(self):
        if self.current_round >= self.rounds:
            if self.auto_exit:
                _request_shutdown()
            return

        self.current_round += 1
        control = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=self.stage0_id,
            message_type="pipeline_control",
            payload={
                "pipeline_id": PIPELINE_ID,
                "round": self.current_round,
                "micro_batch": DEFAULT_MICRO_BATCH,
                "micro_batches": self.micro_batches
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(control)
        print(f"[Server] Sent start_round to {self.stage0_id} (round {self.current_round}/{self.rounds})")


# ----------------- Vehicle -----------------

class PipelineProtoVehicle:
    def __init__(self, vehicle_id: str, role: str, server_host: str, server_port: int,
                 stage0_id: str, stage1_id: str, listen_host: str, listen_port: int,
                 advertise_host: Optional[str] = None, rounds: int = DEFAULT_ROUNDS,
                 auto_exit: bool = False, micro_batches: int = DEFAULT_MICRO_BATCHES,
                 data_dir: str = DEFAULT_DATA_DIR, download: bool = False):
        self.vehicle_id = vehicle_id
        self.role = role
        self.server_host = server_host
        self.server_port = server_port
        self.stage0_id = stage0_id
        self.stage1_id = stage1_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.advertise_host = advertise_host
        self.advertise_host_resolved: Optional[str] = None
        self.total_rounds = max(1, int(rounds))
        self.auto_exit = auto_exit
        self.completed_rounds = 0
        self.micro_batches = max(1, int(micro_batches))
        self.data_dir = data_dir
        self.download = download
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

        self.sequence_id_factory = SequenceIdFactory(PIPELINE_ID)
        self.sequence_registry = SequenceHandlerRegistry(
            self.bridge.message_router.pipeline_comm_manager,
            self.sequence_id_factory
        )

        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._round_state: Dict[int, Dict[str, Any]] = {}
        self._round_cond: Dict[int, threading.Condition] = {}
        self._activation_queue: Queue[PipelineMessage] = Queue()
        self._gradient_queue: Queue[PipelineMessage] = Queue()

    def start(self):
        self._register_handlers()
        self._start_peer_listener()
        self._connect_to_server()
        self._register_with_server()
        self._start_async_workers()
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
        self.advertise_host_resolved = _resolve_advertise_host(
            self.listen_host,
            self.advertise_host,
            self.server_host
        )
        print(f"[{self.role}] Peer listener on {self.listen_host}:{self.listen_port} (advertise {self.advertise_host_resolved})")

    def _start_async_workers(self):
        if self.role == "stage1":
            threading.Thread(target=self._activation_worker, daemon=True).start()
        if self.role == "stage0":
            threading.Thread(target=self._gradient_worker, daemon=True).start()

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
        advertise_host = getattr(self, "advertise_host_resolved", None)
        advertise_host = _resolve_advertise_host(
            self.listen_host,
            advertise_host,
            self.server_host
        )
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
                ok = _probe_endpoint(host, int(port), label=f"peer {vehicle_id}")
                if not ok:
                    print(
                        f"[{self.role}] Warning: peer {vehicle_id} unreachable. "
                        f"Ensure {host}:{port} is reachable or use --advertise-host on peer."
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

        rounds = range(1, self.total_rounds + 1)
        if self.role == "stage1":
            self.sequence_registry.register_for_rounds(
                "activation",
                rounds,
                self.micro_batches,
                self._handle_activation_sequence
            )
        if self.role == "stage0":
            self.sequence_registry.register_for_rounds(
                "gradient",
                rounds,
                self.micro_batches,
                self._handle_gradient_sequence
            )

    def _handle_pipeline_control(self, message: CrossPlatformMessage):
        if self.role != "stage0":
            return
        payload = message.payload or {}
        round_num = payload.get("round", DEFAULT_ROUND)
        micro_batch = payload.get("micro_batch", DEFAULT_MICRO_BATCH)
        micro_batches = payload.get("micro_batches", self.micro_batches)
        threading.Thread(
            target=self._run_stage0_round,
            args=(round_num, micro_batch, micro_batches),
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

    def _run_stage0_round(self, round_num: int, micro_batch: int, micro_batches: int):
        total_micro_batches = max(1, int(micro_batches))
        loader = _build_cifar10_loader(
            DEFAULT_BATCH_SIZE * total_micro_batches,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_NUM_CLASSES,
            num_batches=1,
            data_dir=self.data_dir,
            download=self.download
        )
        images, labels = next(iter(loader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        with self._lock:
            self._round_state[round_num] = {
                "expected": total_micro_batches,
                "received": 0,
                "done": False
            }
            self._round_cond[round_num] = threading.Condition(self._lock)

        image_chunks = list(images.chunk(total_micro_batches))
        label_chunks = list(labels.chunk(total_micro_batches))

        if total_micro_batches == 1:
            print(f"[stage0] 1F1B warmup: sending single micro-batch (round {round_num})")
        else:
            print(f"[stage0] 1F1B warmup: sending first micro-batch (round {round_num})")

        schedule = OneFOneBSchedule(total_micro_batches)
        for micro_idx, (img_mb, lbl_mb) in enumerate(zip(image_chunks, label_chunks)):
            activation = self.model(img_mb)
            seq_id = self.sequence_id_factory.make(round_num, "activation", micro_idx)

            with self._lock:
                self._activation_cache[seq_id] = activation

            pipeline_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="stage0",
                source_id=self.vehicle_id,
                target_id=self.stage1_id,
                data={
                    "activation": activation.detach(),
                    "labels": lbl_mb.detach(),
                    "round": round_num,
                    "micro_batch": micro_idx,
                    "micro_batches": total_micro_batches
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
                phase = schedule.phase(micro_idx).value
                if phase == "warmup":
                    print(f"[stage0] Sent activation warmup (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                elif phase == "cooldown":
                    print(f"[stage0] Sent activation cooldown (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                else:
                    print(f"[stage0] Sent activation steady (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")

        print(f"[stage0] 1F1B cooldown: waiting for gradients (round {round_num})")
        with self._lock:
            cond = self._round_cond.get(round_num)
            while not shutdown_requested and not self._round_state.get(round_num, {}).get("done"):
                if cond:
                    cond.wait(timeout=0.5)
                else:
                    break
            self._round_state.pop(round_num, None)
            self._round_cond.pop(round_num, None)
        print(f"[stage0] 1F1B cooldown: gradients complete (round {round_num})")

    def _handle_activation_sequence(self, messages):
        pipeline_msg = messages[0]
        self._activation_queue.put(pipeline_msg)

    def _activation_worker(self):
        while not shutdown_requested:
            try:
                pipeline_msg = self._activation_queue.get(timeout=0.5)
            except Empty:
                continue

            data = pipeline_msg.data
            activation_data = self._unwrap_tensor_payload(data["activation"])
            labels_data = self._unwrap_tensor_payload(data["labels"])
            activation = torch.tensor(activation_data, device=self.device, dtype=torch.float32)
            labels = torch.tensor(labels_data, device=self.device, dtype=torch.long)
            activation.requires_grad_(True)

            round_num = data.get("round", DEFAULT_ROUND)
            micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
            micro_batches = data.get("micro_batches", self.micro_batches)

            with self._lock:
                if round_num not in self._round_state:
                    self._round_state[round_num] = {"expected": int(micro_batches), "received": 0}
                    self.model.train()
                    self.optimizer.zero_grad()

            phase = get_micro_batch_phase(micro_batch, micro_batches).value

            outputs = self.model(activation)
            loss = self.criterion(outputs, labels)
            loss.backward()

            grad = activation.grad.detach()
            seq_id = self.sequence_id_factory.make(round_num, "gradient", micro_batch)

            grad_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="stage1",
                source_id=self.vehicle_id,
                target_id=self.stage0_id,
                data={
                    "grad": grad,
                    "round": round_num,
                    "micro_batch": micro_batch,
                    "micro_batches": micro_batches
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
                print(
                    f"[stage1] Sent gradient {phase} (loss={loss.item():.4f}, micro {micro_batch + 1}/{micro_batches})"
                )

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            with self._lock:
                state = self._round_state.get(round_num)
                if state:
                    state["received"] += 1
                    if state["received"] >= state["expected"]:
                        self.optimizer.step()
                        self._round_state.pop(round_num, None)
                        if self.auto_exit:
                            self.completed_rounds += 1
                            if self.completed_rounds >= self.total_rounds:
                                _request_shutdown()

    def _handle_gradient_sequence(self, messages):
        pipeline_msg = messages[0]
        self._gradient_queue.put(pipeline_msg)

    def _gradient_worker(self):
        while not shutdown_requested:
            try:
                pipeline_msg = self._gradient_queue.get(timeout=0.5)
            except Empty:
                continue

            data = pipeline_msg.data
            grad_data = self._unwrap_tensor_payload(data["grad"])
            grad = torch.tensor(grad_data, device=self.device, dtype=torch.float32)

            round_num = data.get("round", DEFAULT_ROUND)
            micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
            micro_batches = data.get("micro_batches", self.micro_batches)
            activation_seq_id = self.sequence_id_factory.make(round_num, "activation", micro_batch)
            with self._lock:
                activation = self._activation_cache.pop(activation_seq_id, None)

            if activation is None:
                print("[stage0] Missing activation for gradient")
                continue

            activation.backward(grad)
            print("[stage0] Applied gradient for micro-batch")

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            with self._lock:
                if round_num not in self._round_state:
                    self._round_state[round_num] = {"expected": int(micro_batches), "received": 0}
                state = self._round_state[round_num]
                state["received"] += 1
                if state["received"] >= state["expected"]:
                    self.optimizer.step()
                    state["done"] = True
                    print("[stage0] Applied all gradients and updated weights")

                    if self.auto_exit:
                        self.completed_rounds += 1
                        if self.completed_rounds >= self.total_rounds:
                            _request_shutdown()

                    round_done = CrossPlatformMessage(
                        message_id=str(uuid.uuid4()),
                        source_id=self.vehicle_id,
                        target_id="server",
                        message_type="round_done",
                        payload={
                            "pipeline_id": PIPELINE_ID,
                            "round": round_num,
                            "stage": self.current_stage
                        },
                        requires_ack=False
                    )
                    self.bridge.send_cross_platform_message(round_done)

                    cond = self._round_cond.get(round_num)
                    if cond:
                        cond.notify_all()

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


def _request_shutdown():
    global shutdown_requested
    shutdown_requested = True


def signal_handler(signum, frame):
    _request_shutdown()


def main():
    parser = argparse.ArgumentParser(description="FHDP Pipeline Prototype (ResNet-18)")
    parser.add_argument("--mode", required=True, choices=["server", "vehicle", "validate"], help="server, vehicle, or validate")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--port", type=int, default=5000, help="server port")
    parser.add_argument("--server-host", default="localhost", help="server host (vehicle mode)")
    parser.add_argument("--server-port", type=int, default=5000, help="server port (vehicle mode)")
    parser.add_argument("--vehicle-id", default="vehicle_001", help="vehicle id")
    parser.add_argument("--role", choices=["stage0", "stage1"], help="vehicle role")
    parser.add_argument("--stage0-id", default=DEFAULT_STAGE0_ID, help="stage0 vehicle id")
    parser.add_argument("--stage1-id", default=DEFAULT_STAGE1_ID, help="stage1 vehicle id")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="total training rounds")
    parser.add_argument("--micro-batches", type=int, default=DEFAULT_MICRO_BATCHES, help="micro-batches per round")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="CIFAR-10 data directory")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    parser.add_argument("--auto-exit", action="store_true", help="exit after completing all rounds")
    parser.add_argument("--listen-host", default="0.0.0.0", help="vehicle listen host for peer pipeline data")
    parser.add_argument("--listen-port", type=int, default=0, help="vehicle listen port for peer pipeline data")
    parser.add_argument("--advertise-host", default=None, help="host/IP to advertise to peers (default: auto-detect)")

    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.mode == "validate":
        _validate_pipeline_utils(args.rounds, args.micro_batches)
        return

    if args.mode == "server":
        server = PipelineProtoServer(
            args.host,
            args.port,
            args.stage0_id,
            args.stage1_id,
            args.rounds,
            args.auto_exit,
            args.micro_batches
        )
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
            args.advertise_host,
            rounds=args.rounds,
            auto_exit=args.auto_exit,
            micro_batches=args.micro_batches,
            data_dir=args.data_dir,
            download=args.download
        )
        vehicle.start()
        try:
            while not shutdown_requested:
                time.sleep(5.0)
        finally:
            vehicle.stop()


if __name__ == "__main__":
    main()
