#!/usr/bin/env python3
"""
EdgePipe Jetson Implementation (Two Jetson devices)

Fixed topology:
- Device0 (stronger Jetson) : handles super neurons covering early layers
- Device1 (weaker Jetson)  : handles super neurons covering later layers
- Server                   : coordinate only (no compute)

Usage:
  # Server (coordination only)
  python -m fhdp.EdgePipe.edgepipe_jetson --mode server --host 0.0.0.0 --port 5000

  # Device0
  python -m fhdp.EdgePipe.edgepipe_jetson --mode device --role device0 --device-id agx --server-host <server-ip> --server-port 5000

  # Device1
  python -m fhdp.EdgePipe.edgepipe_jetson --mode device --role device1 --device-id orin --server-host <server-ip> --server-port 5000
"""

import os
import sys
import time
import uuid
import argparse
import threading
import signal
from queue import Queue, Empty
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

# ---- Resolve project root for imports ----
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_roots = [
    os.path.abspath(os.path.join(script_dir, '..')),      # fhdp/EdgePipe -> fhdp/
    os.path.abspath(os.path.join(script_dir, '../..')),   # fhdp/EdgePipe -> project root
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
from fhdp.core import ActivationLEPState
from fhdp.core.pipeline_model import (
    get_pipeline_template,
    serialize_template,
    build_model_split_from_template_payload,
)

from fhdp.EdgePipe.super_neuron import SuperNeuronNetwork
from fhdp.EdgePipe.partitioning import HybridPartitioning
from fhdp.EdgePipe.device_mapping import NeuronDeviceMapping
from fhdp.EdgePipe.pipeline_scheduler import PipelineScheduler
from fhdp.EdgePipe.performance_analysis import PerformanceAnalyzer

# ---- Constants ----
PIPELINE_ID = "edgepipe_jetson"
DEFAULT_DEVICE0_ID = "agx"
DEFAULT_DEVICE1_ID = "orin"
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 12
DEFAULT_ROUND = 10
DEFAULT_ROUNDS = 10
DEFAULT_MICRO_BATCH = 4
DEFAULT_MICRO_BATCHES = 4
DEFAULT_DATA_DIR = os.path.join(script_dir, "data")
DEFAULT_TEMPLATE_ID = "vit_b16_2stage_v1"
DEFAULT_DATASET = "cifar10"
DEFAULT_ROUND_TIMEOUT_SEC = 300
DEFAULT_EVAL_BATCHES = 4
DEFAULT_SAVE_EVERY = 1
DEFAULT_CHECKPOINT_DIR = os.path.join(project_root or script_dir, "logs", "checkpoints", "edgepipe_jetson")

# ---- LEP / activation compression ----
ENABLE_ACTIVATION_LEP = False
LEP_FP16_DTYPE = torch.float16
LEP_LOG_INTERVAL = 10

# ---- EdgePipe Configuration ----
DEFAULT_TOTAL_LAYERS = 6  # 1 input layer + 5 hidden layers
DEFAULT_NEURONS_PER_LAYER = [784, 128, 128, 128, 128, 10]  # MNIST-like model

# ----------------- Utilities -----------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_advertise_host(listen_host: str, advertise_host: Optional[str], server_host: str) -> str:
    if advertise_host:
        return advertise_host
    if listen_host and listen_host not in {"0.0.0.0", "::"}:
        return listen_host
    return EdgePipeJetsonDevice._get_local_ip(server_host)


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
    num_batches: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    download: bool = False,
    train: bool = True
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    total_samples = batch_size * num_batches
    if total_samples < len(dataset):
        dataset = Subset(dataset, list(range(total_samples)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _build_data_loader(
    dataset_name: str,
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    download: bool = False
) -> DataLoader:
    dataset_key = (dataset_name or "cifar10").lower().replace("-", "")
    return _build_cifar10_loader(batch_size, image_size, num_batches, data_dir, download)


def _build_eval_loader(
    dataset_name: str,
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    download: bool = False
) -> Optional[DataLoader]:
    dataset_key = (dataset_name or "cifar10").lower().replace("-", "")
    if dataset_key != "cifar10":
        print(f"[Eval] Dataset {dataset_name} not supported for eval, skip")
        return None
    return _build_cifar10_loader(batch_size, image_size, num_batches, data_dir, download, train=False)


def _resolve_platform_from_device_id(device_id: Optional[str], role: str) -> HardwarePlatform:
    name = (device_id or "").lower()
    if "agx" in name or "xavier" in name:
        return HardwarePlatform.JETSON_XAVIER
    if "orin" in name:
        return HardwarePlatform.JETSON_ORIN
    if "nano" in name:
        return HardwarePlatform.JETSON_NANO
    return HardwarePlatform.JETSON_ORIN if role == "device0" else HardwarePlatform.JETSON_NANO


def _build_capabilities(role: str, device_id: Optional[str] = None) -> HardwareCapabilities:
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

    platform = _resolve_platform_from_device_id(device_id, role)
    if platform == HardwarePlatform.JETSON_XAVIER:
        return HardwareCapabilities(
            platform=HardwarePlatform.JETSON_XAVIER,
            compute_capability=ComputeCapability.EDGE_AI,
            cpu_cores=8,
            cpu_freq=2.0,
            memory_total=32.0,
            gpu_memory=16.0,
            npu_memory=2.0,
            storage_speed='emmc',
            network_speed=10000.0,
            power_profile='high_performance',
            thermal_limit=90.0,
            accelerated_compute=True
        )
    if platform == HardwarePlatform.JETSON_ORIN:
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

# ----------------- EdgePipe Server -----------------

class EdgePipeJetsonServer:
    def __init__(self, host: str, port: int, device0_id: str, device1_id: str,
                 rounds: int, auto_exit: bool, micro_batches: int, template_id: str):
        self.host = host
        self.port = port
        self.device0_id = device0_id
        self.device1_id = device1_id
        self.rounds = max(1, int(rounds))
        self.auto_exit = auto_exit
        self.micro_batches = max(1, int(micro_batches))
        self.template_id = template_id
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

        # EdgePipe specific configuration
        self.total_layers = DEFAULT_TOTAL_LAYERS
        self.neurons_per_layer = DEFAULT_NEURONS_PER_LAYER
        self.total_devices = 2

        # Initialize EdgePipe components
        self.partitioning = HybridPartitioning(self.total_layers, self.total_devices)
        self.super_neuron_network = self.partitioning.perform_partitioning(self.neurons_per_layer)
        
        # Create device network (PRR matrix)
        # Simulate good connectivity between devices
        self.device_network = torch.eye(self.total_devices).numpy()
        # Set high PRR between devices
        self.device_network[0][1] = 0.95
        self.device_network[1][0] = 0.95

        # Optimize neuron to device mapping
        self.mapping = NeuronDeviceMapping(self.super_neuron_network, self.device_network)
        self.best_mapping = self.mapping.optimize_mapping(generations=1000)

        # Create pipeline scheduler
        self.scheduler = PipelineScheduler(self.super_neuron_network)

        # Create performance analyzer
        self.analyzer = PerformanceAnalyzer(self.super_neuron_network)

        print("[Server] EdgePipe initialized:")
        print(f"  Total layers: {self.total_layers}")
        print(f"  Neurons per layer: {self.neurons_per_layer}")
        print(f"  Total devices: {self.total_devices}")
        print(f"  M_layers: {self.partitioning.M_layers}")
        print(f"  Best mapping: {self.best_mapping}")
        print(f"  Performance summary: {self.analyzer.get_performance_summary(self.micro_batches)}")

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

        if self.device0_id in self.registered and self.device1_id in self.registered:
            self._send_pipeline_invite()

    def _handle_pipeline_response(self, message: CrossPlatformMessage):
        payload = message.payload or {}
        if payload.get("accepted"):
            self.accepted.add(message.source_id)
            print(f"[Server] {message.source_id} accepted pipeline invite")
        else:
            print(f"[Server] {message.source_id} declined pipeline invite")

        if self.device0_id in self.accepted and self.device1_id in self.accepted:
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
                "total_rounds": self.rounds,
                "edgepipe_config": {
                    "total_layers": self.total_layers,
                    "neurons_per_layer": self.neurons_per_layer,
                    "M_layers": self.partitioning.M_layers,
                    "best_mapping": self.best_mapping
                }
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(response)

    def _handle_round_done(self, message: CrossPlatformMessage):
        payload = message.payload or {}
        round_num = payload.get("round")
        if message.source_id != self.device0_id:
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
        for device_id in [self.device0_id, self.device1_id]:
            info = self.registered.get(device_id, {})
            host = info.get("host")
            port = info.get("port")
            if host and port:
                endpoints[device_id] = {"host": host, "port": port}

        template = get_pipeline_template(self.template_id, DEFAULT_TEMPLATE_ID)
        template_payload = serialize_template(template)

        # Get EdgePipe configuration
        edgepipe_config = {
            "total_layers": self.total_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "M_layers": self.partitioning.M_layers,
            "layer_groups": self.partitioning.layer_groups,
            "device_allocation": self.partitioning.device_allocation,
            "best_mapping": self.best_mapping
        }

        invite = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id="",
            message_type="pipeline_invite",
            payload={
                "pipeline_id": PIPELINE_ID,
                "template_id": template.template_id,
                "template": template_payload,
                "vehicles": [self.device0_id, self.device1_id],
                "stages": ["device0", "device1"],
                "resource_requirements": [r.value for r in template.resource_requirements],
                "endpoints": endpoints,
                "edgepipe_config": edgepipe_config
            },
            requires_ack=False
        )

        for device_id in [self.device0_id, self.device1_id]:
            invite.target_id = device_id
            self.bridge.send_cross_platform_message(invite)
            print(f"[Server] Sent pipeline invite to {device_id}")

    def _start_next_round(self):
        if self.current_round >= self.rounds:
            if self.auto_exit:
                _request_shutdown()
            return

        self.current_round += 1
        control = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=self.device0_id,
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
        print(f"[Server] Sent start_round to {self.device0_id} (round {self.current_round}/{self.rounds})")

# ----------------- EdgePipe Device -----------------

class EdgePipeJetsonDevice:
    def __init__(self, device_id: str, role: str, server_host: str, server_port: int,
                 device0_id: str, device1_id: str, listen_host: str, listen_port: int,
                 advertise_host: Optional[str] = None, rounds: int = DEFAULT_ROUNDS,
                 auto_exit: bool = False, micro_batches: int = DEFAULT_MICRO_BATCHES,
                 data_dir: str = DEFAULT_DATA_DIR, download: bool = False,
                 dataset: str = DEFAULT_DATASET, image_size: int = DEFAULT_IMAGE_SIZE,
                 num_classes: int = DEFAULT_NUM_CLASSES, eval_batches: int = DEFAULT_EVAL_BATCHES,
                 save_every: int = DEFAULT_SAVE_EVERY, save_dir: str = DEFAULT_CHECKPOINT_DIR):
        self.device_id = device_id
        self.role = role
        self.server_host = server_host
        self.server_port = server_port
        self.device0_id = device0_id
        self.device1_id = device1_id
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
        self.dataset = dataset
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.eval_batches = max(0, int(eval_batches))
        self.save_every = max(0, int(save_every))
        self.save_dir = save_dir
        self.device = _get_device()
        self._timing_start = time.perf_counter()
        self._invite_received_at: Optional[float] = None

        self.current_pipeline_id: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.current_template_id: Optional[str] = None
        self.current_split_key: Optional[str] = None

        # EdgePipe specific configuration
        self.edgepipe_config = None
        self.super_neuron_network = None
        self.assigned_super_neurons = []

        self.stage0: Optional[nn.Module] = None
        self.stage1: Optional[nn.Module] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

        self.endpoint = NetworkEndpoint(
            host=server_host,
            port=server_port,
            protocol=TransportProtocol.TCP,
            compression=CompressionType.NONE
        )
        self.bridge = PlatformBridge(_build_capabilities(role, device_id=device_id), node_id=device_id)

        self.sequence_id_factory = SequenceIdFactory(PIPELINE_ID)
        self.sequence_registry = SequenceHandlerRegistry(
            self.bridge.message_router.pipeline_comm_manager,
            self.sequence_id_factory
        )

        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._round_state: Dict[int, Dict[str, Any]] = {}
        self._round_cond: Dict[int, threading.Condition] = {}
        self._round_metrics: Dict[int, Dict[str, Any]] = {}
        self._eval_state: Dict[int, Dict[str, Any]] = {}
        self._activation_queue: Queue[PipelineMessage] = Queue()
        self._gradient_queue: Queue[PipelineMessage] = Queue()
        self._eval_queue: Queue[PipelineMessage] = Queue()

        self._lep_state = ActivationLEPState()

    def _log_timing(self, event: str, invite_start: Optional[float] = None) -> None:
        now = time.perf_counter()
        elapsed_ms = (now - self._timing_start) * 1000.0
        parts = [f"[TIMING][{self.role}]", event, f"elapsed_ms={elapsed_ms:.2f}"]
        if invite_start is not None:
            invite_elapsed_ms = (now - invite_start) * 1000.0
            parts.append(f"invite_elapsed_ms={invite_elapsed_ms:.2f}")
        print(" ".join(parts))

    def _get_memory_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss = float(usage.ru_maxrss)
            if sys.platform == "darwin":
                stats["rss_mb"] = rss / (1024.0 * 1024.0)
            else:
                stats["rss_mb"] = rss / 1024.0
        except Exception:
            pass

        if torch.cuda.is_available():
            stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        return stats

    def _log_resource(self, event: str) -> None:
        stats = self._get_memory_stats()
        parts = [f"[RESOURCE][{self.role}]", event]
        for key, value in stats.items():
            parts.append(f"{key}={value:.2f}MB")
        print(" ".join(parts))

    def _should_save_round(self, round_num: int) -> bool:
        return self.save_every > 0 and round_num % self.save_every == 0

    def _save_checkpoint(self, round_num: int) -> None:
        if not self._should_save_round(round_num):
            return
        if self.model is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.save_dir,
            f"{PIPELINE_ID}_{self.role}_round{round_num}.pth"
        )
        payload = {
            "pipeline_id": PIPELINE_ID,
            "round": round_num,
            "role": self.role,
            "device_id": self.device_id,
            "template_id": self.current_template_id,
            "split_key": self.current_split_key,
            "model_state": self.model.state_dict()
        }
        torch.save(payload, ckpt_path)
        print(f"[{self.role}] Saved checkpoint: {ckpt_path}")

    def _preload_model_if_needed(self) -> None:
        if self.model is not None:
            return
        template_id = self.current_template_id or DEFAULT_TEMPLATE_ID
        self._init_model_for_template({"template_id": template_id})
        self._log_timing("B_preload_done")
        self._log_resource("B_preload_done")

    def start(self):
        self._register_handlers()
        self._start_peer_listener()
        self._log_timing("A_listener_ready")
        self._log_resource("A_listener_ready")
        self._preload_model_if_needed()
        self._connect_to_server()
        self._register_with_server()
        self._start_async_workers()
        print(f"[{self.role}] Ready on {self.device_id} ({self.device})")

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
        if self.role == "device1":
            threading.Thread(target=self._activation_worker, daemon=True).start()
            if self.eval_batches > 0:
                threading.Thread(target=self._eval_worker, daemon=True).start()
        if self.role == "device0":
            threading.Thread(target=self._gradient_worker, daemon=True).start()

    def stop(self):
        if hasattr(self.bridge, 'message_router'):
            self.bridge.message_router.close()
        print(f"[{self.role}] Stopped")

    def _connect_to_server(self):
        remote_capabilities = _build_capabilities("server")
        endpoint = self.endpoint
        endpoint_desc = f"{getattr(endpoint, 'host', None)}:{getattr(endpoint, 'port', None)}"
        print(f"[{self.role}] Connecting to server {self.server_host}:{self.server_port} via {endpoint_desc}")
        try:
            ok = self.bridge.connect_to_platform(
                remote_node_id="server",
                remote_capabilities=remote_capabilities,
                network_endpoint=self.endpoint
            )
        except Exception as exc:
            print(f"[{self.role}] connect_to_platform failed: {exc}")
            raise
        if not ok:
            raise ConnectionError("Failed to connect to server")
        print(f"[{self.role}] Connected to server {self.server_host}:{self.server_port}")

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
            source_id=self.device_id,
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
        print(
            f"[{self.role}] Sending register to server {self.server_host}:{self.server_port} advertise={advertise_host}:{self.listen_port} device_id={self.device_id}"
        )
        try:
            self.bridge.send_cross_platform_message(msg)
        except Exception as exc:
            print(f"[{self.role}] send register failed: {exc}")
            raise
        print(f"[{self.role}] Register sent")

    def _init_model_for_template(self, template_payload: Optional[Dict[str, Any]] = None) -> None:
        if self.model is not None:
            return
        template_payload = template_payload or {}
        if "template_id" not in template_payload and self.current_template_id:
            template_payload = {**template_payload, "template_id": self.current_template_id}

        template_id, split_key, stage0, stage1 = build_model_split_from_template_payload(
            template_payload,
            default_template_id=DEFAULT_TEMPLATE_ID,
            num_classes=self.num_classes,
        )

        self.current_template_id = template_id
        self.current_split_key = split_key

        self.stage0, self.stage1 = stage0, stage1
        assert self.stage0 is not None and self.stage1 is not None
        if self.role == "device0":
            self.model = self.stage0.to(self.device)
        else:
            self.model = self.stage1.to(self.device)

        assert self.model is not None
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        print(f"[{self.role}] Initialized model split: template={template_id}, split={split_key}")

    def _handle_pipeline_invite(self, message: CrossPlatformMessage):
        payload = message.payload
        self.current_pipeline_id = payload.get("pipeline_id")
        invite_ts = time.perf_counter()
        self._invite_received_at = invite_ts

        # Get EdgePipe configuration
        self.edgepipe_config = payload.get("edgepipe_config", {})
        print(f"[{self.role}] EdgePipe config received: {self.edgepipe_config}")

        template_payload = payload.get("template") or {"template_id": payload.get("template_id")}
        if self.model is None:
            self._init_model_for_template(template_payload)
        self._log_timing("C_invite_model_ready", invite_start=invite_ts)
        self._log_resource("C_invite_model_ready")

        vehicles = payload.get("vehicles", [])
        stages = payload.get("stages", [])
        stage_map = {stages[i]: vehicles[i] for i in range(min(len(vehicles), len(stages)))}
        vehicle_map = {vehicles[i]: stages[i] for i in range(min(len(vehicles), len(stages)))}
        self.device0_id = stage_map.get("device0", self.device0_id)
        self.device1_id = stage_map.get("device1", self.device1_id)

        endpoints = payload.get("endpoints", {})
        for device_id, endpoint_info in endpoints.items():
            if device_id == self.device_id:
                continue
            host = endpoint_info.get("host")
            port = endpoint_info.get("port")
            if host and port:
                self.bridge.message_router.add_route(
                    device_id,
                    NetworkEndpoint(
                        host=host,
                        port=port,
                        protocol=TransportProtocol.TCP,
                        compression=CompressionType.NONE
                    )
                )
                ok = _probe_endpoint(host, int(port), label=f"peer {device_id}")
                if not ok:
                    print(
                        f"[{self.role}] Warning: peer {device_id} unreachable. "
                        f"Ensure {host}:{port} is reachable or use --advertise-host on peer."
                    )

        if self.device_id in vehicle_map:
            self.current_stage = vehicle_map.get(self.device_id, self.current_stage)

        accept = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.device_id,
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
        if self.role == "device1":
            self.sequence_registry.register_for_rounds(
                "activation",
                rounds,
                self.micro_batches,
                self._handle_activation_sequence
            )
            if self.eval_batches > 0:
                self.sequence_registry.register_for_rounds(
                    "eval",
                    rounds,
                    self.eval_batches,
                    self._handle_eval_activation_sequence
                )
        if self.role == "device0":
            self.sequence_registry.register_for_rounds(
                "gradient",
                rounds,
                self.micro_batches,
                self._handle_gradient_sequence
            )

    def _handle_pipeline_control(self, message: CrossPlatformMessage):
        if self.role != "device0":
            return
        payload = message.payload or {}
        if self.model is None:
            self._init_model_for_template()
            if self.model is None:
                print("[device0] Model not initialized, skip round")
                return
        round_num = payload.get("round", DEFAULT_ROUND)
        micro_batch = payload.get("micro_batch", DEFAULT_MICRO_BATCH)
        micro_batches = payload.get("micro_batches", self.micro_batches)
        threading.Thread(
            target=self._run_device0_round,
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

    @staticmethod
    def _to_tensor(value, device: str, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            if device:
                tensor = tensor.to(device)
            return tensor
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _estimate_payload_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            return nbytes
        if isinstance(value, dict):
            return sum(EdgePipeJetsonDevice._estimate_payload_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(EdgePipeJetsonDevice._estimate_payload_bytes(v) for v in value)
        return 0

    def _get_round_metrics(self, round_num: int) -> Dict[str, Any]:
        metrics = self._round_metrics.get(round_num)
        if metrics is None:
            metrics = {
                "round_start": None,
                "round_end": None,
                "images": 0,
                "activation_bytes_sent": 0,
                "activation_bytes_recv": 0,
                "grad_bytes_sent": 0,
                "grad_bytes_recv": 0,
                "wait_time_sec": 0.0,
            }
            self._round_metrics[round_num] = metrics
        return metrics

    def _run_device0_round(self, round_num: int, micro_batch: int, micro_batches: int):
        total_micro_batches = max(1, int(micro_batches))
        loader = _build_data_loader(
            self.dataset,
            DEFAULT_BATCH_SIZE * total_micro_batches,
            self.image_size,
            num_batches=4,
            data_dir=self.data_dir,
            download=self.download
        )
        images, labels = next(iter(loader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        with self._lock:
            metrics = self._get_round_metrics(round_num)
            if metrics["round_start"] is None:
                metrics["round_start"] = time.perf_counter()
            metrics["images"] += int(images.size(0))

        assert self.model is not None
        assert self.optimizer is not None
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
            print(f"[device0] 1F1B warmup: sending single micro-batch (round {round_num})")
        else:
            print(f"[device0] 1F1B warmup: sending first micro-batch (round {round_num})")

        schedule = OneFOneBSchedule(total_micro_batches)
        for micro_idx, (img_mb, lbl_mb) in enumerate(zip(image_chunks, label_chunks)):
            activation = self.model(img_mb)
            seq_id = self.sequence_id_factory.make(round_num, "activation", micro_idx)

            with self._lock:
                self._activation_cache[seq_id] = activation

            activation_payload, activation_info = self._lep_state.apply(
                activation,
                ENABLE_ACTIVATION_LEP,
                LEP_FP16_DTYPE,
            )

            pipeline_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="device0",
                source_id=self.device_id,
                target_id=self.device1_id,
                data={
                    "activation": activation_payload,
                    "activation_info": activation_info,
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

            endpoint = self.bridge.message_router.routing_table.get(self.device1_id)
            if endpoint:
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(activation_payload)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(lbl_mb)
                self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(pipeline_msg, endpoint)
                phase = schedule.phase(micro_idx).value
                if phase == "warmup":
                    print(f"[device0] Sent activation warmup (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                elif phase == "cooldown":
                    print(f"[device0] Sent activation cooldown (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                else:
                    print(f"[device0] Sent activation steady (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")

                if activation_info.get("lep_enabled") and self._lep_state.should_log(LEP_LOG_INTERVAL):
                    reduction = self._lep_state.reduction_ratio()
                    print(
                        f"[device0][LEP] steps={self._lep_state.steps} "
                        f"reduction={reduction * 100.0:.2f}% "
                        f"last_error_norm={activation_info.get('lep_error_norm'):.4f}"
                    )

        if ENABLE_ACTIVATION_LEP and self._lep_state.steps:
            reduction = self._lep_state.reduction_ratio()
            print(
                f"[device0][LEP] round={round_num} activation_bytes_sent={self._lep_state.bytes_sent} "
                f"baseline_bytes={self._lep_state.bytes_baseline} reduction={reduction * 100.0:.2f}%"
            )

        print(f"[device0] 1F1B cooldown: waiting for gradients (round {round_num})")
        wait_started = time.perf_counter()
        with self._lock:
            cond = self._round_cond.get(round_num)
            while not shutdown_requested and not self._round_state.get(round_num, {}).get("done"):
                if cond:
                    cond.wait(timeout=0.5)
                else:
                    break
                if time.perf_counter() - wait_started > DEFAULT_ROUND_TIMEOUT_SEC:
                    print(f"[device0] Round {round_num} timeout waiting for gradients")
                    break
            self._round_state.pop(round_num, None)
            self._round_cond.pop(round_num, None)
        with self._lock:
            metrics = self._get_round_metrics(round_num)
            metrics["round_end"] = time.perf_counter()
            metrics["wait_time_sec"] += max(0.0, time.perf_counter() - wait_started)
            round_start = metrics.get("round_start")
            round_end = metrics.get("round_end")
            total_images = metrics.get("images", 0)
            activation_bytes_sent = metrics.get("activation_bytes_sent", 0)
            grad_bytes_recv = metrics.get("grad_bytes_recv", 0)
            wait_time = metrics.get("wait_time_sec", 0.0)

        if round_start and round_end and round_end > round_start:
            round_time = round_end - round_start
            throughput = total_images / round_time if round_time > 0 else 0.0
            print(
                f"[device0][METRICS] round={round_num} time_sec={round_time:.2f} "
                f"throughput_img_s={throughput:.2f} images={total_images}"
            )
            print(
                f"[device0][METRICS] comm_bytes_sent={activation_bytes_sent} "
                f"comm_bytes_recv={grad_bytes_recv} wait_time_sec={wait_time:.2f}"
            )

        print(f"[device0] 1F1B cooldown: gradients complete (round {round_num})")
        if self.eval_batches > 0:
            self._run_eval_round(round_num)

    def _run_eval_round(self, round_num: int) -> None:
        if self.role != "device0":
            return
        loader = _build_eval_loader(
            self.dataset,
            DEFAULT_BATCH_SIZE,
            self.image_size,
            num_batches=self.eval_batches,
            data_dir=self.data_dir,
            download=self.download
        )
        if loader is None:
            return
        total_batches = len(loader)
        if total_batches == 0:
            print(f"[device0][EVAL] No eval batches available (round {round_num})")
            return

        assert self.model is not None
        self.model.eval()
        print(f"[device0][EVAL] Sending eval activations (round {round_num}, batches={total_batches})")

        for eval_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                activation = self.model(images)

            seq_id = self.sequence_id_factory.make(round_num, "eval", eval_idx)
            pipeline_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="device0",
                source_id=self.device_id,
                target_id=self.device1_id,
                data={
                    "activation": activation.detach(),
                    "activation_info": {"lep_enabled": False},
                    "labels": labels.detach(),
                    "round": round_num,
                    "eval": True,
                    "eval_batch": eval_idx,
                    "eval_batches": total_batches
                },
                data_type="eval_activation",
                sequence_id=seq_id,
                sequence_index=0,
                total_sequence_length=1,
                requires_ack=False,
                compression_type=CompressionType.NONE,
                serialization_format=SerializationFormat.PICKLE
            )
            endpoint = self.bridge.message_router.routing_table.get(self.device1_id)
            if endpoint:
                self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(pipeline_msg, endpoint)

    def _handle_activation_sequence(self, messages):
        pipeline_msg = messages[0]
        self._activation_queue.put(pipeline_msg)

    def _handle_eval_activation_sequence(self, messages):
        pipeline_msg = messages[0]
        self._eval_queue.put(pipeline_msg)

    def _activation_worker(self):
        while not shutdown_requested:
            try:
                pipeline_msg = self._activation_queue.get(timeout=0.5)
            except Empty:
                continue

            if self.model is None:
                self._init_model_for_template()
                if self.model is None:
                    print("[device1] Model not initialized, skip activation")
                    continue

            data = pipeline_msg.data
            activation_info = data.get("activation_info", {})
            activation_data = self._unwrap_tensor_payload(data["activation"])
            labels_data = self._unwrap_tensor_payload(data["labels"])
            with self._lock:
                metrics = self._get_round_metrics(data.get("round", DEFAULT_ROUND))
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(activation_data)
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(labels_data)
            if activation_info.get("lep_enabled"):
                error_norm = activation_info.get("lep_error_norm")
                error_norm_str = f"{error_norm:.4f}" if isinstance(error_norm, (int, float)) else "N/A"
                print(
                    f"[device1][LEP] recv activation dtype={activation_info.get('lep_dtype')} "
                    f"error_norm={error_norm_str}"
                )
            activation = self._to_tensor(activation_data, self.device, torch.float32)
            labels = self._to_tensor(labels_data, self.device, torch.long)
            activation.requires_grad_(True)

            assert self.model is not None
            assert self.optimizer is not None

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
                stage_id="device1",
                source_id=self.device_id,
                target_id=self.device0_id,
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

            endpoint = self.bridge.message_router.routing_table.get(self.device0_id)
            if endpoint:
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["grad_bytes_sent"] += self._estimate_payload_bytes(grad)
                self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(grad_msg, endpoint)
                print(
                    f"[device1] Sent gradient {phase} (loss={loss.item():.4f}, micro {micro_batch + 1}/{micro_batches})"
                )

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            with self._lock:
                state = self._round_state.get(round_num)
                if state:
                    state["received"] += 1
                    if state["received"] >= state["expected"]:
                        self.optimizer.step()
                        self._save_checkpoint(round_num)
                        self._round_state.pop(round_num, None)
                        metrics = self._get_round_metrics(round_num)
                        comm_sent = metrics.get("grad_bytes_sent", 0)
                        comm_recv = metrics.get("activation_bytes_recv", 0)
                        print(
                            f"[device1][METRICS] round={round_num} comm_bytes_sent={comm_sent} "
                            f"comm_bytes_recv={comm_recv}"
                        )
                        if self.auto_exit:
                            self.completed_rounds += 1
                            if self.completed_rounds >= self.total_rounds:
                                _request_shutdown()

    def _eval_worker(self):
        while not shutdown_requested:
            try:
                pipeline_msg = self._eval_queue.get(timeout=0.5)
            except Empty:
                continue

            if self.model is None:
                self._init_model_for_template()
                if self.model is None:
                    print("[device1] Model not initialized, skip eval")
                    continue

            data = pipeline_msg.data
            round_num = data.get("round", DEFAULT_ROUND)
            activation_data = self._unwrap_tensor_payload(data["activation"])
            labels_data = self._unwrap_tensor_payload(data["labels"])
            activation = self._to_tensor(activation_data, self.device, torch.float32)
            labels = self._to_tensor(labels_data, self.device, torch.long)

            with torch.no_grad():
                outputs = self.model(activation)
                preds = outputs.argmax(dim=1)
                correct = int((preds == labels).sum().item())
                total = int(labels.numel())

            with self._lock:
                state = self._eval_state.get(round_num)
                if state is None:
                    state = {
                        "correct": 0,
                        "total": 0,
                        "received": 0,
                        "expected": int(data.get("eval_batches", self.eval_batches))
                    }
                    self._eval_state[round_num] = state
                state["correct"] += correct
                state["total"] += total
                state["received"] += 1
                done = state["received"] >= state["expected"]

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            if done:
                accuracy = state["correct"] / max(1, state["total"])
                print(
                    f"[device1][EVAL] round={round_num} accuracy={accuracy:.4f} "
                    f"correct={state['correct']} total={state['total']}"
                )
                with self._lock:
                    self._eval_state.pop(round_num, None)

    def _handle_gradient_sequence(self, messages):
        pipeline_msg = messages[0]
        self._gradient_queue.put(pipeline_msg)

    def _gradient_worker(self):
        while not shutdown_requested:
            try:
                pipeline_msg = self._gradient_queue.get(timeout=0.5)
            except Empty:
                continue

            if self.model is None:
                self._init_model_for_template()
                if self.model is None:
                    print("[device0] Model not initialized, skip gradient")
                    continue

            data = pipeline_msg.data
            grad_data = self._unwrap_tensor_payload(data["grad"])

            assert self.model is not None
            assert self.optimizer is not None
            grad = self._to_tensor(grad_data, self.device, torch.float32)

            with self._lock:
                metrics = self._get_round_metrics(data.get("round", DEFAULT_ROUND))
                metrics["grad_bytes_recv"] += self._estimate_payload_bytes(grad_data)

            round_num = data.get("round", DEFAULT_ROUND)
            micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
            micro_batches = data.get("micro_batches", self.micro_batches)
            activation_seq_id = self.sequence_id_factory.make(round_num, "activation", micro_batch)
            with self._lock:
                activation = self._activation_cache.pop(activation_seq_id, None)

            if activation is None:
                print("[device0] Missing activation for gradient")
                continue

            activation.backward(grad)
            print("[device0] Applied gradient for micro-batch")

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            with self._lock:
                if round_num not in self._round_state:
                    self._round_state[round_num] = {"expected": int(micro_batches), "received": 0}
                state = self._round_state[round_num]
                state["received"] += 1
                if state["received"] >= state["expected"]:
                    self.optimizer.step()
                    self._save_checkpoint(round_num)
                    state["done"] = True
                    print("[device0] Applied all gradients and updated weights")

                    round_done = CrossPlatformMessage(
                        message_id=str(uuid.uuid4()),
                        source_id=self.device_id,
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

                    if self.auto_exit:
                        self.completed_rounds += 1
                        if self.completed_rounds >= self.total_rounds:
                            _request_shutdown()

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
    parser = argparse.ArgumentParser(description="EdgePipe Jetson Implementation (Two Jetson devices)")
    parser.add_argument("--mode", required=True, choices=["server", "device", "validate"], help="server, device, or validate")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--port", type=int, default=5000, help="server port")
    parser.add_argument("--server-host", default="localhost", help="server host (device mode)")
    parser.add_argument("--server-port", type=int, default=5000, help="server port (device mode)")
    parser.add_argument("--device-id", default="device_001", help="device id")
    parser.add_argument("--role", choices=["device0", "device1"], help="device role")
    parser.add_argument("--device0-id", default=DEFAULT_DEVICE0_ID, help="device0 device id")
    parser.add_argument("--device1-id", default=DEFAULT_DEVICE1_ID, help="device1 device id")
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID, help="pipeline template id")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="total training rounds")
    parser.add_argument("--micro-batches", type=int, default=DEFAULT_MICRO_BATCHES, help="micro-batches per round")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["cifar10"],
        help="dataset name"
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES, help="number of classes")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="input image size")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="dataset root directory")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES, help="eval batches per round (0 to disable)")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="save checkpoint every N rounds (0 to disable)")
    parser.add_argument("--save-dir", default=DEFAULT_CHECKPOINT_DIR, help="checkpoint output directory")
    parser.add_argument("--auto-exit", action="store_true", help="exit after completing all rounds")
    parser.add_argument("--listen-host", default="0.0.0.0", help="device listen host for peer pipeline data")
    parser.add_argument("--listen-port", type=int, default=0, help="device listen port for peer pipeline data")
    parser.add_argument("--advertise-host", default=None, help="host/IP to advertise to peers (default: auto-detect)")

    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.mode == "validate":
        _validate_pipeline_utils(args.rounds, args.micro_batches)
        return

    if args.mode == "server":
        server = EdgePipeJetsonServer(
            args.host,
            args.port,
            args.device0_id,
            args.device1_id,
            args.rounds,
            args.auto_exit,
            args.micro_batches,
            args.template_id
        )
        server.start()
        while not shutdown_requested:
            time.sleep(0.1)
        server.stop()
        return

    if args.mode == "device":
        if not args.role:
            raise ValueError("--role is required in device mode")
        if args.listen_port == 0:
            args.listen_port = 6000 if args.role == "device0" else 6001

        device = EdgePipeJetsonDevice(
            args.device_id,
            args.role,
            args.server_host,
            args.server_port,
            args.device0_id,
            args.device1_id,
            args.listen_host,
            args.listen_port,
            args.advertise_host,
            rounds=args.rounds,
            auto_exit=args.auto_exit,
            micro_batches=args.micro_batches,
            data_dir=args.data_dir,
            download=args.download,
            dataset=args.dataset,
            image_size=args.image_size,
            num_classes=args.num_classes,
            eval_batches=args.eval_batches,
            save_every=args.save_every,
            save_dir=args.save_dir
        )
        device.start()
        try:
            while not shutdown_requested:
                time.sleep(5.0)
        finally:
            device.stop()


if __name__ == "__main__":
    main()
