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
import traceback
import uuid
import argparse
import threading
import signal
import shutil
import tarfile
import zipfile
import urllib.request
from collections import OrderedDict
from queue import Queue, Empty
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

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
from fhdp.core.hardware_adapter import ComputeCapability, build_capabilities
from fhdp.core.pipeline_runtime import (
    SequenceIdFactory,
    SequenceHandlerRegistry,
    OneFOneBSchedule,
    get_micro_batch_phase,
    PerformanceCorrector,
)
from fhdp.core import ActivationLEPState
from fhdp.core.types import VehicleInfo
from fhdp.core.pipeline_model import (
    get_pipeline_template,
    serialize_template,
    build_model_split_from_template_payload,
)
from fhdp.core.data_loader import (
    build_data_loader,
    build_eval_loader,
)
from fhdp.core.resource_monitor import ResourceMonitor, RoundMetrics
from fhdp.core.tensor_utils import (
    pack_tensor_for_send,
    unpack_tensor_payload,
    decode_tensor_payload,
    to_tensor,
)
from fhdp.edge_server.template_manager import TemplateManager


PIPELINE_ID = "proto_resnet18"
DEFAULT_STAGE0_ID = "agx"
DEFAULT_STAGE1_ID = "orin"
DEFAULT_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_ROUND = 5
DEFAULT_ROUNDS = 5
DEFAULT_MICRO_BATCH = 12
DEFAULT_MICRO_BATCHES = 12
DEFAULT_DATA_DIR = os.path.join(script_dir, "data")
DEFAULT_TEMPLATE_ID = "vit_b16_2stage_v1"
AUTO_TEMPLATE_TOKEN = "auto"
DEFAULT_DATASET = "cifar10"
DEFAULT_TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_TINY_IMAGENET_DIR = "tiny-imagenet-200"
DEFAULT_EVAL_BATCHES = 4
DEFAULT_SAVE_EVERY = 10
DEFAULT_CHECKPOINT_DIR = os.path.join(project_root or script_dir, "logs", "checkpoints", "pipeline_proto")
PIPELINE_TIMING_SYNC = os.getenv("PIPELINE_TIMING_SYNC", "0") == "1"
PIPELINE_STEP_TIMEOUT_SEC = float(os.getenv("PIPELINE_STEP_TIMEOUT_SEC", "60"))

LEP_FP16_DTYPE = torch.float16
LEP_LOG_INTERVAL = 10


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


class PipelineProtoServer:
    def __init__(self, host: str, port: int, stage0_id: str, stage1_id: str,
                 rounds: int, auto_exit: bool, micro_batches: int, template_id: str):
        self.host = host
        self.port = port
        self.stage0_id = stage0_id
        self.stage1_id = stage1_id
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
        self.bridge = PlatformBridge(build_capabilities("server"), node_id="server")
        self.template_manager = TemplateManager()
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
        resources = message.payload.get("resources", {})

        self.registered[vehicle_id] = {
            "resources": resources,
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
        vehicles_info = []
        for vehicle_id in [self.stage0_id, self.stage1_id]:
            info = self.registered.get(vehicle_id, {})
            host = info.get("host")
            port = info.get("port")
            if host and port:
                endpoints[vehicle_id] = {"host": host, "port": port}
                
                # Create VehicleInfo object
                resources = info.get("resources", {})
                vehicle_info = VehicleInfo(
                    vehicle_id=vehicle_id,
                    position=(0, 0),  # placeholder
                    velocity=0,  # placeholder
                    direction=0,  # placeholder
                    resources=resources
                )
                vehicles_info.append(vehicle_info)
                print(f"[Server] Vehicle {vehicle_id} resources: gpu_type={resources.get('gpu_type', 'N/A')}, "
                      f"memory_gb={resources.get('memory_gb', 'N/A')}, compute_score={resources.get('compute_score', 'N/A')}")

        # Use template manager to find best template (unless template-id is explicitly set)
        template = None
        template_id = (self.template_id or "").strip()
        if template_id and template_id.lower() != AUTO_TEMPLATE_TOKEN:
            template = get_pipeline_template(template_id, DEFAULT_TEMPLATE_ID)
        else:
            template = self.template_manager.find_template_for_vehicles(vehicles_info)
            if not template:
                # If no suitable template found, use default template
                template = get_pipeline_template(DEFAULT_TEMPLATE_ID, DEFAULT_TEMPLATE_ID)
        
        template_payload = serialize_template(template)
        
        # Output model partition information
        print(f"[Server] Selected template: {template.template_id}")
        print(f"[Server] Model partition info:")
        model_partition = template.model_partition or {}
        if model_partition:
            print(f"  - model_name: {model_partition.get('model_name', 'N/A')}")
            print(f"  - split_key: {model_partition.get('split_key', 'N/A')}")
            print(f"  - stages: {model_partition.get('stages', [])}")
            if "description" in model_partition:
                print(f"  - description: {model_partition.get('description')}")
        else:
            print(f"  - No model partition info in template")
        print(f"[Server] Resource requirements: {[r.value for r in template.resource_requirements]}")
        print(f"[Server] Communication pattern: {template.communication_pattern}")

        invite = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id="",
            message_type="pipeline_invite",
            payload={
                "pipeline_id": PIPELINE_ID,
                "template_id": template.template_id,
                "template": template_payload,
                "vehicles": [self.stage0_id, self.stage1_id],
                "stages": ["stage0", "stage1"],
                "resource_requirements": [r.value for r in template.resource_requirements],
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
        
        # Dynamically adjust micro-batches based on vehicle resources
        adjusted_micro_batches = self.micro_batches
        
        # Check vehicle resource status
        for vehicle_id in [self.stage0_id, self.stage1_id]:
            info = self.registered.get(vehicle_id, {})
            resources = info.get("resources", {})
            # If resources are low, reduce micro-batch count
            if resources.get("memory", 1) < 0.5:
                adjusted_micro_batches = max(1, adjusted_micro_batches // 2)
                break
        
        control = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id=self.stage0_id,
            message_type="pipeline_control",
            payload={
                "pipeline_id": PIPELINE_ID,
                "round": self.current_round,
                "micro_batch": DEFAULT_MICRO_BATCH,
                "micro_batches": adjusted_micro_batches
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(control)
        print(f"[Server] Sent start_round to {self.stage0_id} (round {self.current_round}/{self.rounds}, micro_batches={adjusted_micro_batches})")


# ----------------- Vehicle -----------------

class PipelineProtoVehicle:
    _pinned_buffer_cache: "OrderedDict[Tuple[torch.dtype, Tuple[int, ...]], torch.Tensor]" = OrderedDict()
    _pinned_buffer_lock = threading.Lock()

    def __init__(self, vehicle_id: str, role: str, server_host: str, server_port: int,
                 stage0_id: str, stage1_id: str, listen_host: str, listen_port: int,
                 advertise_host: Optional[str] = None, rounds: int = DEFAULT_ROUNDS,
                 auto_exit: bool = False, micro_batches: int = DEFAULT_MICRO_BATCHES,
                 data_dir: str = DEFAULT_DATA_DIR, download: bool = False,
                 dataset: str = DEFAULT_DATASET, image_size: int = DEFAULT_IMAGE_SIZE,
                 num_classes: int = DEFAULT_NUM_CLASSES,
                 tiny_imagenet_url: str = DEFAULT_TINY_IMAGENET_URL,
                 tiny_imagenet_dir: str = DEFAULT_TINY_IMAGENET_DIR,
                 eval_batches: int = DEFAULT_EVAL_BATCHES,
                 save_every: int = DEFAULT_SAVE_EVERY, save_dir: str = DEFAULT_CHECKPOINT_DIR):
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
        self.dataset = dataset
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.tiny_imagenet_url = tiny_imagenet_url
        self.tiny_imagenet_dir = tiny_imagenet_dir
        self.eval_batches = max(0, int(eval_batches))
        self.save_every = max(0, int(save_every))
        self.save_dir = save_dir
        self.batch_size = max(1, int(DEFAULT_BATCH_SIZE))
        self._next_batch_size: Optional[int] = None
        self._next_micro_batches: Optional[int] = None
        self.corrector: Optional[FHDP_PerformanceCorrector] = None
        self.corrector_enabled = ENABLE_ONLINE_CORRECTOR
        self.device = _get_device()
        self._timing_start = time.perf_counter()
        self._invite_received_at: Optional[float] = None

        self.current_pipeline_id: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.current_template_id: Optional[str] = None
        self.current_split_key: Optional[str] = None

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
        self.bridge = PlatformBridge(build_capabilities(role), node_id=vehicle_id)

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
        
        # CUDA stream for async GPU-CPU transfers
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream()

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

    def _get_round_metrics(self, round_num: int) -> Dict[str, Any]:
        metrics = self._round_metrics.get(round_num)
        if metrics is None:
            metrics = {
                "activation_bytes_sent": 0,
                "activation_bytes_recv": 0,
                "grad_bytes_sent": 0,
                "grad_bytes_recv": 0,
                "activation_send_ts": {},
                "grad_latency_ms": [],
                "stage1_decode_ms": [],
                "stage1_compute_ms": [],
                "stage1_send_ms": [],
                "stage1_e2e_ms": [],
                "stage0_apply_ms": [],
                "stage0_step_ms": [],
            }
            self._round_metrics[round_num] = metrics
        return metrics

    @staticmethod
    def _estimate_payload_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value)
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            return nbytes
        if isinstance(value, dict):
            return sum(PipelineProtoVehicle._estimate_payload_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(PipelineProtoVehicle._estimate_payload_bytes(v) for v in value)
        return 0

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
            "device_id": self.vehicle_id,
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
            if self.eval_batches > 0:
                threading.Thread(target=self._eval_worker, daemon=True).start()
        if self.role == "stage0":
            threading.Thread(target=self._gradient_worker, daemon=True).start()

    def stop(self):
        if hasattr(self.bridge, 'message_router'):
            self.bridge.message_router.close()
        print(f"[{self.role}] Stopped")

    def _connect_to_server(self):
        remote_capabilities = build_capabilities("server")
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
        
        # Collect actual device resources
        resources = {
            "role": self.role,
            "gpu": 1,
            "memory": 1,
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            resources["gpu_type"] = gpu_name
            resources["memory_gb"] = round(gpu_memory, 1)
            
            # Compute score based on GPU type
            gpu_name_lower = gpu_name.lower()
            # Estimate based on memory
            resources["compute_score"] = min(1.0, gpu_memory / 32.0)
        
        msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="register",
            payload={
                "resources": resources
            },
            metadata={
                "client_host": advertise_host,
                "client_port": self.listen_port
            },
            requires_ack=False
        )
        self.bridge.send_cross_platform_message(msg)
        print(f"[{self.role}] Registered with resources: {resources}")

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
        
        # Output model partition details
        model_partition = template_payload.get("model_partition", {})
        print(f"[{self.role}] Model partition details:")
        print(f"  - template_id: {template_id}")
        print(f"  - split_key: {split_key}")
        if model_partition:
            print(f"  - model_name: {model_partition.get('model_name', 'N/A')}")
            print(f"  - stages: {model_partition.get('stages', [])}")

        # Select corresponding model stage based on role
        if self.role == "stage0":
            self.model = stage0.to(self.device)
            print(f"[{self.role}] Initialized stage0 model: template={template_id}, split={split_key}")
            # Output stage0 model architecture summary
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[{self.role}] Stage0 model parameters: {total_params:,}")
        else:
            self.model = stage1.to(self.device)
            print(f"[{self.role}] Initialized stage1 model: template={template_id}, split={split_key}")
            # Output stage1 model architecture summary
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"[{self.role}] Stage1 model parameters: {total_params:,}")

        assert self.model is not None
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        print(f"[{self.role}] Initialized model optimizer")

    def _handle_pipeline_invite(self, message: CrossPlatformMessage):
        payload = message.payload
        self.current_pipeline_id = payload.get("pipeline_id")
        invite_ts = time.perf_counter()
        self._invite_received_at = invite_ts

        template_payload = payload.get("template") or {"template_id": payload.get("template_id")}
        if self.model is None:
            self._init_model_for_template(template_payload)
        else:
            # If model is initialized but template is different, reinitialize
            current_template = template_payload.get("template_id")
            if current_template and current_template != self.current_template_id:
                self._init_model_for_template(template_payload)

        if self.role == "stage0" and self.corrector is None and self.corrector_enabled:
            self.corrector = FHDP_PerformanceCorrector(
                batch_bounds=CORRECTOR_BATCH_BOUNDS,
                micro_bounds=CORRECTOR_MICRO_BATCH_BOUNDS,
                batch_step=CORRECTOR_BATCH_STEP,
                micro_step=CORRECTOR_MICRO_STEP,
                down_threshold=CORRECTOR_WAIT_RATIO_DOWN,
                up_threshold=CORRECTOR_WAIT_RATIO_UP
            )
            print(
                f"[stage0][Corrector] init bounds(batch={CORRECTOR_BATCH_BOUNDS}, "
                f"micro={CORRECTOR_MICRO_BATCH_BOUNDS}) thresholds(down={CORRECTOR_WAIT_RATIO_DOWN}, "
                f"up={CORRECTOR_WAIT_RATIO_UP})"
            )
        
        self._log_timing("C_invite_model_ready", invite_start=invite_ts)
        self._log_resource("C_invite_model_ready")

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
            if self.eval_batches > 0:
                self.sequence_registry.register_for_rounds(
                    "eval",
                    rounds,
                    self.eval_batches,
                    self._handle_eval_activation_sequence
                )
        if self.role == "stage0":
            self.sequence_registry.register_for_rounds(
                "gradient",
                rounds,
                self.micro_batches,
                self._handle_gradient_sequence
            )

    def _handle_pipeline_control(self, message: CrossPlatformMessage):
        payload = message.payload or {}
        round_num = payload.get("round", DEFAULT_ROUND)
        micro_batch = payload.get("micro_batch", DEFAULT_MICRO_BATCH)
        micro_batches = payload.get("micro_batches", self.micro_batches)

        if self.role == "stage1":
            self.micro_batches = max(1, int(micro_batches))
            self.sequence_registry.register_for_rounds(
                "activation",
                [round_num],
                self.micro_batches,
                self._handle_activation_sequence
            )
            print(f"[stage1] Sync round {round_num}: micro_batches={self.micro_batches}")
            return

        if self.role != "stage0":
            return

        if self.model is None:
            self._init_model_for_template()
            if self.model is None:
                print("[stage0] Model not initialized, skip round")
                return

        batch_size = self.batch_size

        if self._next_micro_batches is not None:
            micro_batches = self._next_micro_batches
            self._next_micro_batches = None
        if self._next_batch_size is not None:
            batch_size = self._next_batch_size
            self._next_batch_size = None

        self.micro_batches = max(1, int(micro_batches))
        self.batch_size = max(1, int(batch_size))

        if self.stage1_id:
            control = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id=self.vehicle_id,
                target_id=self.stage1_id,
                message_type="pipeline_control",
                payload={
                    "pipeline_id": PIPELINE_ID,
                    "round": round_num,
                    "micro_batch": micro_batch,
                    "micro_batches": self.micro_batches
                },
                requires_ack=False
            )
            self.bridge.send_cross_platform_message(control)
            print(
                f"[stage0] Sync micro_batches={self.micro_batches} to {self.stage1_id} "
                f"(round {round_num})"
            )

        print(
            f"[stage0] Start round {round_num}: batch_size={self.batch_size}, "
            f"micro_batches={self.micro_batches}"
        )

        threading.Thread(
            target=self._run_stage0_round,
            args=(round_num, micro_batch, self.micro_batches, self.batch_size),
            daemon=True
        ).start()

    def _handle_status(self, message: CrossPlatformMessage):
        print(f"[{self.role}] Server status: {message.payload}")

    def _unwrap_tensor_payload(self, value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            try:
                result = self.bridge.message_router.serialization_manager.deserialize_tensor_zero_copy(data)
            except Exception:
                result = None
            if isinstance(result, (bytes, bytearray, memoryview)) or result is None:
                try:
                    return self.bridge.message_router.serialization_manager.pickle_loads(data)
                except Exception:
                    return value
            return result
        if isinstance(value, dict):
            if value.get("__tensor_zero_copy__") and "data" in value:
                data = value.get("data")
                if data is None:
                    return value
                try:
                    result = self.bridge.message_router.serialization_manager.deserialize_tensor_zero_copy(data)
                except Exception:
                    result = None
                if isinstance(result, (bytes, bytearray, memoryview)) or result is None:
                    try:
                        return self.bridge.message_router.serialization_manager.pickle_loads(data)
                    except Exception:
                        return data
                return result
            if value.get("__tensor_metadata__") and "data" in value:
                return value["data"]
            if "data" in value and value.get("__tensor_bytes__"):
                return value["data"]
        return value

    def _decode_tensor_payload(self, value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            try:
                result = self.bridge.message_router.serialization_manager.deserialize_tensor_zero_copy(data)
            except Exception:
                result = None
            if isinstance(result, (bytes, bytearray, memoryview)) or result is None:
                try:
                    return self.bridge.message_router.serialization_manager.pickle_loads(data)
                except Exception:
                    return value
            return result
        return value

    def _get_pinned_buffer(self, shape, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not PACK_TENSOR_USE_PINNED or not torch.cuda.is_available():
            return None
        shape_tuple = tuple(shape) if shape is not None else ()
        numel = 1
        for dim in shape_tuple:
            numel *= int(dim)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        bytes_needed = numel * elem_size
        if bytes_needed > PINNED_BUFFER_MAX_BYTES:
            return None
        key = (dtype, shape_tuple)
        with self._pinned_buffer_lock:
            buffer = self._pinned_buffer_cache.pop(key, None)
            if buffer is None or buffer.numel() != numel or buffer.dtype != dtype:
                buffer = torch.empty(shape_tuple, dtype=dtype, device="cpu", pin_memory=True)
            self._pinned_buffer_cache[key] = buffer
            while len(self._pinned_buffer_cache) > PINNED_BUFFER_CACHE_SIZE:
                self._pinned_buffer_cache.popitem(last=False)
        return buffer

    def _copy_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu":
            return tensor
        src = tensor.detach()
        if not src.is_contiguous():
            src = src.contiguous()
        buffer = self._get_pinned_buffer(src.shape, src.dtype)
        if buffer is not None:
            buffer.copy_(src, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return buffer
        return src.to("cpu", non_blocking=True).contiguous()

    def _pack_tensor_zero_copy(self, tensor: torch.Tensor):
        if not PACK_TENSOR_ZERO_COPY:
            return tensor
        try:
            serialized = self.bridge.message_router.serialization_manager.serialize_tensor_zero_copy(tensor)
            return {
                "__tensor_zero_copy__": True,
                "data": serialized,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype)
            }
        except Exception:
            return tensor

    def _pack_tensor_for_send(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if PACK_TENSOR_FASTPATH:
                if tensor.device.type == "cpu" and tensor.is_contiguous():
                    return self._pack_tensor_zero_copy(tensor)
                if tensor.device.type == "cuda":
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    return self._pack_tensor_zero_copy(tensor)
            cpu_tensor = self._copy_tensor_to_cpu(tensor)
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()
            return self._pack_tensor_zero_copy(cpu_tensor)
            tensor = value.detach()
            if tensor.is_cuda:
                if non_blocking:
                    tensor = tensor.contiguous()
                else:
                    tensor = tensor.contiguous().to("cpu", non_blocking=False)
            return tensor
        if isinstance(value, dict):
            return {k: self._pack_tensor_for_send(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            packed = [self._pack_tensor_for_send(v) for v in value]
            return tuple(packed) if isinstance(value, tuple) else packed
        return value

    def _async_transfer_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Async transfer tensor from GPU to CPU using CUDA stream"""
        if not tensor.is_cuda:
            return tensor
        
        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                cpu_tensor = tensor.detach().contiguous().to("cpu", non_blocking=True)
            self._transfer_stream.synchronize()
            return cpu_tensor
        else:
            return tensor.detach().contiguous().to("cpu", non_blocking=True)

    def _run_stage0_round(self, round_num: int, micro_batch: int, micro_batches: int, batch_size: int):
        total_micro_batches = max(1, int(micro_batches))
        batch_size_per_micro = max(1, int(batch_size))
        loader = _build_data_loader(
            self.dataset,
            batch_size_per_micro * total_micro_batches,
            self.image_size,
            num_batches=4,
            data_dir=self.data_dir,
            download=self.download,
            tiny_imagenet_url=self.tiny_imagenet_url,
            tiny_imagenet_dir=self.tiny_imagenet_dir
        )
        images, labels = next(iter(loader))
        images = images.to(self.device)
        labels = labels.to(self.device)
        round_start = time.perf_counter()
        total_images = int(images.size(0))

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
            print(f"[stage0] 1F1B warmup: sending single micro-batch (round {round_num})")
        else:
            print(f"[stage0] 1F1B warmup: sending first micro-batch (round {round_num})")

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
            activation_payload = self._pack_tensor_for_send(activation_payload)
            labels_payload = self._pack_tensor_for_send(lbl_mb)

            pipeline_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="stage0",
                source_id=self.vehicle_id,
                target_id=self.stage1_id,
                data={
                    "activation": activation_payload,
                    "activation_info": activation_info,
                    "labels": labels_payload,
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
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(activation_payload)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(labels_payload)
                    metrics["activation_send_ts"][micro_idx] = time.perf_counter()
                self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(pipeline_msg, endpoint)
                phase = schedule.phase(micro_idx).value
                if phase == "warmup":
                    print(f"[stage0] Sent activation warmup (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                elif phase == "cooldown":
                    print(f"[stage0] Sent activation cooldown (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")
                else:
                    print(f"[stage0] Sent activation steady (round {round_num}, micro {micro_idx + 1}/{total_micro_batches})")

                if activation_info.get("lep_enabled") and self._lep_state.should_log(LEP_LOG_INTERVAL):
                    reduction = self._lep_state.reduction_ratio()
                    print(
                        f"[stage0][LEP] steps={self._lep_state.steps} "
                        f"reduction={reduction * 100.0:.2f}% "
                        f"last_error_norm={activation_info.get('lep_error_norm'):.4f}"
                    )

        if ENABLE_ACTIVATION_LEP and self._lep_state.steps:
            reduction = self._lep_state.reduction_ratio()
            print(
                f"[stage0][LEP] round={round_num} activation_bytes_sent={self._lep_state.bytes_sent} "
                f"baseline_bytes={self._lep_state.bytes_baseline} reduction={reduction * 100.0:.2f}%"
            )

        print(f"[stage0] 1F1B cooldown: waiting for gradients (round {round_num})")
        wait_started = time.perf_counter()
        last_wait_log = wait_started
        with self._lock:
            cond = self._round_cond.get(round_num)
            while not shutdown_requested and not self._round_state.get(round_num, {}).get("done"):
                now = time.perf_counter()
                if now - last_wait_log >= 5.0:
                    state = self._round_state.get(round_num, {})
                    print(
                        f"[stage0][WAIT] round={round_num} received={state.get('received', 0)}/"
                        f"{state.get('expected', '?')}"
                    )
                    last_wait_log = now
                if cond:
                    cond.wait(timeout=0.5)
                else:
                    break
            self._round_state.pop(round_num, None)
            self._round_cond.pop(round_num, None)
        round_end = time.perf_counter()
        wait_time = max(0.0, round_end - wait_started)
        round_time = round_end - round_start
        if round_time > 0:
            throughput = total_images / round_time
            wait_ratio = (wait_time / round_time) * 100.0
            print(
                f"[stage0][METRICS] round={round_num} time_sec={round_time:.2f} "
                f"throughput_img_s={throughput:.2f} images={total_images}"
            )
            print(
                f"[stage0][METRICS] network_wait_ratio={wait_ratio:.2f}% "
                f"wait_time_sec={wait_time:.2f}"
            )
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                comm_sent = metrics.get("activation_bytes_sent", 0)
                comm_recv = metrics.get("grad_bytes_recv", 0)
                grad_latency_list = metrics.get("grad_latency_ms", [])
                apply_ms_list = metrics.get("stage0_apply_ms", [])
                step_ms_list = metrics.get("stage0_step_ms", [])
                avg_grad_latency = sum(grad_latency_list) / max(1, len(grad_latency_list))
                avg_apply_ms = sum(apply_ms_list) / max(1, len(apply_ms_list))
                avg_step_ms = sum(step_ms_list) / max(1, len(step_ms_list))
            print(
                f"[stage0][METRICS] comm_bytes_sent={comm_sent} "
                f"comm_bytes_recv={comm_recv}"
            )
            print(
                f"[stage0][METRICS] round={round_num} avg_grad_latency_ms={avg_grad_latency:.2f} "
                f"avg_apply_ms={avg_apply_ms:.2f} avg_step_ms={avg_step_ms:.2f}"
            )

            if self.role == "stage0" and self.corrector and self.corrector_enabled:
                wait_ratio_norm = self.corrector.update_online(wait_ratio)
                next_batch, next_micro, action = self.corrector.suggest_params(
                    self.batch_size,
                    self.micro_batches
                )
                if next_batch != self.batch_size or next_micro != self.micro_batches:
                    self._next_batch_size = next_batch
                    self._next_micro_batches = next_micro
                print(
                    f"[stage0][Corrector] round={round_num} wait_ratio_pct={wait_ratio:.2f}% "
                    f"ratio={wait_ratio_norm:.2f} ema={self.corrector.wait_ratio_ema:.2f} "
                    f"action={action} next_batch={next_batch} next_micro={next_micro}"
                )

            with self._lock:
                self._round_metrics.pop(round_num, None)
        print(f"[stage0] 1F1B cooldown: gradients complete (round {round_num})")
        if self.eval_batches > 0:
            self._run_eval_round(round_num)

    def _run_eval_round(self, round_num: int) -> None:
        if self.role != "stage0":
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
            print(f"[stage0][EVAL] No eval batches available (round {round_num})")
            return

        assert self.model is not None
        self.model.eval()
        print(f"[stage0][EVAL] Sending eval activations (round {round_num}, batches={total_batches})")

        for eval_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                activation = self.model(images)

            activation_payload = self._pack_tensor_for_send(activation)
            labels_payload = self._pack_tensor_for_send(labels)

            seq_id = self.sequence_id_factory.make(round_num, "eval", eval_idx)
            pipeline_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="stage0",
                source_id=self.vehicle_id,
                target_id=self.stage1_id,
                data={
                    "activation": activation_payload,
                    "activation_info": {"lep_enabled": False},
                    "labels": labels_payload,
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
            endpoint = self.bridge.message_router.routing_table.get(self.stage1_id)
            if endpoint:
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(activation_payload)
                    metrics["activation_bytes_sent"] += self._estimate_payload_bytes(labels_payload)
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
                    print("[stage1] Model not initialized, skip activation")
                    continue

            data = pipeline_msg.data
            round_num = data.get("round", DEFAULT_ROUND)
            activation_info = data.get("activation_info", {})
            t_start = time.perf_counter()
            decode_start = t_start
            activation_data = self._unwrap_tensor_payload(data["activation"])
            labels_data = self._unwrap_tensor_payload(data["labels"])
            activation_data = self._decode_tensor_payload(activation_data)
            labels_data = self._decode_tensor_payload(labels_data)
            decode_end = time.perf_counter()
            decode_ms = (decode_end - decode_start) * 1000.0
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(activation_data)
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(labels_data)
                metrics["stage1_decode_ms"].append(decode_ms)
            if activation_info.get("lep_enabled"):
                error_norm = activation_info.get("lep_error_norm")
                error_norm_str = f"{error_norm:.4f}" if isinstance(error_norm, (int, float)) else "N/A"
                print(
                    f"[stage1][LEP] recv activation dtype={activation_info.get('lep_dtype')} "
                    f"error_norm={error_norm_str}"
                )
            activation = torch.as_tensor(activation_data, dtype=torch.float32).to(self.device, non_blocking=True)
            labels = torch.as_tensor(labels_data, dtype=torch.long).to(self.device, non_blocking=True)
            activation.requires_grad_(True)

            assert self.model is not None
            assert self.optimizer is not None

            micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
            micro_batches = data.get("micro_batches", self.micro_batches)

            with self._lock:
                if round_num not in self._round_state:
                    self._round_state[round_num] = {"expected": int(micro_batches), "received": 0}
                    self.model.train()
                    self.optimizer.zero_grad()

            phase = get_micro_batch_phase(micro_batch, micro_batches).value

            compute_start = time.perf_counter()
            outputs = self.model(activation)
            loss = self.criterion(outputs, labels)
            loss.backward()
            compute_end = time.perf_counter()
            compute_ms = (compute_end - compute_start) * 1000.0

            grad = activation.grad.detach()
            seq_id = self.sequence_id_factory.make(round_num, "gradient", micro_batch)

            grad_payload = self._pack_tensor_for_send(grad)
            grad_msg = PipelineMessage(
                pipeline_id=PIPELINE_ID,
                stage_id="stage1",
                source_id=self.vehicle_id,
                target_id=self.stage0_id,
                data={
                    "grad": grad_payload,
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
                send_start = time.perf_counter()
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["grad_bytes_sent"] += self._estimate_payload_bytes(grad_payload)
                self.bridge.message_router.pipeline_comm_manager.send_pipeline_data(grad_msg, endpoint)
                send_end = time.perf_counter()
                send_ms = (send_end - send_start) * 1000.0
                e2e_ms = (send_end - t_start) * 1000.0
                with self._lock:
                    metrics = self._get_round_metrics(round_num)
                    metrics["stage1_compute_ms"].append(compute_ms)
                    metrics["stage1_send_ms"].append(send_ms)
                    metrics["stage1_e2e_ms"].append(e2e_ms)
                print(
                    f"[stage1] Sent gradient {phase} (loss={loss.item():.4f}, micro {micro_batch + 1}/{micro_batches})"
                )
                print(
                    f"[stage1][TIMING] round={round_num} micro={micro_batch + 1}/{micro_batches} "
                    f"decode_ms={decode_ms:.2f} compute_ms={compute_ms:.2f} send_ms={send_ms:.2f} e2e_ms={e2e_ms:.2f}"
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
                        decode_ms_list = metrics.get("stage1_decode_ms", [])
                        compute_ms_list = metrics.get("stage1_compute_ms", [])
                        send_ms_list = metrics.get("stage1_send_ms", [])
                        e2e_ms_list = metrics.get("stage1_e2e_ms", [])
                        avg_decode = sum(decode_ms_list) / max(1, len(decode_ms_list))
                        avg_compute = sum(compute_ms_list) / max(1, len(compute_ms_list))
                        avg_send = sum(send_ms_list) / max(1, len(send_ms_list))
                        avg_e2e = sum(e2e_ms_list) / max(1, len(e2e_ms_list))
                        print(
                            f"[stage1][METRICS] round={round_num} comm_bytes_sent={comm_sent} "
                            f"comm_bytes_recv={comm_recv}"
                        )
                        print(
                            f"[stage1][METRICS] round={round_num} "
                            f"avg_decode_ms={avg_decode:.2f} avg_compute_ms={avg_compute:.2f} "
                            f"avg_send_ms={avg_send:.2f} avg_e2e_ms={avg_e2e:.2f}"
                        )
                        self._round_metrics.pop(round_num, None)
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
                    print("[stage1] Model not initialized, skip eval")
                    continue

            data = pipeline_msg.data
            round_num = data.get("round", DEFAULT_ROUND)
            activation_data = self._unwrap_tensor_payload(data["activation"])
            labels_data = self._unwrap_tensor_payload(data["labels"])
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(activation_data)
                metrics["activation_bytes_recv"] += self._estimate_payload_bytes(labels_data)
            activation = to_tensor(activation_data, self.device, torch.float32)
            labels = to_tensor(labels_data, self.device, torch.long)

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
                    f"[stage1][EVAL] round={round_num} accuracy={accuracy:.4f} "
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
                    print("[stage0] Model not initialized, skip gradient")
                    continue

            data = pipeline_msg.data
            round_num = data.get("round", DEFAULT_ROUND)
            t_grad_recv = time.perf_counter()
            grad_data = self._unwrap_tensor_payload(data["grad"])
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                metrics["grad_bytes_recv"] += self._estimate_payload_bytes(grad_data)

            assert self.model is not None
            assert self.optimizer is not None
            grad = to_tensor(grad_data, self.device, torch.float32)

            micro_batch = data.get("micro_batch", DEFAULT_MICRO_BATCH)
            micro_batches = data.get("micro_batches", self.micro_batches)
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                send_ts = metrics.get("activation_send_ts", {}).pop(micro_batch, None)
                if send_ts is not None:
                    latency_ms = (t_grad_recv - send_ts) * 1000.0
                    metrics["grad_latency_ms"].append(latency_ms)
                else:
                    latency_ms = None
            if latency_ms is not None:
                print(
                    f"[stage0][LATENCY] round={round_num} micro={micro_batch + 1}/{micro_batches} "
                    f"grad_latency_ms={latency_ms:.2f}"
                )
            activation_seq_id = self.sequence_id_factory.make(round_num, "activation", micro_batch)
            with self._lock:
                activation = self._activation_cache.pop(activation_seq_id, None)

            if activation is None:
                print("[stage0] Missing activation for gradient")
                continue

            if PIPELINE_TIMING_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize()
            apply_start = time.perf_counter()
            activation.backward(grad)
            if PIPELINE_TIMING_SYNC and torch.cuda.is_available():
                torch.cuda.synchronize()
            apply_end = time.perf_counter()
            apply_ms = (apply_end - apply_start) * 1000.0
            with self._lock:
                metrics = self._get_round_metrics(round_num)
                metrics["stage0_apply_ms"].append(apply_ms)
            print("[stage0] Applied gradient for micro-batch")
            print(
                f"[stage0][TIMING] round={round_num} micro={micro_batch + 1}/{micro_batches} "
                f"apply_ms={apply_ms:.2f}"
            )

            self.sequence_registry.unregister_sequence(pipeline_msg.sequence_id)

            with self._lock:
                if round_num not in self._round_state:
                    self._round_state[round_num] = {"expected": int(micro_batches), "received": 0}
                state = self._round_state[round_num]
                state["received"] += 1
                received = state["received"]
                expected = state["expected"]
                if received >= expected and not state.get("step_started"):
                    state["step_started"] = True
                    state["done"] = True

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

                    if self.auto_exit:
                        self.completed_rounds += 1
                        if self.completed_rounds >= self.total_rounds:
                            _request_shutdown()

                    cond = self._round_cond.get(round_num)
                    if cond:
                        cond.notify_all()

                    print(
                        f"[stage0][DEBUG] round={round_num} all gradients received "
                        f"({received}/{expected}), starting optimizer step"
                    )

                    step_done = threading.Event()

                    def _run_step():
                        try:
                            if PIPELINE_TIMING_SYNC and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            step_start = time.perf_counter()
                            self.optimizer.step()
                            if PIPELINE_TIMING_SYNC and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            step_end = time.perf_counter()
                            step_ms = (step_end - step_start) * 1000.0
                            with self._lock:
                                metrics = self._round_metrics.get(round_num)
                                if metrics is not None:
                                    metrics["stage0_step_ms"].append(step_ms)
                            self._save_checkpoint(round_num)
                            print("[stage0] Applied all gradients and updated weights")
                            print(f"[stage0][TIMING] round={round_num} optimizer_step_ms={step_ms:.2f}")
                        except Exception:
                            print(f"[stage0][ERROR] optimizer step failed (round {round_num})")
                            traceback.print_exc()
                        finally:
                            step_done.set()

                    def _step_watchdog():
                        if PIPELINE_STEP_TIMEOUT_SEC <= 0:
                            return
                        time.sleep(PIPELINE_STEP_TIMEOUT_SEC)
                        if not step_done.is_set():
                            print(
                                f"[stage0][ERROR] optimizer step timeout after "
                                f"{PIPELINE_STEP_TIMEOUT_SEC:.0f}s (round {round_num})"
                            )
                            _request_shutdown()

                    if PIPELINE_STEP_TIMEOUT_SEC > 0:
                        threading.Thread(target=_step_watchdog, daemon=True).start()
                    threading.Thread(target=_run_step, daemon=True).start()

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
    parser = argparse.ArgumentParser(description="FHDP Pipeline Prototype (ResNet/ViT, micro-batching + 1F1B)")
    parser.add_argument("--mode", required=True, choices=["server", "vehicle", "validate"], help="server, vehicle, or validate")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--port", type=int, default=5000, help="server port")
    parser.add_argument("--server-host", default="localhost", help="server host (vehicle mode)")
    parser.add_argument("--server-port", type=int, default=5000, help="server port (vehicle mode)")
    parser.add_argument("--vehicle-id", default="vehicle_001", help="vehicle id")
    parser.add_argument("--role", choices=["stage0", "stage1"], help="vehicle role")
    parser.add_argument("--stage0-id", default=DEFAULT_STAGE0_ID, help="stage0 vehicle id")
    parser.add_argument("--stage1-id", default=DEFAULT_STAGE1_ID, help="stage1 vehicle id")
    parser.add_argument(
        "--template-id",
        default=DEFAULT_TEMPLATE_ID,
        help=f"pipeline template id (use '{AUTO_TEMPLATE_TOKEN}' for auto selection)"
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="total training rounds")
    parser.add_argument("--micro-batches", type=int, default=DEFAULT_MICRO_BATCHES, help="micro-batches per round")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["cifar10", "imagenet", "tinyimagenet", "tiny-imagenet"],
        help="dataset name"
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES, help="number of classes")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="input image size")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="dataset root directory")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES, help="eval batches per round (0 to disable)")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="save checkpoint every N rounds (0 to disable)")
    parser.add_argument("--save-dir", default=DEFAULT_CHECKPOINT_DIR, help="checkpoint output directory")
    parser.add_argument(
        "--tiny-imagenet-url",
        default=DEFAULT_TINY_IMAGENET_URL,
        help="Tiny ImageNet mirror URL (zip/tar.gz)"
    )
    parser.add_argument(
        "--tiny-imagenet-dir",
        default=DEFAULT_TINY_IMAGENET_DIR,
        help="Tiny ImageNet folder name under --data-dir"
    )
    parser.add_argument("--auto-exit", action="store_true", help="exit after completing all rounds")
    parser.add_argument("--listen-host", default="0.0.0.0", help="vehicle listen host for peer pipeline data")
    parser.add_argument("--listen-port", type=int, default=0, help="vehicle listen port for peer pipeline data")
    parser.add_argument("--advertise-host", default=None, help="host/IP to advertise to peers (default: auto-detect)")

    args = parser.parse_args()
    dataset_key = (args.dataset or "").lower().replace("-", "")
    if dataset_key == "imagenet" and args.num_classes == DEFAULT_NUM_CLASSES:
        args.num_classes = 1000
    if dataset_key == "tinyimagenet":
        if args.num_classes == DEFAULT_NUM_CLASSES:
            args.num_classes = 200
        if args.image_size == DEFAULT_IMAGE_SIZE:
            args.image_size = 224
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
            args.micro_batches,
            args.template_id
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
            download=args.download,
            dataset=args.dataset,
            image_size=args.image_size,
            num_classes=args.num_classes,
            tiny_imagenet_url=args.tiny_imagenet_url,
            tiny_imagenet_dir=args.tiny_imagenet_dir,
            eval_batches=args.eval_batches,
            save_every=args.save_every,
            save_dir=args.save_dir
        )
        vehicle.start()
        try:
            while not shutdown_requested:
                time.sleep(5.0)
        finally:
            vehicle.stop()


if __name__ == "__main__":
    main()
