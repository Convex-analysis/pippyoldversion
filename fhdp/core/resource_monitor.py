"""
Resource monitoring utilities for FHDP pipeline training.

Provides memory and compute resource monitoring for vehicles and edge devices.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Any, Optional

import torch


class ResourceMonitor:
    """Monitor system resources including CPU memory and CUDA GPU memory"""

    def __init__(self):
        self._timing_start = time.perf_counter()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics (RSS, CUDA allocated/reserved)"""
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

    def log_resource(self, event: str, stats: Optional[Dict[str, float]] = None) -> None:
        """Log resource usage for an event"""
        if stats is None:
            stats = self.get_memory_stats()

        parts = [f"[RESOURCE] {event}"]
        for key, value in stats.items():
            parts.append(f"{key}={value:.2f}MB")
        print(" ".join(parts))

    def log_timing(self, event: str, invite_start: Optional[float] = None) -> None:
        """Log timing information for an event"""
        now = time.perf_counter()
        elapsed_ms = (now - self._timing_start) * 1000.0
        parts = [f"[TIMING] {event} elapsed_ms={elapsed_ms:.2f}"]

        if invite_start is not None:
            invite_elapsed_ms = (now - invite_start) * 1000.0
            parts.append(f"invite_elapsed_ms={invite_elapsed_ms:.2f}")

        print(" ".join(parts))

    def reset_timing(self) -> None:
        """Reset timing start to current time"""
        self._timing_start = time.perf_counter()

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds since last reset"""
        return (time.perf_counter() - self._timing_start) * 1000.0


class RoundMetrics:
    """Collect and track metrics for a training round"""

    def __init__(self):
        self.activation_bytes_sent = 0
        self.activation_bytes_recv = 0
        self.grad_bytes_sent = 0
        self.grad_bytes_recv = 0
        self.activation_send_ts: Dict[int, float] = {}
        self.grad_latency_ms: list = []
        self.stage1_decode_ms: list = []
        self.stage1_compute_ms: list = []
        self.stage1_send_ms: list = []
        self.stage1_e2e_ms: list = []
        self.stage0_apply_ms: list = []
        self.stage0_step_ms: list = []

    def reset(self) -> None:
        """Reset all metrics"""
        self.activation_bytes_sent = 0
        self.activation_bytes_recv = 0
        self.grad_bytes_sent = 0
        self.grad_bytes_recv = 0
        self.activation_send_ts.clear()
        self.grad_latency_ms.clear()
        self.stage1_decode_ms.clear()
        self.stage1_compute_ms.clear()
        self.stage1_send_ms.clear()
        self.stage1_e2e_ms.clear()
        self.stage0_apply_ms.clear()
        self.stage0_step_ms.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        def avg(lst: list) -> float:
            return sum(lst) / max(1, len(lst))

        return {
            "activation_bytes_sent": self.activation_bytes_sent,
            "activation_bytes_recv": self.activation_bytes_recv,
            "grad_bytes_sent": self.grad_bytes_sent,
            "grad_bytes_recv": self.grad_bytes_recv,
            "avg_grad_latency_ms": avg(self.grad_latency_ms),
            "avg_decode_ms": avg(self.stage1_decode_ms),
            "avg_compute_ms": avg(self.stage1_compute_ms),
            "avg_send_ms": avg(self.stage1_send_ms),
            "avg_e2e_ms": avg(self.stage1_e2e_ms),
            "avg_apply_ms": avg(self.stage0_apply_ms),
            "avg_step_ms": avg(self.stage0_step_ms),
        }


def estimate_payload_bytes(value) -> int:
    """Estimate byte size of a payload (tensor, bytes, dict, list, etc.)"""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    if isinstance(value, dict):
        return sum(estimate_payload_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_payload_bytes(v) for v in value)

    return 0