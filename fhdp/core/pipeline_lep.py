"""
Lightweight LEP (activation compression) utilities for FHDP pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ActivationLEPState:
    """Track LEP residual and compression stats for activation payloads."""

    residual: Optional[torch.Tensor] = None
    steps: int = 0
    bytes_sent: int = 0
    bytes_baseline: int = 0

    def apply(
        self,
        activation: torch.Tensor,
        enabled: bool,
        fp16_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        payload = activation.detach()
        info: Dict[str, Any] = {"lep_enabled": False}
        if not enabled:
            return payload, info

        with torch.no_grad():
            residual = self.residual
            if residual is not None and residual.shape != payload.shape:
                residual = None
            if residual is not None:
                payload = payload + residual

            quantized = payload.to(fp16_dtype)
            dequantized = quantized.to(torch.float32)
            self.residual = payload - dequantized
            payload = quantized

            info = {
                "lep_enabled": True,
                "lep_dtype": str(fp16_dtype),
                "lep_error_norm": float(torch.norm(self.residual).item()),
            }

        if isinstance(payload, torch.Tensor):
            numel = payload.numel()
            bytes_per_elem = 2 if payload.dtype == torch.float16 else 4
            self.steps += 1
            self.bytes_sent += numel * bytes_per_elem
            self.bytes_baseline += numel * 4

        return payload, info

    def should_log(self, log_interval: int) -> bool:
        return log_interval > 0 and self.steps > 0 and self.steps % log_interval == 0

    def reduction_ratio(self) -> float:
        if self.bytes_baseline <= 0:
            return 0.0
        return 1.0 - (self.bytes_sent / max(1, self.bytes_baseline))
