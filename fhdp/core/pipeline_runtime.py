"""
Reusable pipeline runtime utilities for FHDP.

Includes:
- Sequence id construction
- Sequence handler registration helpers
- 1F1B micro-batch phase utilities
- EMA latency tracking and dynamic micro-batch adjustment
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Sequence, Tuple, Union


class MicroBatchPhase(str, Enum):
    WARMUP = "warmup"
    STEADY = "steady"
    COOLDOWN = "cooldown"


def get_micro_batch_phase(micro_idx: int, total_micro_batches: int) -> MicroBatchPhase:
    """Return 1F1B phase for a micro-batch index."""
    total = max(1, int(total_micro_batches))
    idx = int(micro_idx)
    if idx == 0:
        return MicroBatchPhase.WARMUP
    if idx == total - 1:
        return MicroBatchPhase.COOLDOWN
    return MicroBatchPhase.STEADY


@dataclass(frozen=True)
class OneFOneBSchedule:
    """Simple 1F1B micro-batch phase scheduler."""
    micro_batches: int

    def __post_init__(self):
        if int(self.micro_batches) < 1:
            raise ValueError("micro_batches must be >= 1")

    def phase(self, micro_idx: int) -> MicroBatchPhase:
        return get_micro_batch_phase(micro_idx, self.micro_batches)

    def iter_micro_batches(self) -> Iterable[int]:
        for idx in range(int(self.micro_batches)):
            yield idx


@dataclass(frozen=True)
class SequenceIdFactory:
    """Factory for pipeline sequence ids."""
    pipeline_id: str
    delimiter: str = "|"

    def make(self, round_num: int, kind: str, micro_idx: int) -> str:
        return f"{self.pipeline_id}{self.delimiter}r{int(round_num)}{self.delimiter}{kind}{self.delimiter}m{int(micro_idx)}"


class EMALatencyTracker:
    """EMA latency tracker for thermal adaptation"""
    
    def __init__(self, alpha: float = 0.1, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self.ema_latency = 0.0
        self.wait_times = []
    
    def update(self, wait_time: float) -> float:
        """Update EMA latency with new wait time"""
        self.wait_times.append(wait_time)
        if len(self.wait_times) > 10:
            self.wait_times.pop(0)
        
        # Update EMA
        if self.ema_latency == 0:
            self.ema_latency = wait_time
        else:
            self.ema_latency = self.alpha * wait_time + (1 - self.alpha) * self.ema_latency
        
        return self.ema_latency
    
    def should_adjust(self) -> bool:
        """Check if micro-batch size should be adjusted"""
        return self.ema_latency > self.threshold


class MicroBatchAdjuster:
    """Dynamic micro-batch size adjuster based on thermal throttling"""
    
    def __init__(self, initial_micro_batches: int, min_micro_batches: int = 1):
        self.initial_micro_batches = initial_micro_batches
        self.min_micro_batches = min_micro_batches
        self.current_micro_batches = initial_micro_batches
        self.latency_tracker = EMALatencyTracker()
    
    def update_wait_time(self, wait_time: float) -> int:
        """Update wait time and adjust micro-batch size if needed"""
        ema_latency = self.latency_tracker.update(wait_time)
        
        if self.latency_tracker.should_adjust() and self.current_micro_batches > self.min_micro_batches:
            self.current_micro_batches = max(self.min_micro_batches, self.current_micro_batches // 2)
        elif not self.latency_tracker.should_adjust() and self.current_micro_batches < self.initial_micro_batches:
            new_size = min(self.initial_micro_batches, int(self.current_micro_batches * 1.25))
            if new_size > self.current_micro_batches:
                self.current_micro_batches = new_size
        
        return self.current_micro_batches
    
    def get_current_micro_batches(self) -> int:
        """Get current micro-batch size"""
        return self.current_micro_batches


class PerformanceCorrector:
    """Physics-guided online performance corrector for dynamic parameter tuning.
    
    Predicts round time and wait ratio based on batch size and micro-batch count.
    Suggests optimal parameters based on observed network wait ratios.
    """
    
    def __init__(
        self,
        batch_bounds: Tuple[int, int] = (4, 64),
        micro_bounds: Tuple[int, int] = (1, 16),
        batch_step: int = 2,
        micro_step: int = 1,
        down_threshold: float = 0.6,
        up_threshold: float = 0.3,
        model_alpha: float = 0.105,
        model_beta: float = 2.35,
        model_gamma: float = 22.8,
        model_beta_wait: float = 1.0
    ):
        self.wait_ratio_ema = None
        self.batch_bounds = batch_bounds
        self.micro_bounds = micro_bounds
        self.batch_step = max(1, int(batch_step))
        self.micro_step = max(1, int(micro_step))
        self.down_threshold = float(down_threshold)
        self.up_threshold = float(up_threshold)
        self.model_alpha = float(model_alpha)
        self.model_beta = float(model_beta)
        self.model_gamma = float(model_gamma)
        self.model_beta_wait = float(model_beta_wait)

    def update_online(self, network_wait_ratio: float) -> float:
        """Update EMA with new network wait ratio observation"""
        ratio = float(network_wait_ratio)
        if ratio > 1.0:
            ratio = ratio / 100.0
        ratio = max(0.0, min(1.0, ratio))
        
        if self.wait_ratio_ema is None:
            self.wait_ratio_ema = ratio
        else:
            self.wait_ratio_ema = 0.8 * self.wait_ratio_ema + 0.2 * ratio
        
        return ratio

    def predict_round_time(self, m: int, n: int) -> float:
        """Predict total round time based on batch size m and micro-batches n"""
        return self.model_alpha * m * n + self.model_beta * n + self.model_gamma

    def predict_wait_ratio(self, m: int, n: int) -> float:
        """Predict network wait ratio based on batch size m and micro-batches n"""
        t_round = max(self.predict_round_time(m, n), 1e-6)
        t_wait = self.model_gamma + self.model_beta_wait * n
        return max(0.0, min(1.0, t_wait / t_round))

    def suggest_params(self, batch_size: int, micro_batches: int) -> Tuple[int, int, str]:
        """Suggest next batch/micro parameters based on current observations"""
        current_batch = int(batch_size)
        current_micro = int(micro_batches)
        min_b, max_b = self.batch_bounds
        min_m, max_m = self.micro_bounds

        candidates = set()
        for dm in (-self.batch_step, 0, self.batch_step):
            for dn in (-self.micro_step, 0, self.micro_step):
                cand_b = max(min_b, min(max_b, current_batch + dm))
                cand_m = max(min_m, min(max_m, current_micro + dn))
                candidates.add((cand_b, cand_m))

        valid = []
        for cand_b, cand_m in candidates:
            t_round = self.predict_round_time(cand_b, cand_m)
            throughput = (cand_b * cand_m) / max(t_round, 1e-6)
            wait_ratio_pred = self.predict_wait_ratio(cand_b, cand_m)
            valid.append((cand_b, cand_m, throughput, wait_ratio_pred))

        ema = self.wait_ratio_ema if self.wait_ratio_ema is not None else 0.0

        if ema > self.down_threshold:
            candidates = [item for item in valid if item[0] <= current_batch and item[1] <= current_micro]
            if not candidates:
                candidates = valid
            best = min(candidates, key=lambda x: (x[3], -x[2]))
        elif ema < self.up_threshold:
            candidates = [item for item in valid if item[0] >= current_batch and item[1] >= current_micro]
            if not candidates:
                candidates = valid
            within = [item for item in candidates if item[3] <= self.down_threshold]
            pool = within or candidates
            best = max(pool, key=lambda x: (x[2], -x[3]))
        else:
            best = (current_batch, current_micro, 0.0, ema)

        next_batch, next_micro = best[0], best[1]
        if next_batch < current_batch or next_micro < current_micro:
            action = "down"
        elif next_batch > current_batch or next_micro > current_micro:
            action = "up"
        else:
            action = "hold"

        return next_batch, next_micro, action


class SequenceHandlerRegistry:
    """Helper to register/unregister pipeline sequence handlers."""

    def __init__(self, pipeline_comm_manager, sequence_factory: SequenceIdFactory):
        self.pipeline_comm_manager = pipeline_comm_manager
        self.sequence_factory = sequence_factory

    @staticmethod
    def _normalize_rounds(rounds: Union[int, Sequence[int]]) -> Iterable[int]:
        if isinstance(rounds, int):
            return range(1, rounds + 1)
        return rounds

    def register_for_rounds(
        self,
        kind: str,
        rounds: Union[int, Sequence[int]],
        micro_batches: int,
        handler: Callable,
    ) -> List[str]:
        seq_ids: List[str] = []
        for round_num in self._normalize_rounds(rounds):
            for micro_idx in range(int(micro_batches)):
                seq_id = self.sequence_factory.make(round_num, kind, micro_idx)
                self.pipeline_comm_manager.register_sequence_handler(seq_id, handler)
                seq_ids.append(seq_id)
        return seq_ids

    def unregister_sequence(self, sequence_id: str) -> None:
        self.pipeline_comm_manager.unregister_sequence_handler(sequence_id)

    def unregister_for_rounds(
        self,
        kind: str,
        rounds: Union[int, Sequence[int]],
        micro_batches: int,
    ) -> List[str]:
        seq_ids: List[str] = []
        for round_num in self._normalize_rounds(rounds):
            for micro_idx in range(int(micro_batches)):
                seq_id = self.sequence_factory.make(round_num, kind, micro_idx)
                self.pipeline_comm_manager.unregister_sequence_handler(seq_id)
                seq_ids.append(seq_id)
        return seq_ids
