"""
Reusable pipeline runtime utilities for FHDP.

Includes:
- Sequence id construction
- Sequence handler registration helpers
- 1F1B micro-batch phase utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Sequence, Union


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
