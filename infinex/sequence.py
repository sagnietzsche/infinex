"""Phase 5: sequence lifecycle + single-sequence and static batching loops.

Static batching's inefficiency (finished sequences sit idle until the
batch drains) is exactly what continuous batching in Phase 6 fixes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from .model import TinyTransformer


class SequenceState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(enum.Enum):
    STOP = "stop"      # hit EOS
    LENGTH = "length"  # hit max_tokens
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0       # 0 = disabled
    top_p: float = 1.0   # 1.0 = disabled
    max_tokens: int = 64
    seed: int | None = None


@dataclass
class Sequence:
    """One request's full generation state."""

    request_id: int
    prompt_ids: list[int]
    params: SamplingParams
    generated_ids: list[int] = field(default_factory=list)
    state: SequenceState = SequenceState.WAITING
    finish_reason: FinishReason | None = None
    # Set on admission (Phase 4/6); block table lives in the BlockPool,
    # keyed by request_id.
    rng: np.random.Generator | None = None

    def __len__(self) -> int:
        """Total tokens: prompt + generated."""
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError

    def check_stop(self, eos_id: int) -> bool:
        """Update state/finish_reason if EOS emitted or max_tokens hit.
        Returns True if the sequence just finished."""
        raise NotImplementedError


def generate(
    model: TinyTransformer,
    seq: Sequence,
    eos_id: int,
) -> list[int]:
    """Single-sequence loop: prefill, then decode until stop.
    Returns generated ids."""
    raise NotImplementedError


def generate_static_batch(
    model: TinyTransformer,
    seqs: list[Sequence],
    eos_id: int,
) -> list[list[int]]:
    """Static batching: prefill each sequence, then synchronized decode
    steps computing one token per live sequence per step. Finished
    sequences idle until the whole batch drains."""
    raise NotImplementedError
