"""Phase 6: continuous batching engine (Orca-style iteration-level scheduling).

Each engine step decides which sequences run:
  - admit new prefills while the paged allocator has capacity
    (prompt blocks + headroom),
  - run one decode step for every running sequence,
  - sequences join and leave the batch at any step.

Preemption: on OutOfBlocksError mid-generation, evict the lowest-priority
running sequence, free its blocks, requeue it. Recompute-on-resume is the
policy here (swap-out is the documented alternative).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .model import TinyTransformer
from .paged_kv import BlockPool
from .sequence import FinishReason, Sequence


@dataclass
class StepOutput:
    """What one engine step produced, routed to per-request queues in Phase 7."""

    # request_id -> newly emitted token id (one per running sequence)
    new_tokens: dict[int, int] = field(default_factory=dict)
    # request_id -> finish reason, for sequences that ended this step
    finished: dict[int, FinishReason] = field(default_factory=dict)
    # request_ids preempted (freed + requeued) this step
    preempted: list[int] = field(default_factory=list)


class Engine:
    """The serving engine: waiting queue + running set + paged allocator."""

    def __init__(
        self,
        model: TinyTransformer,
        pool: BlockPool,
        eos_id: int,
        max_running: int = 64,
        headroom_blocks: int = 4,
    ) -> None:
        self.model = model
        self.pool = pool
        self.eos_id = eos_id
        self.max_running = max_running
        self.headroom_blocks = headroom_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        """Enqueue a new request (FIFO)."""
        raise NotImplementedError

    def _can_admit(self, seq: Sequence) -> bool:
        """Admission control: enough free blocks for the prompt plus
        headroom_blocks."""
        raise NotImplementedError

    def _admit(self, seq: Sequence) -> None:
        """Allocate blocks, prefill, move WAITING -> RUNNING."""
        raise NotImplementedError

    def _preempt_one(self) -> int:
        """Evict the lowest-priority running sequence: free its blocks,
        reset generation progress (recompute-on-resume), requeue.
        Returns the preempted request_id."""
        raise NotImplementedError

    def step(self) -> StepOutput:
        """One scheduling iteration:
        1. admit waiting prefills while capacity allows,
        2. one decode step for every running sequence (preempting on
           OutOfBlocksError),
        3. retire finished sequences and free their blocks.
        """
        raise NotImplementedError

    def has_work(self) -> bool:
        raise NotImplementedError
