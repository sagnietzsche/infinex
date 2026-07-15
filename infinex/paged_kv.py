"""Phase 4: paged KV cache (the vLLM / PagedAttention idea).

A global block pool shared by all sequences, per layer:
  K, V tensors of shape [num_blocks, block_size, n_heads, head_dim]
plus a free list of block ids. Each sequence owns a block table (ordered
list of block ids): logical token position t lives at
  block = table[t // block_size], offset = t % block_size.

Tests that matter:
  - paged attention output == contiguous attention output exactly
  - allocation exhaustion raises OutOfBlocksError cleanly
  - freed blocks return to the pool and get reused
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OutOfBlocksError(Exception):
    """Raised when the pool cannot satisfy an allocation."""


@dataclass
class PoolStats:
    num_blocks: int
    blocks_free: int
    blocks_used: int
    fragmentation: float  # wasted slots in partially-filled tail blocks / total slots used


class BlockPool:
    """Global paged KV storage shared across sequences."""

    def __init__(
        self,
        n_layers: int,
        num_blocks: int,
        block_size: int,
        n_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        raise NotImplementedError

    # --- allocation -------------------------------------------------------

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Reserve enough blocks for `n_tokens` and create the block
        table for `seq_id`. Raises OutOfBlocksError if insufficient."""
        raise NotImplementedError

    def free(self, seq_id: int) -> None:
        """Return all of seq_id's blocks to the free list."""
        raise NotImplementedError

    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def blocks_needed(self, n_tokens: int) -> int:
        raise NotImplementedError

    def stats(self) -> PoolStats:
        raise NotImplementedError

    # --- KV access --------------------------------------------------------

    def append(self, seq_id: int, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write K/V [n_new, n_heads, head_dim] at the sequence's next
        logical positions, grabbing a new block when the last one is
        full. Raises OutOfBlocksError if the pool is exhausted."""
        raise NotImplementedError

    def advance(self, seq_id: int, n_new: int) -> None:
        """Advance seq length after all layers appended."""
        raise NotImplementedError

    def seq_len(self, seq_id: int) -> int:
        raise NotImplementedError

    def gather(self, seq_id: int, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Reassemble contiguous (K, V) [seq_len, n_heads, head_dim] from
        the block table. Mainly for testing against the contiguous path."""
        raise NotImplementedError


def paged_attention(
    q: np.ndarray,
    pool: BlockPool,
    seq_id: int,
    layer: int,
) -> np.ndarray:
    """Attention that reads K/V *through* the block table (loop over
    blocks or index_select) rather than gathering first.

    q: [n_heads, head_dim] for a single decode position.
    Returns [n_heads, head_dim]. Must equal dense attention exactly.
    """
    raise NotImplementedError
