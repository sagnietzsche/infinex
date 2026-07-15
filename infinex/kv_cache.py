"""Phase 3 (cache half): contiguous per-sequence KV cache (v1).

One cache per layer: K and V tensors of shape
[max_seq_len, n_heads, head_dim] plus a length counter.
This is the baseline the paged cache (Phase 4) must match exactly.
"""

from __future__ import annotations

import numpy as np


class ContiguousKVCache:
    """Contiguous KV storage for a single sequence, all layers."""

    def __init__(
        self,
        n_layers: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of cached positions (same across layers)."""
        raise NotImplementedError

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append K/V for new positions at `layer`.

        k, v: [n_new, n_heads, head_dim]. Raises if capacity exceeded.
        The length counter only advances once all layers have appended
        (caller advances it via `advance`).
        """
        raise NotImplementedError

    def advance(self, n_new: int) -> None:
        """Advance the position counter after all layers appended."""
        raise NotImplementedError

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (K, V) views over the filled prefix:
        each [seq_len, n_heads, head_dim]."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
