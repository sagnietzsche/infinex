"""Phase 3: tiny transformer with prefill/decode forward modes.

Random-initialized weights are fine -- this is an inference engine, not
a training project. Keep dimensions small (d_model 64-128, 2-4 layers,
2-4 heads, vocab 256-1024) so everything runs on CPU in seconds.

The key correctness test: full no-cache forward over a sequence must
produce the same logits (within float tolerance) as prefill + N decode
steps. That test is reused in Phase 4 against the paged cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kv_cache import ContiguousKVCache


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 512
    seed: int = 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class TinyTransformer:
    """Decoder-only transformer: embedding + positional encoding, then
    per layer: LayerNorm -> causal MHA -> residual -> LayerNorm -> MLP
    -> residual; final LayerNorm and tied/untied LM head.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Random-init all weights from a seeded RNG (config.seed)."""
        raise NotImplementedError

    # --- internals -------------------------------------------------------

    def _attention(
        self,
        layer: int,
        x: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        causal: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Multi-head attention for `x` [n_new, d_model] attending over
        cached K/V plus its own new positions.

        Returns (output [n_new, d_model], k_new, v_new) where k_new/v_new
        are [n_new, n_heads, head_dim] for the caller to append to cache.
        """
        raise NotImplementedError

    def _mlp(self, layer: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- forward modes ----------------------------------------------------

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Cache-free full forward with causal mask.

        Returns logits for every position: [seq_len, vocab_size].
        Used as ground truth in tests.
        """
        raise NotImplementedError

    def prefill(self, token_ids: list[int], cache: ContiguousKVCache) -> np.ndarray:
        """Process the whole prompt at once, filling `cache`.

        Returns logits for the last position: [vocab_size].
        """
        raise NotImplementedError

    def decode(self, token_id: int, cache: ContiguousKVCache) -> np.ndarray:
        """Process one token, attending over cache + the new position,
        appending to `cache`. Returns logits: [vocab_size]."""
        raise NotImplementedError
