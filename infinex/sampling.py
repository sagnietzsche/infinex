"""Phase 1: sampling primitives.

All functions are pure functions over 1-D logits vectors (np.ndarray of
shape [vocab_size]). RNG is always passed in explicitly -- never use
global random state -- so decoding is reproducible per request.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Subtract max(logits) before exponentiating. Must satisfy:
      - softmax(x) == softmax(x + c) for any scalar c
      - no overflow on logits like [1000, 1001]
    """
    raise NotImplementedError


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by 1/T.

    T -> 0 approaches argmax, T -> inf approaches uniform.
    T == 0 is handled by the caller (greedy path in `sample`), so this
    may assume temperature > 0.
    """
    raise NotImplementedError


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits, set the rest to -inf.

    k >= vocab_size (or k <= 0 meaning 'disabled') must be a no-op.
    """
    raise NotImplementedError


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filtering over a probability vector.

    Sort descending, keep the smallest prefix with cumulative sum >= p
    (the token that crosses the threshold IS included), mask the rest,
    renormalize. p >= 1.0 must be a no-op.
    """
    raise NotImplementedError


def sample(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Full sampling pipeline: temperature -> top-k -> softmax -> top-p -> draw.

    temperature == 0 short-circuits to greedy argmax.
    Returns the selected token id.
    """
    raise NotImplementedError
