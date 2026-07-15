"""Phase 3 tests: the correctness core.

The single most important test in the repo:
cache-free full forward == prefill + step-by-step decode.
"""

import numpy as np

from llmserve.kv_cache import ContiguousKVCache
from llmserve.model import TinyTransformer


def test_prefill_decode_matches_full_forward(model, rng):
    """Run the model with no cache over the full sequence, and with
    prefill + N decode steps; assert last-position logits match at every
    step within float tolerance."""
    raise NotImplementedError


def test_cache_capacity_enforced(model, config):
    """Appending past max_seq_len raises."""
    raise NotImplementedError


def test_forward_is_deterministic(model):
    raise NotImplementedError
