"""Phase 4 tests: paged cache correctness and allocator behavior."""

import numpy as np
import pytest

from llmserve.paged_kv import BlockPool, OutOfBlocksError, paged_attention


def test_gather_matches_appended_kv():
    """Round-trip: append K/V then gather; must equal the source exactly."""
    raise NotImplementedError


def test_paged_attention_equals_dense_attention(model, rng):
    """Attention through the block table == contiguous attention, exactly."""
    raise NotImplementedError


def test_allocation_exhaustion_raises():
    """Requesting more blocks than exist raises OutOfBlocksError,
    leaving the pool unchanged."""
    raise NotImplementedError


def test_freed_blocks_are_reused():
    """free() returns blocks to the pool; a subsequent allocate() reuses
    those exact block ids."""
    raise NotImplementedError


def test_block_boundary_append():
    """Appending across a block boundary grabs a new block at exactly
    block_size tokens."""
    raise NotImplementedError


def test_usage_accounting():
    """stats() reflects blocks used/free and fragmentation."""
    raise NotImplementedError
