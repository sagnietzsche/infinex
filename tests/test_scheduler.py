"""Phase 5/6 tests: batching and continuous scheduling."""

import pytest

from llmserve.scheduler import Engine
from llmserve.sequence import SamplingParams, Sequence, SequenceState, generate, generate_static_batch


def test_single_sequence_generation_stops_on_eos_or_length(model):
    raise NotImplementedError


def test_static_batch_matches_single_sequence_outputs(model):
    """Each sequence in a static batch produces the same tokens it would
    alone (same seeds)."""
    raise NotImplementedError


def test_continuous_batching_admits_mid_flight(model):
    """A request added while others are decoding joins the batch on a
    later step without waiting for the batch to drain."""
    raise NotImplementedError


def test_admission_control_respects_free_blocks(model):
    """A prompt needing more blocks than free stays WAITING."""
    raise NotImplementedError


def test_preemption_frees_and_requeues(model):
    """On pool exhaustion mid-decode, a sequence is evicted, its blocks
    freed, and it eventually finishes after resumption (recompute)."""
    raise NotImplementedError


def test_engine_drains_all_requests(model):
    """step() until has_work() is False; every request finishes with a
    valid finish reason."""
    raise NotImplementedError
