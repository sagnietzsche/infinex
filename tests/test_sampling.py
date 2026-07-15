"""Phase 1 tests: sampling primitives."""

import numpy as np
import pytest

from llmserve import sampling


class TestSoftmax:
    def test_shift_invariance(self):
        """softmax(x) == softmax(x + c)"""
        raise NotImplementedError

    def test_no_overflow_on_large_logits(self):
        """[1000, 1001] must not produce nan/inf."""
        raise NotImplementedError

    def test_sums_to_one(self):
        raise NotImplementedError


class TestTemperature:
    def test_low_temperature_approaches_argmax(self):
        raise NotImplementedError

    def test_high_temperature_approaches_uniform(self):
        raise NotImplementedError

    def test_zero_temperature_is_greedy(self, rng):
        """sample(T=0) returns argmax deterministically."""
        raise NotImplementedError


class TestTopK:
    def test_keeps_k_largest(self):
        raise NotImplementedError

    def test_k_larger_than_vocab_is_noop(self):
        raise NotImplementedError


class TestTopP:
    def test_threshold_crossing_token_included(self):
        """The off-by-one: the token that crosses p is IN the nucleus."""
        raise NotImplementedError

    def test_p_equal_one_is_noop(self):
        raise NotImplementedError

    def test_renormalizes(self):
        raise NotImplementedError


class TestSample:
    def test_reproducible_with_same_rng_seed(self):
        raise NotImplementedError

    def test_all_equal_logits(self, rng):
        raise NotImplementedError
