"""Shared fixtures: fixed seeds and tiny model dimensions everywhere."""

import numpy as np
import pytest

from llmserve.model import ModelConfig, TinyTransformer


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, d_ff=128,
        max_seq_len=128, seed=0,
    )


@pytest.fixture
def model(config: ModelConfig) -> TinyTransformer:
    return TinyTransformer(config)
