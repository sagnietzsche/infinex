"""Phase 2 tests: tokenizer round-trip."""

from llmserve.tokenizer import Tokenizer


def test_round_trip():
    """decode(encode(s)) == s for text within the vocab."""
    raise NotImplementedError


def test_unknown_chars_map_to_unk():
    raise NotImplementedError


def test_special_tokens_skipped_in_decode():
    raise NotImplementedError
