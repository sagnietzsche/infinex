"""Phase 2: character-level tokenizer.

Not tokenizer research -- just a clean interface for the rest of the
stack. Must satisfy round-trip: decode(encode(s)) == s.
"""

from __future__ import annotations


class Tokenizer:
    """Character-level tokenizer with special tokens.

    Special token ids occupy the front of the vocab:
      PAD=0, BOS=1, EOS=2, UNK=3, then one id per character.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, corpus: str | None = None) -> None:
        """Build vocab from the characters in `corpus` (or a default
        printable-ASCII vocab if None)."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Map text to token ids; unknown characters map to UNK."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Map ids back to text, skipping special tokens."""
        raise NotImplementedError
