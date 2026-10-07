"""Qwen tokenizer adapter implementing the Tokenizer protocol used by ``sequence.py``.

The Qwen vocabulary already covers Vietnamese, so words from ``words.json`` are
encoded as-is; no diacritic stripping or Telex conversion is required.
"""

from __future__ import annotations


class QwenTokenizer:
    """Thin wrapper exposing the ids expected by the training sequence builder."""

    def __init__(self, tokenizer, end_padding_id: int | None = None) -> None:
        self._tokenizer = tokenizer
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.padding_id = int(pad)
        self.end_padding_id = int(end_padding_id) if end_padding_id is not None else int(tokenizer.eos_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))
