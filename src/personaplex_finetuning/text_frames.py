"""Decode frame-aligned model text for inference, evaluation and benchmark."""

from __future__ import annotations

from typing import Sequence

WORD_MARKERS = ("▁", "Ġ")


def decode_text_tokens(tokenizer, token_ids) -> str:
    """Decode generated text IDs with the tokenizer that produced the training targets.

    Handles the PersonaPlex SentencePiece wrapper, the Qwen wrapper and a raw
    Hugging Face tokenizer.
    """
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids))
    hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if hf_tokenizer is not None and hasattr(hf_tokenizer, "decode"):
        return str(hf_tokenizer.decode(token_ids))
    processor = getattr(tokenizer, "_processor", None)
    if processor is not None and hasattr(processor, "decode_ids"):
        return str(processor.decode_ids(token_ids))
    if processor is not None:
        return "".join(processor.id_to_piece(token) for token in token_ids).strip()
    return "".join(str(token) for token in token_ids)


def token_piece(tokenizer, token: int) -> str:
    """Single token surface form without decoding the whole run."""
    if hasattr(tokenizer, "id_to_piece"):
        return str(tokenizer.id_to_piece(token))
    hf_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if hf_tokenizer is not None and hasattr(hf_tokenizer, "convert_ids_to_tokens"):
        return str(hf_tokenizer.convert_ids_to_tokens(int(token)))
    processor = getattr(tokenizer, "_processor", None)
    if processor is not None:
        return str(processor.id_to_piece(token))
    return ""


def _starts_word(piece: str) -> bool:
    return piece.startswith(WORD_MARKERS)


def text_frames_to_words(
    text_frames: Sequence[tuple[int, int]],
    tokenizer,
    frame_sec: float,
    start_sec: float = 0.0,
) -> list[tuple[float, float, str]]:
    """Group the per-frame text stream into timed words.

    A word is a run of consecutive non-padding tokens, split again where a piece
    starts with a SentencePiece/Qwen word marker. Its text is decoded by the
    run's own tokenizer.
    """
    ignored = {0, int(tokenizer.padding_id), int(getattr(tokenizer, "end_padding_id", 0))}
    words: list[tuple[float, float, str]] = []
    run: list[tuple[int, int]] = []

    def flush() -> None:
        if run:
            text = decode_text_tokens(tokenizer, [token for _, token in run]).strip()
            if text:
                words.append(
                    (
                        start_sec + run[0][0] * frame_sec,
                        start_sec + (run[-1][0] + 1) * frame_sec,
                        text,
                    )
                )
            run.clear()

    previous_frame = None
    for frame, token in text_frames:
        if token in ignored or (previous_frame is not None and frame != previous_frame + 1):
            flush()
        if token not in ignored:
            if run and _starts_word(token_piece(tokenizer, token)):
                flush()
            run.append((frame, token))
        previous_frame = frame
    flush()
    return words
