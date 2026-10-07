import unittest
import warnings
from pathlib import Path

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.sequence import (
    PersonaPlexTrainingExampleBuilder,
    count_unplaceable_text_tokens,
)


class DenseCodec:
    frame_rate = 12.5
    codebooks = 8

    def encode_conversation(self, _path, _channel, _start, _end):
        return tuple(tuple(0 for _ in range(8)) for _ in range(8))

    def encode_voice_prompt(self, _path):
        return tuple(tuple(0 for _ in range(2)) for _ in range(8))

    def sine(self, frames):
        return tuple(tuple(0 for _ in range(frames)) for _ in range(8))

    def silence(self, frames):
        return tuple(tuple(0 for _ in range(frames)) for _ in range(8))


class DenseTokenizer:
    """Two tokens per word; enough to overflow short windows."""

    padding_id = 3
    end_padding_id = 0

    def encode(self, text):
        return [len(text), len(text) + 1]


def overflow_sample(window_start: float, window_end: float) -> PreparedSample:
    step = 0.02
    words = tuple(
        Word("agent", f"w{index}", window_start + index * step, window_start + (index + 1) * step)
        for index in range(10)
    )
    return PreparedSample(
        sample_id="overflow",
        conversation_wav=Path("conversation.wav"),
        voice_prompt_wav=Path("voice_prompt.wav"),
        words=words,
        text_prompt="Be helpful.",
        metadata={},
        audio=AudioInfo(24000, 2, 60),
        window_start_sec=window_start,
        window_end_sec=window_end,
    )


class TokenBudgetTest(unittest.TestCase):
    def test_counts_unplaceable_tokens(self) -> None:
        """A 0.4s window has 5 frames, but the 40-token word needs 40 frames."""
        sample = overflow_sample(10.0, 10.4)
        overflow = count_unplaceable_text_tokens(
            sample.words, sample.window_start_sec, sample.window_end_sec,
            DenseCodec.frame_rate, DenseTokenizer(),
        )
        self.assertGreater(overflow, 0)

    def test_zero_overflow_for_sparse_text(self) -> None:
        sample = overflow_sample(10.0, 10.4)
        sample = PreparedSample(
            sample_id=sample.sample_id,
            conversation_wav=sample.conversation_wav,
            voice_prompt_wav=sample.voice_prompt_wav,
            words=(Word("agent", "hi", 10.1, 10.2),),
            text_prompt=sample.text_prompt,
            metadata={},
            audio=sample.audio,
            window_start_sec=10.0,
            window_end_sec=14.0,
        )
        overflow = count_unplaceable_text_tokens(
            sample.words, 10.0, 14.0, DenseCodec.frame_rate, DenseTokenizer()
        )
        self.assertEqual(overflow, 0)

    def test_builder_warns_instead_of_silently_dropping(self) -> None:
        """The dialogue builder used to truncate text tokens silently; the
        truncation must at least surface as a warning."""
        builder = PersonaPlexTrainingExampleBuilder(
            codec=DenseCodec(), tokenizer=DenseTokenizer(),
            initial_tokens=[1] * 17, zero_token=-1,
        )
        with self.assertWarns(RuntimeWarning):
            builder.build(overflow_sample(10.0, 10.4))

    def test_apply_delays_shifts_masks_and_streams(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=DenseCodec(), tokenizer=DenseTokenizer(),
            initial_tokens=[1] * 17, zero_token=-1,
        )
        example = builder.build(overflow_sample(10.0, 14.0))
        delays = [0] * 17
        delays[2] = 1
        delayed = builder.apply_delays(example, delays)
        max_delay = 1
        for stream in delayed.input_codes:
            self.assertEqual(len(stream), example.total_frames + 1 + max_delay)
        # the delayed stream must be masked at its first position
        self.assertFalse(delayed.loss_mask[2][0])
        self.assertFalse(delayed.loss_mask[2][1])

    def test_invalid_delays_rejected(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=DenseCodec(), tokenizer=DenseTokenizer(),
            initial_tokens=[1] * 17, zero_token=-1,
        )
        example = builder.build(overflow_sample(10.0, 14.0))
        with self.assertRaises(ValueError):
            builder.apply_delays(example, [0] * 16)
        with self.assertRaises(ValueError):
            builder.apply_delays(example, [0] * 16 + [-1])


if __name__ == "__main__":
    unittest.main()
