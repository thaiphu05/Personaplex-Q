import unittest

from personaplex_finetuning.train_utils import (
    role_pass_is_synced,
    sample_position_for_step,
    write_val_scalars,
)


class RecordingWriter:
    """Minimal TensorBoard stand-in capturing ``add_scalar`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, int]] = []

    def add_scalar(self, name: str, value: object, step: int) -> None:
        self.calls.append((name, value, step))


class ValScalarWriterTest(unittest.TestCase):
    def test_writes_at_optimizer_step(self) -> None:
        """Regression test for the eval bug: scalars were written at an
        undefined ``step + 1`` name, raising NameError on the first eval."""
        writer = RecordingWriter()
        metrics = {"val/loss_total": 1.25, "val/loss_text": 0.5}

        write_val_scalars(writer, metrics, optimizer_step=500)

        self.assertEqual(
            [call[2] for call in writer.calls],
            [500, 500],
            "validation metrics must use the optimizer step, not a micro-step",
        )

    def test_none_writer_is_safe(self) -> None:
        write_val_scalars(None, {"val/loss_total": 0.0}, optimizer_step=1)


class SamplePositionTest(unittest.TestCase):
    def test_positions_are_rank_disjoint(self) -> None:
        seen = set()
        for rank in range(4):
            position = sample_position_for_step(micro_step=3, process_index=rank, num_processes=4)
            self.assertNotIn(position, seen)
            seen.add(position)

    def test_positions_non_decreasing_across_blocks(self) -> None:
        positions = [sample_position_for_step(m, process_index=0, num_processes=2) for m in range(5)]
        self.assertEqual(positions, [0, 0, 2, 2, 4])

    def test_role_pass_is_synced_across_ranks(self) -> None:
        """Every rank must sample from the same speaker-role pass, otherwise
        DDP averages gradients from the LEFT and RIGHT role views together."""
        for micro_step in range(6):
            self.assertTrue(
                role_pass_is_synced(micro_step, num_processes=4, chunk_count=10),
                f"ranks diverged across role passes at micro_step {micro_step}",
            )

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            sample_position_for_step(-1, process_index=0, num_processes=1)
        with self.assertRaises(ValueError):
            sample_position_for_step(0, process_index=2, num_processes=2)
        with self.assertRaises(ValueError):
            role_pass_is_synced(0, num_processes=2, chunk_count=0)


if __name__ == "__main__":
    unittest.main()
