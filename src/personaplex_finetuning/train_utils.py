"""Pure helpers shared by the training loop, kept import-light for unit tests.

These functions were extracted from ``train.py`` so the guardrails around
step bookkeeping and rank-disjoint sampling can be exercised without
importing Torch or Accelerate.
"""

from __future__ import annotations


def write_val_scalars(writer, metrics: dict[str, float], optimizer_step: int) -> None:
    """Write validation metrics to TensorBoard at the optimizer step.

    The writer must always receive the *optimizer* step; using an undefined
    or micro-step counter here silently corrupts the TensorBoard x-axis or
    raises ``NameError`` at the first evaluation.
    """
    for name, value in metrics.items():
        if writer is not None:
            writer.add_scalar(name, value, optimizer_step)


def sample_position_for_step(micro_step: int, process_index: int, num_processes: int) -> int:
    """Return a rank-disjoint sample position that keeps every rank inside
    the same dataset pass for a given micro-step.

    ``(micro_step // num_processes) * num_processes + process_index`` assigns
    consecutive positions to consecutive ranks within one block. All ranks
    therefore share the same ``position // chunk_count`` (i.e. the same role
    pass) so DDP never averages gradients across swapped speaker roles.
    """
    if micro_step < 0 or num_processes < 1 or not 0 <= process_index < num_processes:
        raise ValueError("invalid distributed sample selection inputs")
    return (micro_step // num_processes) * num_processes + process_index


def role_pass_is_synced(micro_step: int, num_processes: int, chunk_count: int) -> bool:
    """True when every rank lands in the same role pass for a micro-step."""
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")
    passes = {
        sample_position_for_step(micro_step, rank, num_processes) // chunk_count
        for rank in range(num_processes)
    }
    return len(passes) == 1
