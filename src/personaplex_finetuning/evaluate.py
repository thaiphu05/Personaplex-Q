"""Evaluate a trained checkpoint on unseen samples from any prepared manifest.

For every sample: free-running generation (``agent.wav`` / ``agent.txt``, the
same path as ``inference.generate``), the original dialogue for comparison,
and teacher-forced accuracies (``metrics.json``) with the target text
(``ref_agent.txt``). ``summary.json`` collects per-sample results and means.

python -m personaplex_finetuning.evaluate --config configs/qwen3-0.6b.yaml \
    --checkpoint runs/<run> [--manifest <prepared>/val.jsonl] [--limit 10]

``train.run`` calls ``evaluate_samples`` on the run's validation samples when
``train.final_eval_samples`` > 0, writing to ``<run>/eval``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from .config import Config
from .data import PreparedSample

logger = logging.getLogger(__name__)

AGENT_CODEBOOKS = 8
METRIC_KEYS = ("agent_cb0_acc", "agent_all_cb_acc", "text_word_acc")


def teacher_forced_summary(output, example, filler_ids, torch) -> dict:
    """Agent predictions and accuracies over the dialogue part of one example.

    Returns ``audio`` ([8, T_dialogue] codes, targets filled in where the delay
    leaves no prediction), ``text_ids`` / ``ref_ids`` (non-filler text tokens)
    and ``metrics`` (CB0, all-codebook and word accuracies).
    """
    labels = torch.tensor(example.labels, dtype=torch.long, device=output.logits.device)
    loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=labels.device)
    start = int(example.prompt_frames)
    filler = torch.tensor(sorted(set(int(token) for token in filler_ids)), device=labels.device)

    logits = output.logits[0, :AGENT_CODEBOOKS].float().nan_to_num()
    mask = output.mask[0, :AGENT_CODEBOOKS]
    targets = labels[1 : 1 + AGENT_CODEBOOKS]
    predicted = logits.argmax(dim=-1)
    audio = torch.where(mask, predicted, targets)[:, start:]

    audio_valid = (mask & loss_mask[1 : 1 + AGENT_CODEBOOKS])[:, start:]
    hits = (predicted[:, start:] == targets[:, start:]) & audio_valid

    text_logits = output.text_logits[0, 0].float().nan_to_num()
    text_pred = text_logits.argmax(dim=-1)[start:]
    text_target = labels[0, start:]
    text_valid = (output.text_mask.reshape(-1) & loss_mask[0])[start:]
    is_filler = torch.isin(text_target, filler)
    word = text_valid & ~is_filler

    def ratio(numerator, denominator) -> float | None:
        count = int(denominator.sum())
        return float(numerator.sum()) / count if count else None

    text_ids = text_pred[text_valid & ~torch.isin(text_pred, filler)].tolist()
    ref_ids = text_target[word].tolist()
    return {
        "audio": audio,
        "text_ids": text_ids,
        "ref_ids": ref_ids,
        "metrics": {
            "dialogue_frames": int(audio.shape[-1]),
            "agent_cb0_acc": ratio(hits[0], audio_valid[0]),
            "agent_all_cb_acc": ratio(hits, audio_valid),
            "text_word_acc": ratio((text_pred == text_target) & word, word),
        },
    }


def teacher_forced(config: Config, runtime, model, sample: PreparedSample) -> dict:
    """Training forward on ``sample`` and its ``teacher_forced_summary``."""
    import torch

    from .inference import inference_autocast
    from .train import build_example

    example = build_example(config, sample, runtime)
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    with torch.no_grad(), inference_autocast(config):
        output = model.forward_train(codes)
    filler = (0, runtime.tokenizer.padding_id, runtime.tokenizer.end_padding_id)  # same filter as inference
    return teacher_forced_summary(output, example, filler, torch)


def resolve_checkpoint(path: Path) -> Path:
    """Adapter file from a ``lora.safetensors``, a checkpoint dir, or a run dir.

    A run dir resolves to ``checkpoints/best`` when present, else the
    ``checkpoints/checkpoint_<step>`` with the highest step.
    """
    from .inference import _adapter_file

    path = Path(path)
    if path.is_file() or (path / "lora.safetensors").is_file():
        return _adapter_file(path)
    checkpoints = path / "checkpoints"
    if (checkpoints / "best" / "lora.safetensors").is_file():
        return checkpoints / "best" / "lora.safetensors"
    stepped = []
    for candidate in checkpoints.glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", candidate.name)
        if match and (candidate / "lora.safetensors").is_file():
            stepped.append((int(match.group(1)), candidate))
    if not stepped:
        raise FileNotFoundError(f"no checkpoint found in {path} (expected lora.safetensors, a checkpoint dir, or a run dir)")
    return max(stepped)[1] / "lora.safetensors"


def evaluate_samples(config: Config, runtime, model, samples: list[PreparedSample], output_dir: Path,
                     gen_cfg: dict | None = None, checkpoint: str | None = None,
                     manifest: str | None = None) -> dict:
    """Generate and score every sample with an already loaded model; write ``summary.json``.

    A failing sample is recorded with its error and does not stop the others.
    """
    from .inference import _export_context, decode_text, export_stereo, generate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sample in samples:
        sample_dir = output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            _export_context(sample, sample_dir)
            generate(
                config, sample, sample_dir / "agent.wav", sample_dir / "agent.txt",
                adapter=None, runtime=runtime, model=model, gen_cfg=gen_cfg,
            )
            export_stereo(sample_dir)
            summary = teacher_forced(config, runtime, model, sample)
            (sample_dir / "ref_agent.txt").write_text(decode_text(config, runtime, summary["ref_ids"]), encoding="utf-8")
            (sample_dir / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2))
            results.append({"sample_id": sample.sample_id, "metrics": summary["metrics"], "error": None})
        except Exception as exc:  # one bad sample must not abort the evaluation
            logger.warning("evaluation failed for %s: %s", sample.sample_id, exc)
            results.append({"sample_id": sample.sample_id, "metrics": None, "error": str(exc)})

    scored = [result["metrics"] for result in results if result["metrics"] is not None]
    mean = {}
    for key in METRIC_KEYS:
        values = [metrics[key] for metrics in scored if metrics.get(key) is not None]
        mean[key] = sum(values) / len(values) if values else None
    report = {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "count": len(results),
        "failed": sum(result["error"] is not None for result in results),
        "mean": mean,
        "samples": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def select_samples(samples: list[PreparedSample], sample_ids: list[str] | None, limit: int | None) -> list[PreparedSample]:
    """Keep the requested ``sample_ids`` (in that order), then cut to ``limit``."""
    if sample_ids:
        by_id = {sample.sample_id: sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise KeyError(f"sample_id not in manifest: {missing[:5]}")
        samples = [by_id[sample_id] for sample_id in sample_ids]
    return samples[:limit] if limit else samples


def default_manifest(config: Config) -> Path:
    """``data.eval_manifest``, else the prepared ``val.jsonl``; never the training manifest."""
    manifest = config.eval_manifest_path or config.val_manifest
    if manifest is None:
        raise FileNotFoundError(
            "no evaluation manifest: pass --manifest, set data.eval_manifest, "
            f"or provide {config.prepared_dir / 'val.jsonl'}"
        )
    return Path(manifest)


def main() -> int:
    from .config import load_config
    from .data import PreparedDataset
    from .inference import attach_adapter, load_inference_runtime

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Run dir, checkpoint dir, or lora.safetensors")
    parser.add_argument("--manifest", default=None, help="Prepared .jsonl (default: data.eval_manifest, else val.jsonl)")
    parser.add_argument("--limit", type=int, default=10, help="Max samples (0 = all)")
    parser.add_argument("--sample-id", action="append", default=None, help="Evaluate only these sample ids (repeatable)")
    parser.add_argument("--output-dir", default=None, help="Default: <checkpoint dir>/eval_<manifest name>")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    adapter = resolve_checkpoint(Path(args.checkpoint)).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else default_manifest(config)
    samples = select_samples(
        PreparedDataset(manifest, config.window_seconds).load(), args.sample_id, args.limit or None,
    )
    output_dir = Path(args.output_dir) if args.output_dir else adapter.parent / f"eval_{manifest.stem}"
    print(f"Evaluating {adapter} on {len(samples)} sample(s) from {manifest} -> {output_dir}")

    runtime = load_inference_runtime(config)
    model = runtime.model
    attach_adapter(config, model, adapter)
    model.eval()
    report = evaluate_samples(
        config, runtime, model, samples, output_dir, gen_cfg=config.generation,
        checkpoint=str(adapter), manifest=str(manifest),
    )
    print(json.dumps({key: report[key] for key in ("count", "failed", "mean")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
