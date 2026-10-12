"""Locate where generated audio goes wrong: codec, text, CB0 or CB1-7.

For each sample of a held-out manifest (default: data.eval_manifest, else
val.jsonl) it writes, next to the original dialogue:

  ref_resynth.wav      Mimi decode of the reference agent codes (codec ceiling)
  tf_agent.wav         teacher-forced argmax codes (perfect context)
  free.wav/.txt        free-running generation
  force_text.wav/.txt  free-running, reference text forced each frame
  force_text_cb0.wav   free-running, reference text + agent CB0 forced;
                       only CB1-7 come from the depth transformer
  metrics.json         teacher-forced accuracy per agent codebook

Listening order ref_resynth -> force_text_cb0 -> force_text -> free shows
which stream breaks the audio. Uses the package code unchanged.

python -m tools.audio_probe --config configs/qwen3-0.6b.yaml --checkpoint runs/<run> --limit 5
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
from pathlib import Path

AGENT_CODEBOOKS = 8
MODES = ("free", "force_text", "force_text_cb0")
MEAN_KEYS = ("text_word_acc", "agent_all_cb_acc") + tuple(f"agent_cb{k}_acc" for k in range(AGENT_CODEBOOKS))


def per_codebook_accuracy(predicted, targets, valid) -> list[float | None]:
    """Accuracy of each codebook row over its valid frames ([K, T] tensors)."""
    accuracies = []
    for k in range(predicted.shape[0]):
        count = int(valid[k].sum())
        hits = int(((predicted[k] == targets[k]) & valid[k]).sum())
        accuracies.append(hits / count if count else None)
    return accuracies


def force_cb0(generator, token: int) -> None:
    """Provide the agent CB0 of the frame LMGen generates on its next ``step``.

    LMGen keeps provided tokens in its ring cache: a provided target replaces
    the sampled one, feeds the depth transformer as CB1's input token and stays
    in the cache as the next frame's input, exactly like forced user audio.
    """
    state = generator._streaming_state
    delay = int(generator.lm_model.delays[1])
    position = (state.offset + delay) % state.cache.shape[2]
    state.cache[:, 1, position] = int(token)
    state.provided[:, 1, position] = True


def mean_metrics(rows: list[dict]) -> dict:
    mean = {}
    for key in MEAN_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        mean[key] = sum(values) / len(values) if values else None
    return mean


def _decode(runtime, codes, torch):
    """Non-streaming Mimi decode of [8, T] codes to a 1-D float array."""
    with torch.no_grad():
        pcm = runtime.codec.mimi.decode(codes.unsqueeze(0))
    return pcm.squeeze().float().cpu().numpy()


def teacher_forced_probe(config, runtime, model, example, torch):
    """Teacher-forced argmax codes, reference codes and per-codebook accuracy."""
    from personaplex_finetuning.evaluate import teacher_forced_summary
    from personaplex_finetuning.inference import inference_autocast

    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    with torch.no_grad(), inference_autocast(config):
        output = model.forward_train(codes)
    filler = (0, runtime.tokenizer.padding_id, runtime.tokenizer.end_padding_id)
    summary = teacher_forced_summary(output, example, filler, torch)

    start = int(example.prompt_frames)
    labels = torch.tensor(example.labels, dtype=torch.long, device=output.logits.device)
    loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=labels.device)
    predicted = output.logits[0, :AGENT_CODEBOOKS].float().nan_to_num().argmax(dim=-1)[:, start:]
    targets = labels[1 : 1 + AGENT_CODEBOOKS, start:]
    valid = (output.mask[0, :AGENT_CODEBOOKS] & loss_mask[1 : 1 + AGENT_CODEBOOKS])[:, start:]
    metrics = dict(summary["metrics"])
    for k, accuracy in enumerate(per_codebook_accuracy(predicted, targets, valid)):
        metrics[f"agent_cb{k}_acc"] = accuracy
    return summary["audio"], targets, metrics


def generate_mode(config, runtime, model, sample, example, mode: str, torch):
    """Free-running generation over the example's dialogue with optional forcing.

    Mirrors ``inference.generate`` (same LMGen setup, alignment step and
    training voice-prompt codes) but feeds the example's own user tokens so
    every mode sees identical input.
    """
    import numpy as np

    from personaplex_finetuning.inference import (
        align_generator_with_training,
        inference_autocast,
        step_training_system_prompts,
    )
    from personaplex_finetuning.text_normalization import encode_system_prompt

    gen = dict(config.generation or {})
    lm_module = importlib.import_module("moshi.models.lm")
    compile_module = importlib.import_module("moshi.utils.compile")
    generator = lm_module.LMGen(
        model, audio_silence_frame_cnt=int(gen.get("audio_silence_frame_cnt", 6)),
        sample_rate=runtime.codec.sample_rate, frame_rate=runtime.codec.frame_rate,
        device=str(config.device), use_sampling=bool(gen.get("use_sampling", True)),
        temp=float(gen.get("temp", 0.8)), temp_text=float(gen.get("temp_text", 0.7)),
        top_k=int(gen.get("top_k", 250)), top_k_text=int(gen.get("top_k_text", 25)),
    )
    generator.zero_text_code = runtime.tokenizer.padding_id
    generator.text_prompt_tokens = encode_system_prompt(runtime.tokenizer, sample.text_prompt, config.text_prompt_template)
    voice_codes = runtime.codec.encode_voice_prompt(sample.voice_prompt_wav)

    streams = example.input_codes
    start = int(example.prompt_frames)
    frames = len(streams[0]) - start
    filler = (0, runtime.tokenizer.padding_id, runtime.tokenizer.end_padding_id)
    pcm, text_ids = [], []
    is_qwen = config.backbone == "qwen"
    torch.manual_seed(config.seed)  # same sampling noise for every mode
    if is_qwen:
        model.start_streaming_caches()
    graph_guard = compile_module.no_cuda_graph() if is_qwen else contextlib.nullcontext()
    try:
        with torch.no_grad(), inference_autocast(config), graph_guard, \
                runtime.codec.mimi.streaming(1), generator.streaming(1):
            align_generator_with_training(generator)
            step_training_system_prompts(generator, voice_codes, config.device)
            for d in range(frames):
                t = start + d
                user = torch.tensor([[[int(streams[9 + k][t])] for k in range(AGENT_CODEBOOKS)]], device=config.device)
                kwargs = {"input_tokens": user}
                if mode in ("force_text", "force_text_cb0"):
                    kwargs["text_token"] = int(streams[0][t])
                if mode == "force_text_cb0":
                    force_cb0(generator, int(streams[1][t]))
                tokens = generator.step(**kwargs)
                if tokens is None:
                    continue
                pcm.append(runtime.codec.mimi.decode(tokens[:, 1:9]).squeeze().detach().float().cpu().numpy())
                token = int(tokens[0, 0, 0])
                if token not in filler:
                    text_ids.append(token)
    finally:
        if is_qwen:
            model.stop_streaming_caches()
    if not pcm:
        raise RuntimeError(f"{mode}: generation produced no frames")
    return np.concatenate(pcm), text_ids


def main() -> int:
    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset
    from personaplex_finetuning.evaluate import default_manifest, resolve_checkpoint, select_samples
    from personaplex_finetuning.inference import _export_context, attach_adapter, decode_text, load_inference_runtime
    from personaplex_finetuning.train import build_example

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Run dir, checkpoint dir, or lora.safetensors")
    parser.add_argument("--manifest", default=None, help="Prepared .jsonl (default: data.eval_manifest, else val.jsonl)")
    parser.add_argument("--limit", type=int, default=5, help="Max samples (0 = all)")
    parser.add_argument("--sample-id", action="append", default=None, help="Probe only these sample ids (repeatable)")
    parser.add_argument("--output-dir", default=None, help="Default: <checkpoint dir>/audio_probe_<manifest name>")
    args = parser.parse_args()

    import numpy as np
    import sphn
    import torch

    config = load_config(args.config)
    adapter = resolve_checkpoint(Path(args.checkpoint)).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else default_manifest(config)
    samples = select_samples(PreparedDataset(manifest, config.window_seconds).load(), args.sample_id, args.limit or None)
    output_dir = Path(args.output_dir) if args.output_dir else adapter.parent / f"audio_probe_{manifest.stem}"
    print(f"Audio probe: {adapter} on {len(samples)} sample(s) from {manifest} -> {output_dir}")

    runtime = load_inference_runtime(config)
    model = runtime.model
    attach_adapter(config, model, adapter)
    model.eval()
    rate = runtime.codec.sample_rate

    def write_wav(path: Path, audio) -> None:
        sphn.write_wav(str(path), np.ascontiguousarray(audio), rate)

    results = []
    for sample in samples:
        sample_dir = output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            _export_context(sample, sample_dir)
            example = build_example(config, sample, runtime)
            tf_codes, ref_codes, metrics = teacher_forced_probe(config, runtime, model, example, torch)
            write_wav(sample_dir / "ref_resynth.wav", _decode(runtime, ref_codes, torch))
            write_wav(sample_dir / "tf_agent.wav", _decode(runtime, tf_codes, torch))
            for mode in MODES:
                audio, text_ids = generate_mode(config, runtime, model, sample, example, mode, torch)
                write_wav(sample_dir / f"{mode}.wav", audio)
                (sample_dir / f"{mode}.txt").write_text(decode_text(config, runtime, text_ids), encoding="utf-8")
            (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            results.append({"sample_id": sample.sample_id, "metrics": metrics, "error": None})
            print(json.dumps({"sample_id": sample.sample_id, **metrics}))
        except Exception as exc:  # one bad sample must not stop the probe
            results.append({"sample_id": sample.sample_id, "metrics": None, "error": str(exc)})
            print(json.dumps({"sample_id": sample.sample_id, "error": str(exc)}))

    report = {
        "checkpoint": str(adapter),
        "manifest": str(manifest),
        "count": len(results),
        "failed": sum(row["error"] is not None for row in results),
        "mean": mean_metrics([row["metrics"] for row in results if row["metrics"] is not None]),
        "samples": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: report[key] for key in ("count", "failed", "mean")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
