"""Train vs inference parity on one training sample.

Feeds LMGen the exact training tokens (every stream forced, nothing sampled)
and compares its per-frame text and agent-CB0 predictions with the training
forward. Identical model + identical tokens must give identical predictions;
the report shows whether they do and, if not, from which dialogue frame on.
It also compares the voice-prompt codes of both paths (full-file encode at
training vs LMGen's per-frame streaming encode).

python -m tools.inference_parity \
    --config configs/qwen3-0.6b-overfit.yaml \
    --adapter runs/qwen3-06b-overfit/train_*/checkpoints/checkpoint_000500 \
    --index 0 --output-dir outputs/parity [--train-voice-codes]

``--train-voice-codes`` replays the training voice-prompt codes instead of
LMGen's own encoding, isolating that difference.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path


def compare_streams(train_ids, lmgen_ids, targets) -> dict:
    """Agreement between training and forced-LMGen argmax ids for one stream.

    All inputs are equal-length 1-D integer sequences over dialogue frames.
    """
    if not len(train_ids) == len(lmgen_ids) == len(targets):
        raise ValueError(f"length mismatch: {len(train_ids)}, {len(lmgen_ids)}, {len(targets)}")
    frames = len(targets)
    first = None
    for frame, (train, lmgen, target) in enumerate(zip(train_ids, lmgen_ids, targets)):
        if train != lmgen:
            first = {"frame": frame, "train": int(train), "lmgen": int(lmgen), "target": int(target)}
            break

    def rate(pairs) -> float | None:
        return sum(a == b for a, b in pairs) / frames if frames else None

    # Agreement of lmgen[t] with train[t + shift]: a peak away from shift 0
    # means the inference stream runs that many frames off the training one.
    shifts = {}
    for shift in range(-2, 3):
        pairs = [(lmgen_ids[t], train_ids[t + shift]) for t in range(frames) if 0 <= t + shift < frames]
        shifts[str(shift)] = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
    return {
        "frames": frames,
        "lmgen_vs_train_by_shift": shifts,
        "lmgen_vs_train": rate(zip(lmgen_ids, train_ids)),
        "lmgen_vs_target": rate(zip(lmgen_ids, targets)),
        "train_vs_target": rate(zip(train_ids, targets)),
        "first_mismatch": first,
    }


def compare_codes(train_codes, lmgen_codes) -> dict:
    """Frame counts and per-codebook code mismatches of two [8][T] code lists."""
    train_frames = len(train_codes[0]) if train_codes else 0
    lmgen_frames = len(lmgen_codes[0]) if lmgen_codes else 0
    shared = min(train_frames, lmgen_frames)
    mismatches = [
        sum(int(a) != int(b) for a, b in zip(train[:shared], lmgen[:shared]))
        for train, lmgen in zip(train_codes, lmgen_codes)
    ]
    return {"train_frames": train_frames, "lmgen_frames": lmgen_frames, "compared_frames": shared,
            "mismatches_per_codebook": mismatches}


def main() -> int:
    import importlib

    import torch

    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset
    from personaplex_finetuning.inference import align_generator_with_training, attach_adapter, load_inference_runtime
    from personaplex_finetuning.text_normalization import encode_system_prompt
    from personaplex_finetuning.train import build_example
    from tools.teacher_forced_decode import summarize

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/parity")
    parser.add_argument("--train-voice-codes", action="store_true",
                        help="Replay the training voice-prompt codes instead of LMGen's streaming encode.")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip align_generator_with_training (reproduces the old one-frame offset).")
    args = parser.parse_args()

    config = load_config(args.config)
    sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
    print(f"Parity sample {sample.sample_id}: window {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f} s")

    runtime = load_inference_runtime(config)
    model = runtime.model
    attach_adapter(config, model, Path(args.adapter).resolve())
    model.eval()
    mimi = runtime.codec.mimi
    device = config.device
    is_qwen = config.backbone == "qwen"

    precision = str(getattr(config, "mixed_precision", "bf16")).lower()
    if str(device).startswith("cuda") and torch.cuda.is_available() and precision in ("bf16", "fp16"):
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    else:
        autocast = contextlib.nullcontext

    # 1. Training path: teacher-forced forward on the training example.
    example = build_example(config, sample, runtime)
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), autocast():
        train_out = model.forward_train(codes)
    filler = (0, runtime.tokenizer.padding_id, runtime.tokenizer.end_padding_id)
    teacher_forced = summarize(train_out, example, filler, torch)["metrics"]
    start = int(example.prompt_frames)
    train_text_logits = train_out.text_logits[0, 0, start:].float()
    train_audio_logits = train_out.logits[0, :8].float()  # [8, T, card], undelayed frames

    lm_module = importlib.import_module("moshi.models.lm")
    compile_module = importlib.import_module("moshi.utils.compile")
    gen = dict(config.generation or {})

    def make_generator():
        generator = lm_module.LMGen(
            model, audio_silence_frame_cnt=int(gen.get("audio_silence_frame_cnt", 6)),
            sample_rate=runtime.codec.sample_rate, frame_rate=runtime.codec.frame_rate,
            device=str(device), use_sampling=False, return_logits=True,
        )
        generator.zero_text_code = runtime.tokenizer.padding_id  # as in inference.generate
        generator.text_prompt_tokens = encode_system_prompt(runtime.tokenizer, sample.text_prompt, config.text_prompt_template)
        return generator

    # 2. Voice prompt: training full-file encode vs LMGen per-frame streaming encode.
    train_voice = runtime.codec.encode_voice_prompt(sample.voice_prompt_wav)
    probe = make_generator()
    probe.load_voice_prompt(str(sample.voice_prompt_wav))
    with torch.no_grad(), mimi.streaming(1):
        frames = [frame[0, :, 0].tolist() for frame in probe._encode_voice_prompt_frames(mimi)]
    lmgen_voice = [[frame[k] for frame in frames] for k in range(8)]
    voice_report = compare_codes(train_voice, lmgen_voice)

    # 3. LMGen path with every stream forced to the training tokens.
    generator = make_generator()
    streams = example.input_codes
    dialogue_frames = len(streams[0]) - start
    text_steps, audio_steps = [], []
    if is_qwen:
        model.start_streaming_caches()
    try:
        with torch.no_grad(), autocast(), compile_module.no_cuda_graph(), mimi.streaming(1), generator.streaming(1):
            if not args.no_align:
                align_generator_with_training(generator)
            aligned_offset = int(generator._streaming_state.offset)
            if args.train_voice_codes:
                for frame in range(len(train_voice[0])):
                    voice_frame = torch.tensor([[train_voice[k][frame]] for k in range(8)], device=device).unsqueeze(0)
                    generator._step_voice_prompt_frame(voice_frame, [])
                generator._step_audio_silence()
                generator._step_text_prompt()
                generator._step_audio_silence()
            else:
                generator.load_voice_prompt(str(sample.voice_prompt_wav))
                generator.step_system_prompts(mimi)
            lmgen_prompt_frames = int(generator._streaming_state.offset) - aligned_offset
            for d in range(dialogue_frames):
                t = start + d
                user = torch.tensor([[streams[9 + k][t]] for k in range(8)], device=device).unsqueeze(0)
                agent = torch.tensor([[streams[1 + k][t]] for k in range(8)], device=device).unsqueeze(0)
                _, logits = generator.step(input_tokens=user, moshi_tokens=agent, text_token=int(streams[0][t]))
                if logits is None:
                    raise RuntimeError(f"LMGen returned no logits at dialogue frame {d}")
                text_logits, audio_logits = logits
                text_steps.append(text_logits[0, 0, 0].float())
                audio_steps.append(audio_logits[0, :8].float())
    finally:
        if is_qwen:
            model.stop_streaming_caches()

    lmgen_text_logits = torch.stack(text_steps)
    lmgen_audio_logits = torch.stack(audio_steps)  # [D, 8, card], one row per LMGen step

    def stream_report(train_logits, lmgen_logits, targets) -> dict:
        report = compare_streams(
            train_logits.nan_to_num().argmax(-1).tolist(), lmgen_logits.argmax(-1).tolist(), targets,
        )
        diff = (train_logits.nan_to_num() - lmgen_logits).abs().amax(dim=-1)
        report["logit_max_abs_diff"] = {"mean": float(diff.mean()), "max": float(diff.max())}
        return report

    result = {
        "sample_id": sample.sample_id,
        "train_voice_codes": bool(args.train_voice_codes),
        "aligned": not args.no_align,
        "prompt_frames": {"train": start, "lmgen": lmgen_prompt_frames},
        "voice_prompt": voice_report,
        "teacher_forced": teacher_forced,
        "text": stream_report(train_text_logits, lmgen_text_logits, list(streams[0][start:])),
    }
    # LMGen step d predicts codebook k for frame d - delay_k (CB1..7 lag one frame),
    # while the training logits are already undelayed: align on the frame.
    for k in range(8):
        delay = int(model.delays[1 + k])
        frames = dialogue_frames - delay
        result[f"agent_cb{k}"] = stream_report(
            train_audio_logits[k, start : start + frames],
            lmgen_audio_logits[delay : delay + frames, k],
            list(streams[1 + k][start : start + frames]),
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
