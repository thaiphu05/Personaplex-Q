"""Teacher-forced decode of one training sample, bypassing LMGen.

Runs the training forward on the exact training example and decodes the
model's argmax predictions for the agent (text + CB0..CB7) over the dialogue.
If this replays the sample but free-running inference (``tools.inference_smoke``)
does not, the model is fine and the gap is in the inference path.

python -m tools.teacher_forced_decode \
    --config configs/qwen3-0.6b-overfit.yaml \
    --adapter runs/qwen3-06b-overfit/train_*/checkpoints/checkpoint_000500 \
    --index 0 --output-dir outputs/overfit_tf

Writes ``tf_agent.wav`` / ``tf_agent.txt`` (model) and ``ref_agent.txt`` (target).
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

AGENT_CODEBOOKS = 8


def summarize(output, example, filler_ids, torch) -> dict:
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


def main() -> int:
    import numpy as np
    import sphn
    import torch

    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset
    from personaplex_finetuning.inference import attach_adapter, decode_text, load_inference_runtime
    from personaplex_finetuning.train import build_example

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/teacher_forced")
    args = parser.parse_args()

    config = load_config(args.config)
    sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
    print(f"Teacher-forced sample {sample.sample_id}: window {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f} s")

    runtime = load_inference_runtime(config)
    model = runtime.model
    attach_adapter(config, model, Path(args.adapter).resolve())
    model.eval()

    example = build_example(config, sample, runtime)
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    precision = str(getattr(config, "mixed_precision", "bf16")).lower()
    if str(config.device).startswith("cuda") and torch.cuda.is_available() and precision in ("bf16", "fp16"):
        autocast = torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    else:
        autocast = contextlib.nullcontext()
    with torch.no_grad(), autocast:
        output = model.forward_train(codes)

    filler = (0, runtime.tokenizer.padding_id, runtime.tokenizer.end_padding_id)  # same filter as inference
    result = summarize(output, example, filler, torch)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pcm = runtime.codec.mimi.decode(result["audio"].unsqueeze(0))
    sphn.write_wav(str(output_dir / "tf_agent.wav"), np.ascontiguousarray(pcm.squeeze().float().cpu().numpy()), runtime.codec.sample_rate)
    (output_dir / "tf_agent.txt").write_text(decode_text(config, runtime, result["text_ids"]), encoding="utf-8")
    (output_dir / "ref_agent.txt").write_text(decode_text(config, runtime, result["ref_ids"]), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    print(json.dumps({"output_dir": str(output_dir), **result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
