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
import json
from pathlib import Path

# The scoring lives in the package so training-time evaluation can use it too.
from personaplex_finetuning.evaluate import teacher_forced_summary as summarize  # noqa: F401  (re-exported)


def main() -> int:
    import numpy as np
    import sphn
    import torch

    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset
    from personaplex_finetuning.inference import attach_adapter, decode_text, load_inference_runtime
    from personaplex_finetuning.evaluate import teacher_forced

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

    result = teacher_forced(config, runtime, model, sample)

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
