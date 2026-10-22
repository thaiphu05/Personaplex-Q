"""
python -m your_package.smoke \
    --adapter outputs/checkpoint/adapter.pt \
    --input-file ./test.mp3 \
    --voice-prompt voice.wav
    --text-prompt ""

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.inference import generate_folder, smoke


def select_inference_window(sample, start: float | None, window_seconds: float):
    """Optionally replace the dataset default with an exact inference window."""
    if start is None:
        return sample
    if start < 0:
        raise ValueError(f"--start must be non-negative, got {start}")
    end = start + window_seconds
    if end > sample.audio.duration_sec:
        raise ValueError(
            f"--start {start:g} requires {window_seconds:g} seconds, but the conversation is only "
            f"{sample.audio.duration_sec:g} seconds long"
        )
    return sample.with_window(start, end)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds for one exact data.window_seconds inference window.",
    )
    parser.add_argument(
        "--input-path",
        "--input-file",
        dest="input_file",
        type=Path,
        default=None,
        help="Optional WAV/MP3 input used instead of the sample conversation audio.",
    )
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--input-dir", type=Path, default=None, help="Run every WAV/MP3/FLAC in this folder (one output subdir per file).")
    parser.add_argument("--voice-prompt", type=Path, default=None, help="Agent voice prompt WAV (folder mode; default: dataset sample).")
    parser.add_argument("--text-prompt", default=None, help="System text prompt (folder mode; default: dataset sample).")
    parser.add_argument("--extensions", default="wav,mp3,flac", help="Comma-separated audio extensions for --input-dir.")
    parser.add_argument("--recursive", action="store_true", help="Walk --input-dir recursively.")
    parser.add_argument("--limit", type=int, default=None, help="Max files in --input-dir.")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.input_dir is not None:
        voice_prompt = args.voice_prompt
        text_prompt = args.text_prompt
        if voice_prompt is None or text_prompt is None:
            sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
            voice_prompt = voice_prompt or sample.voice_prompt_wav
            text_prompt = text_prompt or sample.text_prompt
        report = generate_folder(
            config=config,
            input_dir=args.input_dir.resolve(),
            adapter=Path(args.adapter).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            voice_prompt_wav=Path(voice_prompt).resolve(),
            text_prompt=str(text_prompt),
            extensions=tuple(part.strip() for part in args.extensions.split(",") if part.strip()),
            recursive=args.recursive,
            limit=args.limit,
        )
        print(json.dumps({key: report[key] for key in ("count", "failed")}, indent=2))
        return 0

    sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
    sample = select_inference_window(sample, args.start, config.window_seconds)
    print(
        f"Inference sample {sample.sample_id}: "
        f"window {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f} seconds"
    )
    smoke(
        config=config,
        sample=sample,
        adapter=Path(args.adapter).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        input_file=args.input_file.resolve() if args.input_file is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
