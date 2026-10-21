"""Word-timestamped ASR on generated audio, offline and explicit.

The model path is always passed in by the caller: nothing here downloads a
checkpoint or reads a default from the environment.
"""

from __future__ import annotations

from pathlib import Path

WHISPER_SAMPLE_RATE = 16000
HF_FILES = ("config.json", "preprocessor_config.json")


def is_hf_whisper_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and all((path / name).is_file() for name in HF_FILES)


class WhisperWordASR:
    """Local Whisper: an openai-whisper ``.pt`` file or a Hugging Face directory."""

    def __init__(self, model_path: Path, language: str = "vi", device: str = "cuda") -> None:
        model_path = Path(model_path)
        self.model_path = model_path
        self.language = language
        if is_hf_whisper_dir(model_path):
            import torch
            from transformers import pipeline

            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=str(model_path),
                device=device,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                model_kwargs={"local_files_only": True},
            )
            self.model = None
        elif model_path.is_file():
            import whisper

            self.pipeline = None
            self.model = whisper.load_model(str(model_path), device=device)
        else:
            raise FileNotFoundError(
                f"ASR model missing: {model_path}; expected an openai-whisper .pt file "
                f"or a Hugging Face directory with {', '.join(HF_FILES)}"
            )

    def transcribe_words(self, wav_path: Path, start_sec: float = 0.0) -> list[dict]:
        """``[{"word", "start", "end"}]`` in seconds from the start of ``wav_path``."""
        import sphn

        pcm, _ = sphn.read(str(wav_path), sample_rate=WHISPER_SAMPLE_RATE)
        audio = pcm.mean(axis=0).astype("float32")[int(start_sec * WHISPER_SAMPLE_RATE):]
        if self.pipeline is not None:
            result = self.pipeline(
                {"raw": audio, "sampling_rate": WHISPER_SAMPLE_RATE},
                return_timestamps="word",
                chunk_length_s=30,
                generate_kwargs={"language": self.language, "task": "transcribe"},
            )
            words = []
            for chunk in result.get("chunks", []):
                start, end = chunk["timestamp"]
                if chunk["text"].strip() and start is not None:
                    words.append(
                        {
                            "word": chunk["text"].strip(),
                            "start": start_sec + float(start),
                            "end": start_sec + float(end if end is not None else start),
                        }
                    )
            return words
        result = self.model.transcribe(
            audio,
            language=self.language,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return [
            {
                "word": word["word"].strip(),
                "start": start_sec + float(word["start"]),
                "end": start_sec + float(word["end"]),
            }
            for segment in result.get("segments", [])
            for word in segment.get("words", [])
            if word["word"].strip()
        ]
