"""Explicit local PersonaPlex runtime; deliberately contains no Hub fallback."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


def _pad_audio_window(audio, sample_rate: int, duration_sec: float):
    """Match the fixed final-chunk padding performed by the reference dataset loader."""
    import numpy as np

    expected_samples = round(sample_rate * duration_sec)
    if audio.shape[-1] >= expected_samples:
        return audio[..., :expected_samples]
    return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, expected_samples - audio.shape[-1])])


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    source: Path
    model_root: Path
    moshi_weight: Path
    mimi_weight: Path
    tokenizer: Path


@dataclass(frozen=True)
class RuntimePaths:
    model_root: Path
    source: Path

    def validate(self) -> ResolvedRuntimePaths:
        root = Path(self.model_root).resolve()
        source = Path(self.source).resolve()
        required = {
            "model.safetensors": root / "model.safetensors",
            "tokenizer-e351c8d8-checkpoint125.safetensors": root / "tokenizer-e351c8d8-checkpoint125.safetensors",
            "tokenizer_spm_32k_3.model": root / "tokenizer_spm_32k_3.model",
        }
        required_source = source / "moshi" / "models" / "loaders.py"
        if not required_source.is_file():
            raise FileNotFoundError(
                f"PersonaPlex source must contain moshi/models/loaders.py: {source}"
            )
        for name, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"required local PersonaPlex asset missing: {path} ({name})")
        return ResolvedRuntimePaths(
            source=source, model_root=root, moshi_weight=required["model.safetensors"],
            mimi_weight=required["tokenizer-e351c8d8-checkpoint125.safetensors"],
            tokenizer=required["tokenizer_spm_32k_3.model"],
        )


class SentencePieceTokenizer:
    padding_id = 3
    end_padding_id = 0

    def __init__(self, path: Path) -> None:
        sentencepiece = importlib.import_module("sentencepiece")
        self._processor = sentencepiece.SentencePieceProcessor(str(path))

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text))


class MimiCodec:
    """Mimi adapter using the same source helpers as PersonaPlex inference."""

    codebooks = 8

    def __init__(self, mimi, sample_rate: int, frame_rate: float, device: str, lm_helpers, cache_dir: Path | None = None) -> None:
        self.mimi = mimi
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.device = device
        self._helpers = lm_helpers
        self._voice_cache: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._cache_dir = cache_dir  # Optional persistent disk cache for conversation encoding

    def encode_conversation_stereo(self, path: Path, agent_channel: int, user_channel: int, start_sec: float, end_sec: float):
        import numpy as np
        import sphn
        import torch
        duration_sec = end_sec - start_sec
        # Windowed seek + direct resample in C++ (avoids decoding entire 30m file)
        audio, _ = sphn.read(str(path), start_sec=start_sec, duration_sec=duration_sec, sample_rate=self.sample_rate)
        audio = _pad_audio_window(audio, self.sample_rate, duration_sec)
        if agent_channel not in (0, 1) or user_channel not in (0, 1):
            raise ValueError(f"invalid channels {agent_channel}, {user_channel} for {path}")
        agent_audio = audio[agent_channel : agent_channel + 1]
        user_audio = audio[user_channel : user_channel + 1]
        # Batch encode both channels in a single Mimi GPU forward pass
        batch = torch.as_tensor(np.stack([agent_audio, user_audio], axis=0), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            codes = self.mimi.encode(batch)
        agent_codes = tuple(tuple(int(token) for token in stream.tolist()) for stream in codes[0])
        user_codes = tuple(tuple(int(token) for token in stream.tolist()) for stream in codes[1])
        return agent_codes, user_codes

    def encode_conversation_stereo_cached(self, path: Path, agent_channel: int, user_channel: int, start_sec: float, end_sec: float):
        """Stereo encode with persistent disk cache.

        On first call, encodes via Mimi GPU forward and saves a ``.pt`` sidecar.
        On subsequent calls, loads from the sidecar – no GPU work needed.
        Falls back to live encoding if cache_dir is not set.
        """
        if self._cache_dir is None:
            return self.encode_conversation_stereo(path, agent_channel, user_channel, start_sec, end_sec)

        import torch
        key = f"{path.name}_ch{agent_channel}{user_channel}_{start_sec:.3f}_{end_sec:.3f}"
        cache_file = self._cache_dir / (key + ".pt")
        if cache_file.is_file():
            data = torch.load(str(cache_file), map_location="cpu", weights_only=True)
            return data["agent"], data["user"]

        agent_codes, user_codes = self.encode_conversation_stereo(path, agent_channel, user_channel, start_sec, end_sec)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent_codes, "user": user_codes}, str(cache_file))
        return agent_codes, user_codes

    def encode_conversation(self, path: Path, channel: int, start_sec: float, end_sec: float):
        import sphn
        import torch
        duration_sec = end_sec - start_sec
        audio, _ = sphn.read(str(path), start_sec=start_sec, duration_sec=duration_sec, sample_rate=self.sample_rate)
        audio = _pad_audio_window(audio, self.sample_rate, duration_sec)
        if channel not in (0, 1):
            raise ValueError(f"invalid conversation window {start_sec}:{end_sec} for {path}")
        return self._encode(audio[channel : channel + 1], torch)

    def encode_voice_prompt(self, path: Path):
        key = str(path)
        if key in self._voice_cache:
            return self._voice_cache[key]
        import torch
        audio = self._helpers.load_audio(str(path), self.sample_rate)
        audio = self._helpers.normalize_audio(audio, self.sample_rate, -24.0)
        if audio.ndim == 1:
            audio = audio[None, :]
        codes = self._encode(audio[:1], torch)
        self._voice_cache[key] = codes
        return codes

    def sine(self, frames: int):
        """User placeholder for the prompt frames: LMGen's fixed sine tokens.

        Inference (LMGen) and the reference PersonaPlex finetune fill these
        frames with the constant ``SINE_TOKENS``; encoding a real sine wave
        through Mimi yields different, frame-varying codes, so the trained
        prompt context would not match the one seen at inference.
        """
        return self._constant_frames(self._helpers.SINE_TOKENS, frames)

    def silence(self, frames: int):
        """Agent silence for the prompt frames: LMGen's fixed ``SILENCE_TOKENS``."""
        return self._constant_frames(self._helpers.SILENCE_TOKENS, frames)

    def _constant_frames(self, tokens, frames: int):
        if frames < 0:
            raise ValueError("frames must be non-negative")
        codes = [int(token) for token in tokens]
        if len(codes) != self.codebooks:
            raise ValueError(f"expected {self.codebooks} prompt tokens, got {len(codes)}")
        return tuple((code,) * frames for code in codes)

    def _encode(self, audio, torch):
        with torch.no_grad():
            codes = self.mimi.encode(torch.as_tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
        return tuple(tuple(int(token) for token in stream.tolist()) for stream in codes)



@dataclass
class PersonaPlexRuntime:
    model: object
    codec: MimiCodec
    tokenizer: SentencePieceTokenizer
    initial_tokens: tuple[int, ...]
    zero_token: int
    delays: tuple[int, ...]
    qwen_family: str | None = None


@dataclass(frozen=True)
class QwenRuntimePaths:
    """Assets for the Qwen backbone swap.

    ``model_root`` keeps pointing at the local PersonaPlex checkpoint: the
    Mimi weights and the depth transformer are reused from it, while the text
    side comes from the Qwen model id.
    """

    model_root: Path
    source: Path
    qwen_model_id: str


def _steal_pp_module(pp_model, name: str):
    """Detach a submodule from the PersonaPlex model so it can be re-attached
    to the Qwen wrapper without double registration."""
    module = getattr(pp_model, name)
    pp_model._modules.pop(name, None)
    return module


def load_qwen_runtime(paths: QwenRuntimePaths, device: str = "cuda") -> PersonaPlexRuntime:
    """Build the Qwen-swap runtime from local PersonaPlex assets + a Qwen model.

    The PersonaPlex 7B transformer is loaded first only to graft its depth
    stack, then released before the Qwen decoder is loaded, keeping the peak
    GPU footprint to roughly one model at a time.
    """
    torch = importlib.import_module("torch")
    loaders = importlib.import_module("moshi.models.loaders")
    lm_helpers = importlib.import_module("moshi.models.lm")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    lm_dtype = torch.bfloat16
    dev_type = getattr(device, "type", str(device))
    if dev_type == "mps":
        try:
            _ = torch.zeros((1,), dtype=torch.bfloat16, device="mps")
        except Exception:
            lm_dtype = torch.float16

    resolved = RuntimePaths(paths.model_root, paths.source).validate()

    # 1. Graft the depth stack out of the PersonaPlex checkpoint.
    pp_model = loaders.get_moshi_lm(resolved.moshi_weight, device=device, dtype=lm_dtype)
    depformer = _steal_pp_module(pp_model, "depformer")
    depformer_in = _steal_pp_module(pp_model, "depformer_in")
    depformer_emb = _steal_pp_module(pp_model, "depformer_emb")
    linears = _steal_pp_module(pp_model, "linears")
    # Helium normalizes its hidden state before depformer_in; keeping that norm
    # lets the grafted depformer_in see the context distribution it learned.
    out_norm = _steal_pp_module(pp_model, "out_norm")
    # Helium's audio embedding tables seed the Qwen ones (PCA to the new width).
    helium_audio_emb = [table.weight.detach() for table in _steal_pp_module(pp_model, "emb")]
    schedule = pp_model.depformer_weights_per_step_schedule
    delays = tuple(int(delay) for delay in pp_model.delays)
    card = int(pp_model.card)
    dep_q = int(pp_model.dep_q)
    n_q = int(pp_model.n_q)
    del pp_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Load the frozen Qwen backbone and its tokenizer.
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(paths.qwen_model_id)
    qwen = transformers.AutoModelForCausalLM.from_pretrained(paths.qwen_model_id, torch_dtype=lm_dtype)
    qwen.to(device)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad = False

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    end_padding = int(tokenizer.eos_token_id)
    text_initial = int(pad)

    # Backbone hidden size varies (Qwen3-8B: 4096, Qwen3-4B: 2560,
    # Qwen2.5-3B: 2048, Qwen3-0.6B: 1024); never assume the default. The
    # wrapper's backbone_proj maps it into the 4096-dim Helium context that the
    # grafted depformer_in expects, at every backbone size.
    hidden_size = getattr(qwen.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(qwen.config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("cannot determine backbone hidden size from the Qwen config")
    hidden_size = int(hidden_size)
    # Count the hidden states the decoder actually returns (embeddings + layers)
    # instead of trusting the config across transformers versions.
    decoder = qwen.get_decoder() if hasattr(qwen, "get_decoder") else qwen.model
    with torch.no_grad():
        probe = torch.zeros(1, 1, hidden_size, device=device, dtype=lm_dtype)
        num_hidden_states = len(decoder(inputs_embeds=probe, output_hidden_states=True, return_dict=True).hidden_states)

    # 3. Assemble the wrapper.
    from .qwen_lm import QwenMoshiLM
    from .qwen_tokenizer import QwenTokenizer

    model = QwenMoshiLM(
        qwen,
        card=card,
        n_q=n_q,
        dep_q=dep_q,
        dim=hidden_size,
        delays=delays,
        text_padding_token_id=text_initial,
        end_of_text_padding_id=end_padding,
        text_initial_token_id=text_initial,
        depformer=depformer,
        depformer_in=depformer_in,
        depformer_emb=depformer_emb,
        linears=linears,
        depformer_weights_per_step_schedule=schedule,
        out_norm=out_norm,
        audio_emb_init=helium_audio_emb,
        num_hidden_states=num_hidden_states,
        device=device,
        dtype=lm_dtype,
    )
    del helium_audio_emb
    model.train()

    # 4. Mimi codec (frozen) for conversation encoding.
    mimi = loaders.get_mimi(resolved.mimi_weight, device=device)
    codec = MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, device, lm_helpers)

    initial_tokens = tuple(int(value) for value in model._get_initial_token()[0, :, 0].tolist())
    if len(initial_tokens) != 17:
        raise RuntimeError("loaded checkpoint is not the expected 17-stream PersonaPlex model")
    return PersonaPlexRuntime(
        model=model,
        codec=codec,
        tokenizer=QwenTokenizer(tokenizer),
        initial_tokens=initial_tokens,
        zero_token=int(model.zero_token_id),
        delays=delays,
        qwen_family=str(getattr(qwen.config, "model_type", "")),
    )


def load_runtime(paths: RuntimePaths, device: str = "cuda", qlora: bool = False, quant_type: str = "nf4") -> PersonaPlexRuntime:
    """Load from explicit local assets only. No Hugging Face function is imported."""
    resolved = paths.validate()
    source = str(resolved.source)
    if source not in sys.path:
        sys.path.insert(0, source)
    torch = importlib.import_module("torch")
    loaders = importlib.import_module("moshi.models.loaders")
    lm_helpers = importlib.import_module("moshi.models.lm")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    mimi = loaders.get_mimi(resolved.mimi_weight, device=device)
    lm_dtype = torch.bfloat16
    dev_type = getattr(device, "type", str(device))
    if dev_type == "mps":
        try:
            _ = torch.zeros((1,), dtype=torch.bfloat16, device="mps")
        except Exception:
            lm_dtype = torch.float16

    if qlora:
        from .lora import quantize_model_4bit
        model = loaders.get_moshi_lm(resolved.moshi_weight, device="cpu", dtype=lm_dtype)
        model = quantize_model_4bit(model, device=device, quant_type=quant_type)
    else:
        model = loaders.get_moshi_lm(resolved.moshi_weight, device=device, dtype=lm_dtype)

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    initial = tuple(int(value) for value in model._get_initial_token()[0, :, 0].tolist())
    if len(initial) != 17 or model.dep_q != 16 or model.n_q != 16:
        raise RuntimeError("loaded checkpoint is not the expected 17-stream PersonaPlex model")
    return PersonaPlexRuntime(
        model=model,
        codec=MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, device, lm_helpers),
        tokenizer=SentencePieceTokenizer(resolved.tokenizer),
        initial_tokens=initial,
        zero_token=int(model.zero_token_id),
        delays=tuple(int(delay) for delay in model.delays),
    )
