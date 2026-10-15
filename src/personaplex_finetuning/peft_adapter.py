"""Qwen backbone LoRA injection and trainable-parameter management.

Two Qwen generations are supported behind one entry point:

- ``qwen2`` (Qwen2.5-3B/7B): classic GQA decoder — LoRA on ``q_proj``/``v_proj``.
- ``qwen3`` (Qwen3-0.6B/4B/8B): classic GQA decoder — LoRA on ``q_proj``/``v_proj``.
- ``qwen3_5`` (Qwen3.5-9B): 3:1 hybrid (24 Gated DeltaNet + 8 Gated
  Attention) whose q/k/v/gate projection is a single fused ``attn_qkv``
  Linear — LoRA on ``attn_qkv`` (8 layers) + ``ffn_gate``/``ffn_up``/
  ``ffn_down`` (32 layers), the main adaptation channel for the DeltaNet
  layers that carry no attention.

The LoRA surface is auto-detected from ``config.model_type`` with a
module-name probe fallback, so switching backbones only requires changing
``model.qwen_id``. The depth transformer is nudged with FP32 LoRA as well
instead of being retrained (plan option B).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora import LoRALinear

INTERFACE_NAMES = (
    "depformer_in.",
    "linears.",
    "depformer_emb.",
    "text_depth_adapter.",
    "backbone_proj.",
    "out_norm.",
    "cb0_head.",
    "cb0_text_adapter.",
    "cb0_mlp.",
    "stream_gains.",
    "layer_mix.",
    "cb0_depth_gate",
    "acoustic_heads.",
)
# Interface owned only by the depformer decoder; a depformer checkpoint warm-starting
# an ``audio_decoder="heads"`` model drops these keys.
DEPFORMER_DECODER_NAMES = (
    "depformer.",
    "depformer_in.",
    "depformer_emb.",
    "linears.",
    "text_depth_adapter.",
    "backbone_proj.",
    "out_norm.",
    "cb0_depth_gate",
)
QWEN_TRAIN_STAGES = ("interface", "joint")


@dataclass(frozen=True)
class QwenSpec:
    """LoRA surface description for one Qwen generation."""

    family: str
    lora_targets: tuple[str, ...]


QWEN_SPECS: dict[str, QwenSpec] = {
    "qwen2": QwenSpec("qwen2", ("q_proj", "v_proj")),
    "qwen3": QwenSpec("qwen3", ("q_proj", "v_proj")),
    "qwen3_5": QwenSpec("qwen3_5", ("attn_qkv", "ffn_gate", "ffn_up", "ffn_down")),
}


def detect_qwen_spec(model) -> QwenSpec:
    """Identify the Qwen generation of a wrapped backbone.

    Prefers ``config.model_type``; falls back to probing the actual module
    names so the detection survives transformers renames.
    """
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type in QWEN_SPECS:
        return QWEN_SPECS[model_type]
    names = {name for name, _ in model.qwen.named_modules()}
    if any(name.endswith("attn_qkv") for name in names):
        return QWEN_SPECS["qwen3_5"]
    if any(name.endswith("q_proj") for name in names):
        return QWEN_SPECS["qwen3"]
    raise ValueError(
        "could not identify the Qwen architecture; expected model_type 'qwen3' or 'qwen3_5', "
        "or module names containing 'attn_qkv' / 'q_proj'"
    )


def _wrap_linears(root, rank: int, alpha: int, names: tuple[str, ...]) -> list[str]:
    """Wrap every ``nn.Linear`` under ``root`` whose name ends with a target suffix."""
    targets: list[tuple[object, str]] = []
    for name, module in root.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(target) for target in names):
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        targets.append((parent, attribute))
    if not targets:
        raise RuntimeError(f"no Linear modules matched LoRA targets {names}")
    for parent, attribute in targets:
        setattr(
            parent,
            attribute,
            LoRALinear(getattr(parent, attribute), rank=rank, alpha=alpha, adapter_dtype=torch.float32),
        )
    return [attribute for _, attribute in targets]


def inject_qwen_lora(model, rank: int = 64, alpha: int = 128, targets=None) -> list[str]:
    """Wrap the Qwen attention/FFN linears with FP32 LoRA adapters.

    ``targets`` selects the module-name suffixes explicitly; when omitted
    the surface is auto-detected from the model generation.
    """
    if not hasattr(model, "qwen"):
        raise ValueError("inject_qwen_lora expects a QwenMoshiLM wrapper with a .qwen attribute")
    spec = detect_qwen_spec(model)
    names = tuple(targets) if targets else spec.lora_targets
    return _wrap_linears(model.qwen, rank, alpha, names)


def inject_depformer_lora(model, rank: int = 64, alpha: int = 128) -> list[str]:
    """Wrap every Linear inside the reused depth transformer with FP32 LoRA."""
    if model.depformer is None:
        raise ValueError("depformer is not attached to the QwenMoshiLM wrapper")
    wrapped = []
    for name, module in model.depformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.depformer.get_submodule(parent_name) if parent_name else model.depformer
        setattr(
            parent,
            attribute,
            LoRALinear(getattr(parent, attribute), rank=rank, alpha=alpha, adapter_dtype=torch.float32),
        )
        wrapped.append(f"depformer.{name}")
    if not wrapped:
        raise RuntimeError("no Linear modules found in the depth transformer for LoRA")
    return wrapped


def _is_lora(name: str) -> bool:
    return ".lora_a." in name or ".lora_b." in name


def _is_qwen_adapter_param(name: str, depformer_full: bool = False) -> bool:
    """Every parameter a Qwen-swap checkpoint owns, trainable or not.

    Audio embeddings are always included: they are freshly initialized at load
    time, so a checkpoint without them could not reproduce the trained model.
    With ``depformer_mode="full"`` the dense depth transformer is owned too.
    """
    return (
        _is_lora(name)
        or name.startswith("emb.")
        or name.startswith(INTERFACE_NAMES)
        or (depformer_full and name.startswith("depformer."))
    )


def _depformer_full(model) -> bool:
    return getattr(model, "depformer_mode", "lora") == "full"


def configure_qwen_trainable(model, ft_embed: bool = False, stage: str = "joint") -> None:
    """Freeze the Qwen core; keep LoRA, interface and the active CB0 branch trainable.

    Trainable groups after this call:
    - interface: ``stream_gains``, ``layer_mix``, ``backbone_proj``, ``out_norm``,
      ``depformer_in``, ``linears[1:]``, ``depformer_emb``, ``text_depth_adapter``
      (``audio_decoder="depformer"``) or ``acoustic_heads`` (``"heads"``)
    - agent-CB0 branch per ``model.cb0_source``: ``cb0_head`` +
      ``cb0_text_adapter`` + ``cb0_mlp`` (qwen, merge), ``linears.0`` (depformer, merge),
      ``cb0_depth_gate`` (merge); unused branches stay frozen
    - audio embeddings ``emb.*`` (always in the ``interface`` stage, otherwise
      when ``ft_embed`` is True)
    - ``joint`` stage only: Qwen LoRA, plus the depth transformer as LoRA
      adapters (``depformer_mode="lora"``) or as dense weights (``"full"``)

    The ``interface`` stage aligns the freshly initialized bridges with the
    frozen backbone and the grafted depth stack before LoRA starts adapting.
    """
    if stage not in QWEN_TRAIN_STAGES:
        raise ValueError(f"qwen train stage must be one of {QWEN_TRAIN_STAGES}, got {stage!r}")
    train_lora = stage == "joint"
    train_emb = ft_embed or stage == "interface"
    cb0_source = getattr(model, "cb0_source", "qwen")
    train_depformer = _depformer_full(model) and stage == "joint"
    for name, parameter in model.named_parameters():
        if _is_lora(name):
            parameter.requires_grad = train_lora
        elif name.startswith("qwen."):
            parameter.requires_grad = False
        elif name.startswith("emb."):
            parameter.requires_grad = train_emb
        elif name.startswith("depformer."):
            parameter.requires_grad = train_depformer
        elif name.startswith(("cb0_head.", "cb0_text_adapter.", "cb0_mlp.")):
            parameter.requires_grad = cb0_source != "depformer"
        elif name.startswith("linears.0."):
            parameter.requires_grad = cb0_source != "qwen"
        elif name == "cb0_depth_gate":
            parameter.requires_grad = cb0_source == "merge"
        elif name.startswith(INTERFACE_NAMES):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def build_qwen_adapters(model, config, depformer_mode: str | None = None) -> list[str]:
    """Wrap a fresh Qwen-swap model with the configured adapters and trainable set.

    The single construction path for training, reload verification, inference
    and evaluation, so all of them rebuild the same parameter layout.
    ``depformer_mode`` overrides ``config.depformer_mode`` (checkpoint conversion).
    """
    mode = depformer_mode or getattr(config, "depformer_mode", "lora")
    qwen_targets = getattr(config, "qwen_targets", None)
    targets = tuple(qwen_targets.split(",")) if qwen_targets else None
    wrapped = inject_qwen_lora(model, rank=config.lora_qwen_rank, alpha=config.lora_qwen_alpha, targets=targets)
    if getattr(model, "audio_decoder", "depformer") == "heads":
        mode = "none"  # no depth transformer to adapt
    elif mode == "lora":
        wrapped += inject_depformer_lora(model, rank=config.lora_depformer_rank, alpha=config.lora_depformer_alpha)
    elif mode == "full":
        # FP32 master weights: small Adam steps vanish in bf16; autocast still runs bf16.
        model.depformer.float()
    else:
        raise ValueError(f"depformer_mode must be lora or full, got {mode!r}")
    model.depformer_mode = mode
    configure_qwen_trainable(model, ft_embed=config.ft_embed, stage=getattr(config, "train_stage", "joint"))
    return wrapped


def merge_depformer_lora(model, ft_embed: bool = False, stage: str = "joint") -> int:
    """Fold depth-transformer LoRA into dense FP32 weights and switch to ``full``.

    ``W <- W + (alpha / rank) * B @ A`` per adapter, then the plain Linear
    replaces its ``LoRALinear`` wrapper, so a LoRA checkpoint continues as
    full fine-tuning from exactly the same function. Returns the merge count.
    """
    adapters = [(name, module) for name, module in model.depformer.named_modules() if isinstance(module, LoRALinear)]
    for name, module in adapters:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.depformer.get_submodule(parent_name) if parent_name else model.depformer
        base = module.base
        with torch.no_grad():
            delta = (module.lora_b.weight.float() @ module.lora_a.weight.float()) * module.scale
            base.weight.data = base.weight.float() + delta
        setattr(parent, attribute, base)
    model.depformer.float()
    model.depformer_mode = "full"
    configure_qwen_trainable(model, ft_embed=ft_embed, stage=stage)
    return len(adapters)


def read_adapter_meta(adapter_file) -> dict:
    """``adapter.json`` saved next to a checkpoint (empty when absent)."""
    import json
    from pathlib import Path

    meta_file = Path(adapter_file).parent / "adapter.json"
    if not meta_file.is_file():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except ValueError:
        return {}


# Values of the mode keys for checkpoints saved before each option existed.
CHECKPOINT_MODE_DEFAULTS = {"cb0_source": "qwen", "depformer_mode": "lora", "audio_decoder": "depformer"}


def checkpoint_modes(meta: dict) -> dict[str, str]:
    """CB0 source, depformer mode and audio decoder a checkpoint was trained with."""
    return {key: str(meta.get(key, default)) for key, default in CHECKPOINT_MODE_DEFAULTS.items()}


def trainable_summary(model) -> dict[str, int]:
    """Trainable parameter counts per group, to measure (not guess) run size."""
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora(name):
            group = "qwen_lora" if name.startswith("qwen.") else "depformer_lora"
        elif name.startswith("depformer."):
            group = "depformer_dense"
        elif name.startswith("emb."):
            group = "audio_embeddings"
        elif name.startswith("acoustic_heads."):
            group = "audio_heads"
        else:
            group = "interface"
        groups[group] = groups.get(group, 0) + parameter.numel()
    return groups


def qwen_adapter_state_dict(model):
    """Collect every adapter parameter (LoRA, interface, audio embeddings, dense depformer if full)."""
    full = _depformer_full(model)
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if _is_qwen_adapter_param(name, full)
    }


def load_qwen_adapter(model, path, warm_start_heads: bool = False) -> list[str]:
    """Load a Qwen-swap checkpoint, rejecting any key mismatch.

    ``warm_start_heads`` loads a depformer-decoder checkpoint into an
    ``audio_decoder="heads"`` model: the depformer-only keys are dropped and
    the new ``acoustic_heads`` keep their fresh init. Every other key must match.

    Returns the parameters the checkpoint did not supply (kept at their init),
    so a caller restoring optimizer state knows the parameter set grew.
    """
    from safetensors.torch import load_file
    state = load_file(str(path))
    # Added after checkpoints already existed; their identity/zero init
    # reproduces those checkpoints exactly, so they may be absent.
    allowed_missing = {"cb0_depth_gate"} | {name for name, _ in model.named_parameters() if name.startswith("cb0_mlp.")}
    if warm_start_heads:
        state = {name: value for name, value in state.items() if not name.startswith(DEPFORMER_DECODER_NAMES)}
        allowed_missing |= {name for name, _ in model.named_parameters() if name.startswith("acoustic_heads.")}
    full = _depformer_full(model)
    expected = {name for name, _ in model.named_parameters() if _is_qwen_adapter_param(name, full)}
    supplied = set(state)
    missing = sorted(expected - supplied - allowed_missing)
    unexpected = sorted(supplied - {name for name, _ in model.named_parameters()})
    if not expected or missing or unexpected:
        raise RuntimeError(
            f"qwen adapter mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)
    return sorted(expected - supplied)
