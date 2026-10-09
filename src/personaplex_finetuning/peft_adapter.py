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
    "stream_gains.",
    "layer_mix.",
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


def _is_qwen_adapter_param(name: str) -> bool:
    """Every parameter a Qwen-swap checkpoint owns, trainable or not.

    Audio embeddings are always included: they are freshly initialized at load
    time, so a checkpoint without them could not reproduce the trained model.
    """
    return _is_lora(name) or name.startswith("emb.") or name.startswith(INTERFACE_NAMES)


def configure_qwen_trainable(model, ft_embed: bool = False, stage: str = "joint") -> None:
    """Freeze the Qwen core and the depth stack; keep LoRA + interface trainable.

    Trainable groups after this call:
    - interface: ``stream_gains``, ``layer_mix``, ``backbone_proj``, ``out_norm``,
      ``cb0_head``, ``cb0_text_adapter``, ``depformer_in``,
      ``linears`` (except ``linears.0``: CB0 comes from ``cb0_head``),
      ``depformer_emb``, ``text_depth_adapter``
    - audio embeddings ``emb.*`` (always in the ``interface`` stage, otherwise
      when ``ft_embed`` is True)
    - ``joint`` stage only: Qwen and depth-transformer LoRA adapters

    The ``interface`` stage aligns the freshly initialized bridges with the
    frozen backbone and the grafted depth stack before LoRA starts adapting.
    """
    if stage not in QWEN_TRAIN_STAGES:
        raise ValueError(f"qwen train stage must be one of {QWEN_TRAIN_STAGES}, got {stage!r}")
    train_lora = stage == "joint"
    train_emb = ft_embed or stage == "interface"
    for name, parameter in model.named_parameters():
        if _is_lora(name):
            parameter.requires_grad = train_lora
        elif name.startswith("qwen."):
            parameter.requires_grad = False
        elif name.startswith("emb."):
            parameter.requires_grad = train_emb
        elif name.startswith("linears.0."):
            # CB0 comes from cb0_head; the depformer's CB0 output is never used.
            parameter.requires_grad = False
        elif name.startswith(INTERFACE_NAMES):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False


def qwen_adapter_state_dict(model):
    """Collect every adapter parameter (LoRA, interface, audio embeddings)."""
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if _is_qwen_adapter_param(name)}


def load_qwen_adapter(model, path) -> None:
    """Load a Qwen-swap checkpoint, rejecting any key mismatch."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    expected = {name for name, _ in model.named_parameters() if _is_qwen_adapter_param(name)}
    supplied = set(state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - {name for name, _ in model.named_parameters()})
    if not expected or missing or unexpected:
        raise RuntimeError(
            f"qwen adapter mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)
