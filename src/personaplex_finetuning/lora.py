"""Minimal LoRA injection for PersonaPlex transformer Linear modules."""

from __future__ import annotations

import torch


def quantize_model_4bit(model, device: str, quant_type: str = "nf4"):
    """Replace dense Linear modules before moving their NF4 weights to CUDA.

    bitsandbytes quantizes ``Linear4bit`` on ``.to(cuda)``; see
    https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit
    """
    import torch
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("QLoRA requires bitsandbytes; install requirements.txt") from exc
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("quant_type must be nf4 or fp4")
    targets: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        targets.append((parent, attribute, module))
    if not targets:
        raise RuntimeError("no Linear modules found to quantize for QLoRA")
    for parent, attribute, base in targets:
        quantized = bnb.nn.Linear4bit(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type=quant_type,
            device="cpu",
        )
        quantized.load_state_dict(base.state_dict())
        setattr(parent, attribute, quantized)
    return model.to(device)


class LoRALinear(torch.nn.Module):
    """Linear module augmented with a low-rank adapter.

    The wrapped base module is frozen; only ``lora_a``/``lora_b`` train.
    ``forward`` computes ``base(x) + lora_b(lora_a(x)) * scale`` with
    ``scale = alpha / rank``.

    Args:
        base: the original ``nn.Linear`` (or 4-bit linear) module.
        rank: adapter rank.
        alpha: scaling numerator; effective scale is ``alpha / rank``.
        adapter_dtype: dtype for adapter weights; defaults to the base weight
            dtype. Use ``torch.float32`` to keep optimizer updates stable
            under bf16 mixed precision.
        dropout: dropout applied before the adapter path.
    """

    def __init__(self, base, rank: int, alpha: float, adapter_dtype=None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = torch.nn.Dropout(dropout)
        if adapter_dtype is None:
            adapter_dtype = getattr(base, "compute_dtype", None) or base.weight.dtype
        if not adapter_dtype.is_floating_point:
            adapter_dtype = torch.bfloat16
        self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=adapter_dtype)
        self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=adapter_dtype)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value):
        base_output = self.base(value)
        lora_input = value if value.dtype == self.lora_a.weight.dtype else value.to(self.lora_a.weight.dtype)
        delta = self.lora_b(self.lora_a(self.dropout(lora_input))) * self.scale
        return base_output + delta.to(base_output.dtype)

    @property
    def weight(self):
        """Effective weight for Moshi modules that access ``Linear.weight`` directly."""
        base_weight = self.base.weight
        quant_state = getattr(base_weight, "quant_state", None)
        if quant_state is not None:
            import bitsandbytes.functional as bnb_functional
            base_weight = bnb_functional.dequantize_4bit(base_weight.data, quant_state)
        lora_weight = (self.lora_b.weight @ self.lora_a.weight) * self.scale
        if base_weight.shape != lora_weight.shape or base_weight.numel() != lora_weight.numel():
            return base_weight.to(self.lora_a.weight.dtype)
        return base_weight.to(self.lora_a.weight.dtype) + lora_weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def inject_lora(model, rank: int, alpha: float, dropout: float = 0.0, prefixes: tuple[str, ...] = ("transformer",)) -> list[str]:
    import torch

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")

    targets: list[tuple[str, object, str]] = []
    for prefix in prefixes:
        root = getattr(model, prefix, None)
        if root is None:
            continue
        for name, module in root.named_modules():
            is_dense_linear = isinstance(module, torch.nn.Linear)
            is_4bit_linear = module.__class__.__module__.startswith("bitsandbytes.") and module.__class__.__name__ == "Linear4bit"
            if is_dense_linear or is_4bit_linear:
                parent_name, _, attribute = name.rpartition(".")
                parent = root.get_submodule(parent_name) if parent_name else root
                targets.append((f"{prefix}.{name}", parent, attribute))
    if not targets:
        raise RuntimeError("no PersonaPlex transformer Linear modules found for LoRA")
    for name, parent, attribute in targets:
        setattr(parent, attribute, LoRALinear(getattr(parent, attribute), rank=rank, alpha=alpha, dropout=dropout))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_a." in name or ".lora_b." in name
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise AssertionError("only LoRA parameters may be trainable")
    return [name for name, _, _ in targets]


def adapter_state_dict(model):
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if ".lora_a." in name or ".lora_b." in name}


def load_adapter(model, path) -> None:
    """Load LoRA adapter weights, rejecting any key mismatch.

    Adapters saved with a different rank, alpha, or target prefix silently
    leave part of the model untrained; fail loudly instead.
    """
    from safetensors.torch import load_file
    state = load_file(str(path))
    model_sd = model.state_dict()
    expected = {name for name, _ in model_sd.items() if ".lora_a." in name or ".lora_b." in name}
    supplied = set(state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - set(model_sd))
    if not expected or missing or unexpected:
        raise RuntimeError(
            f"adapter mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)
