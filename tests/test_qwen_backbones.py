import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised on documentation-only environments
    torch = None
    nn = None


def _make_backbone(linear_names):
    """Build a module tree: one named Linear per layer, with dotted paths."""
    backbone = nn.Module()
    backbone.layers = nn.ModuleList()
    for name in linear_names:
        layer = nn.Module()
        setattr(layer, name, nn.Linear(4, 4))
        backbone.layers.append(layer)
    return backbone


def _make_wrapper(linear_names, model_type=None):
    from personaplex_finetuning.qwen_lm import QwenMoshiLM  # only for hasattr checks

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen = _make_backbone(linear_names)
            if model_type is not None:
                self.config = SimpleNamespace(model_type=model_type)

    return Wrapper()


@unittest.skipIf(torch is None, "PyTorch is required for Qwen backbone tests")
class DetectQwenSpecTest(unittest.TestCase):
    def test_detects_qwen3_by_model_type(self) -> None:
        from personaplex_finetuning.peft_adapter import detect_qwen_spec

        model = _make_wrapper(["q_proj", "v_proj"], model_type="qwen3")
        spec = detect_qwen_spec(model)
        self.assertEqual(spec.family, "qwen3")
        self.assertEqual(spec.lora_targets, ("q_proj", "v_proj"))

    def test_detects_qwen3_5_by_model_type(self) -> None:
        from personaplex_finetuning.peft_adapter import detect_qwen_spec

        model = _make_wrapper(["attn_qkv", "ffn_gate"], model_type="qwen3_5")
        spec = detect_qwen_spec(model)
        self.assertEqual(spec.family, "qwen3_5")
        self.assertEqual(spec.lora_targets, ("attn_qkv", "ffn_gate", "ffn_up", "ffn_down"))

    def test_detects_by_module_probe_without_config(self) -> None:
        from personaplex_finetuning.peft_adapter import detect_qwen_spec

        self.assertEqual(detect_qwen_spec(_make_wrapper(["attn_qkv"])).family, "qwen3_5")
        self.assertEqual(detect_qwen_spec(_make_wrapper(["q_proj", "k_proj"])).family, "qwen3")

    def test_unknown_architecture_rejected(self) -> None:
        from personaplex_finetuning.peft_adapter import detect_qwen_spec

        with self.assertRaises(ValueError):
            detect_qwen_spec(_make_wrapper(["linear"] * 3))


@unittest.skipIf(torch is None, "PyTorch is required for Qwen backbone tests")
class InjectQwenLoraTest(unittest.TestCase):
    def _is_wrapped(self, module):
        from personaplex_finetuning.lora import LoRALinear

        return isinstance(module, LoRALinear)

    def test_qwen3_wraps_only_q_and_v(self) -> None:
        from personaplex_finetuning.peft_adapter import inject_qwen_lora

        model = _make_wrapper(["q_proj", "k_proj", "v_proj", "o_proj", "ffn_gate"], model_type="qwen3")
        inject_qwen_lora(model, rank=64, alpha=128)

        for name, module in model.qwen.named_modules():
            if not isinstance(module, nn.Linear) and not self._is_wrapped(module):
                continue
            attribute = name.rpartition(".")[2]
            if attribute in {"q_proj", "v_proj"}:
                self.assertTrue(self._is_wrapped(module), name)
            else:
                self.assertFalse(self._is_wrapped(module), f"{name} must stay untouched for qwen3")

    def test_qwen3_5_wraps_attention_and_ffn(self) -> None:
        from personaplex_finetuning.peft_adapter import inject_qwen_lora

        model = _make_wrapper(["attn_qkv", "ffn_gate", "ffn_up", "ffn_down"], model_type="qwen3_5")
        inject_qwen_lora(model, rank=64, alpha=128)

        for name, module in model.qwen.named_modules():
            if not self._is_wrapped(module):
                continue
            attribute = name.rpartition(".")[2]
            self.assertIn(attribute, {"attn_qkv", "ffn_gate", "ffn_up", "ffn_down"})

    def test_targets_override(self) -> None:
        from personaplex_finetuning.peft_adapter import inject_qwen_lora

        model = _make_wrapper(["q_proj", "v_proj", "k_proj"], model_type="qwen3")
        inject_qwen_lora(model, rank=32, alpha=64, targets=("q_proj", "k_proj"))

        wrapped = {
            name.rpartition(".")[2]
            for name, module in model.qwen.named_modules()
            if self._is_wrapped(module)
        }
        self.assertEqual(wrapped, {"q_proj", "k_proj"})

    def test_adapters_are_fp32_with_scale_two(self) -> None:
        from personaplex_finetuning.peft_adapter import inject_qwen_lora

        model = _make_wrapper(["q_proj", "v_proj"], model_type="qwen3")
        inject_qwen_lora(model, rank=64, alpha=128)
        wrapped = [module for module in model.qwen.modules() if self._is_wrapped(module)]
        self.assertTrue(wrapped)
        for module in wrapped:
            self.assertEqual(module.lora_a.weight.dtype, torch.float32)
            self.assertEqual(module.scale, 2.0)

    def test_missing_targets_raise(self) -> None:
        from personaplex_finetuning.peft_adapter import inject_qwen_lora

        model = _make_wrapper(["other"], model_type="qwen3")
        with self.assertRaises(RuntimeError):
            inject_qwen_lora(model, rank=64, alpha=128)


if __name__ == "__main__":
    unittest.main()
