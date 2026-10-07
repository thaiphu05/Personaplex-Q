from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError:  # pragma: no cover - exercised on documentation-only environments
    torch = None
    save_file = None


@unittest.skipIf(torch is None, "PyTorch is required for LoRA module tests")
class StrictAdapterLoadTest(unittest.TestCase):
    def _make_model(self) -> torch.nn.Module:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.projection = torch.nn.Linear(4, 4)
                self.transformer.projection.lora_a = torch.nn.Linear(4, 2)
                self.transformer.projection.lora_b = torch.nn.Linear(2, 4)
                self.transformer.gate = torch.nn.Linear(4, 4)
                self.transformer.gate.lora_a = torch.nn.Linear(4, 2)
                self.transformer.gate.lora_b = torch.nn.Linear(2, 4)
        return Model()

    def _adapter(self, model, mutate=None) -> Path:
        from personaplex_finetuning.lora import adapter_state_dict

        state = adapter_state_dict(model)
        if mutate is not None:
            state = mutate(state)
        path = Path(tempfile.mkdtemp()) / "lora.safetensors"
        save_file(state, str(path))
        return path

    def test_loads_matching_adapter(self) -> None:
        from personaplex_finetuning.lora import load_adapter

        model = self._make_model()
        adapter = self._adapter(model)
        load_adapter(model, adapter)  # must not raise

    def test_missing_key_rejected(self) -> None:
        """A partial adapter (e.g. different prefix or stage) must fail loudly
        instead of leaving part of the model untrained."""
        from personaplex_finetuning.lora import load_adapter

        model = self._make_model()
        adapter = self._adapter(
            model, mutate=lambda state: {k: v for k, v in state.items() if "gate" not in k}
        )
        with self.assertRaises(RuntimeError):
            load_adapter(model, adapter)

    def test_unexpected_key_rejected(self) -> None:
        """Keys that do not exist in the model (typos, stale layers) must be
        rejected instead of silently dropped."""
        from personaplex_finetuning.lora import load_adapter

        model = self._make_model()
        adapter = self._adapter(
            model,
            mutate=lambda state: {**state, "transformer.stale.lora_a.weight": torch.zeros(4, 2)},
        )
        with self.assertRaises(RuntimeError):
            load_adapter(model, adapter)

    def test_adapter_state_dict_contains_only_lora_weights(self) -> None:
        from personaplex_finetuning.lora import adapter_state_dict

        model = self._make_model()
        state = adapter_state_dict(model)
        self.assertEqual(len(state), 4)
        for name in state:
            self.assertTrue(".lora_a." in name or ".lora_b." in name, name)


if __name__ == "__main__":
    unittest.main()
