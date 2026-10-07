import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised on documentation-only environments
    torch = None
    nn = None


class FakeQwen(nn.Module):
    """Minimal Qwen3.5 stand-in: embed tokens, linear head, hidden states."""

    def __init__(self, vocab: int = 50, dim: int = 8) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=True):
        hidden = torch.relu(inputs_embeds)
        return SimpleNamespace(hidden_states=[hidden, hidden])


@unittest.skipIf(torch is None, "PyTorch is required for Qwen wrapper tests")
class QwenMoshiLMTest(unittest.TestCase):
    def _build(self, vocab: int = 50, dim: int = 8, card: int = 8, n_q: int = 4):
        from personaplex_finetuning.qwen_lm import QwenMoshiLM, ZeroEmbedding

        depformer_dim = dim // 2
        model = QwenMoshiLM(
            FakeQwen(vocab=vocab, dim=dim),
            card=card,
            n_q=n_q,
            dep_q=n_q,
            dim=dim,
            depformer_dim=depformer_dim,
            delays=[0, 0, 1, 1, 1],
            text_padding_token_id=0,
            end_of_text_padding_id=0,
            text_initial_token_id=0,
            depformer=nn.Linear(depformer_dim, depformer_dim),
            depformer_in=nn.ModuleList([nn.Linear(dim, depformer_dim, bias=False) for _ in range(n_q)]),
            linears=nn.ModuleList([nn.Linear(depformer_dim, card, bias=False) for _ in range(n_q)]),
            depformer_emb=nn.ModuleList(
                [ZeroEmbedding(card + 1, depformer_dim) for _ in range(n_q - 1)]
            ),
            dtype=torch.float32,
        )
        return model

    def test_forward_train_output_shapes(self) -> None:
        model = self._build()
        T = 6
        codes = torch.randint(0, 8, (1, 5, T))
        codes[:, 0] = torch.randint(0, 50, (1, T))  # text stream uses the Qwen vocab

        output = model(codes)

        self.assertEqual(tuple(output.logits.shape), (1, 4, T, 8))
        self.assertEqual(tuple(output.mask.shape), (1, 4, T))
        self.assertEqual(tuple(output.text_logits.shape), (1, 1, T, 50))
        self.assertEqual(tuple(output.text_mask.shape), (1, 1, T))

    def test_delay_positions_are_masked(self) -> None:
        model = self._build()
        T = 6
        codes = torch.randint(0, 8, (1, 5, T))
        codes[:, 0] = 1

        output = model(codes)
        # audio delays are [0, 1, 1, 1]: the last frame of streams 2..4 is invalid
        self.assertTrue(bool(output.mask[0, 0, -1]))
        self.assertFalse(bool(output.mask[0, 1, -1]))
        self.assertFalse(bool(output.mask[0, 2, -1]))
        self.assertFalse(bool(output.mask[0, 3, -1]))
        # the invalid positions carry NaN logits, never an arbitrary value
        self.assertTrue(torch.isnan(output.logits[0, 1, -1]).all())

    def test_zero_tokens_mask_targets(self) -> None:
        model = self._build()
        T = 6
        codes = torch.randint(0, 8, (1, 5, T))
        codes[:, 0] = 1
        codes[0, 1, :] = -1  # agent semantic codebook entirely absent

        output = model(codes)
        self.assertFalse(bool(output.mask[0, 0].any()))

    def test_gradients_flow_into_interface(self) -> None:
        model = self._build()
        T = 4
        codes = torch.randint(0, 8, (1, 5, T))
        codes[:, 0] = 1

        output = model(codes)
        (output.logits.sum() + output.text_logits.sum()).backward()

        self.assertIsNotNone(model.text_depth_adapter.weight.grad)
        self.assertIsNotNone(model.emb[0].weight.grad)
        self.assertIsNotNone(model.linears[0].weight.grad)

    def test_forward_rejects_wrong_stream_count(self) -> None:
        model = self._build()
        with self.assertRaises(ValueError):
            model(torch.randint(0, 8, (1, 3, 4)))


if __name__ == "__main__":
    unittest.main()
