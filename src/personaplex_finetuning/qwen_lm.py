"""Qwen backbone wrapper reproducing the PersonaPlex 17-stream training contract.

The Helium temporal transformer is replaced by a frozen Qwen3.5 decoder; Mimi
and the depth transformer are reused from the PersonaPlex checkpoint. Only the
audio-side interface and small adapters are trained. The module is
self-contained (torch only) so the delay/undelay helpers do not depend on the
vendored ``moshi`` package.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch
from torch import nn


@dataclasses.dataclass
class QwenLMOutput:
    """Mirrors ``moshi.models.lm.LMOutput`` for the shared training loss."""

    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


def _delay_sequence(delays: list[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """Shift each stream by its delay, filling the head with ``padding``."""
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: list[int], tensor: torch.Tensor, fill_value: float = float("nan")):
    """Reverse ``_delay_sequence``; invalid tail positions are NaN + masked."""
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all(delay == 0 for delay in delays):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


class ZeroEmbedding(nn.Embedding):
    """Embedding whose ``zero_idx`` inputs produce exactly zero vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int, zero_idx: int = -1, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_zero = input == self.zero_idx
        safe = input.clamp(min=0)
        out = super().forward(safe)
        return torch.where(is_zero[..., None], torch.zeros_like(out), out)


class QwenMoshiLM(nn.Module):
    """PersonaPlex-style 17-stream model with a frozen Qwen backbone.

    Streams follow the PersonaPlex layout: index 0 is agent text, 1..8 are the
    agent audio codebooks, 9..16 are the user audio codebooks. The Qwen decoder
    consumes the summed embeddings and produces the hidden context feeding the
    depth transformer.
    """

    def __init__(
        self,
        qwen,
        *,
        card: int = 2048,
        n_q: int = 16,
        dep_q: int = 16,
        dim: int = 4096,
        depformer_dim: int = 1024,
        delays: Optional[list[int]] = None,
        text_padding_token_id: int,
        end_of_text_padding_id: int,
        text_initial_token_id: int,
        depformer: Optional[nn.Module] = None,
        depformer_in: Optional[nn.ModuleList] = None,
        linears: Optional[nn.ModuleList] = None,
        depformer_emb: Optional[nn.ModuleList] = None,
        depformer_weights_per_step_schedule: Optional[list[int]] = None,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if delays is None:
            delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        if len(delays) != n_q + 1:
            raise ValueError(f"delays must contain {n_q + 1} entries, got {len(delays)}")
        if dep_q > n_q:
            raise ValueError(f"dep_q={dep_q} must not exceed n_q={n_q}")
        self.qwen = qwen
        self.card = card
        self.n_q = n_q
        self.dep_q = dep_q
        self.dim = dim
        self.depformer_dim = depformer_dim
        self.delays = delays
        self.text_padding_token_id = text_padding_token_id
        self.end_of_text_padding_id = end_of_text_padding_id
        self.text_initial_token_id = text_initial_token_id
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        # Incremental KV-cache decoding state (LMGen feeds one frame per call).
        self._hf_cache = None
        self._cache_streaming = False

        factory_kwargs = {"device": device, "dtype": dtype}
        self.emb = nn.ModuleList(
            [ZeroEmbedding(card + 1, dim, zero_idx=self.zero_token_id, **factory_kwargs) for _ in range(n_q)]
        )
        if depformer_in is None:
            depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False, **factory_kwargs) for _ in range(dep_q)]
            )
        self.depformer_in = depformer_in
        if depformer_emb is None:
            depformer_emb = nn.ModuleList(
                [ZeroEmbedding(card + 1, depformer_dim, zero_idx=self.zero_token_id, **factory_kwargs) for _ in range(dep_q - 1)]
            )
        self.depformer_emb = depformer_emb
        if linears is None:
            linears = nn.ModuleList(
                [nn.Linear(depformer_dim, card, bias=False, **factory_kwargs) for _ in range(dep_q)]
            )
        self.linears = linears
        # Replaces the PersonaPlex ``depformer_text_emb`` embedding table: the
        # text vocabulary changed from 32000 (Helium) to the Qwen vocabulary,
        # so the depth token input is projected from the Qwen token embedding.
        self.text_depth_adapter = nn.Linear(dim, depformer_dim, bias=False, **factory_kwargs)
        if depformer is None:
            raise ValueError("depformer must be provided (reused from the PersonaPlex checkpoint)")
        self.depformer = depformer

    # ------------------------------------------------------------------ #
    # PersonaPlex-compatible properties used by the training pipeline.    #
    # ------------------------------------------------------------------ #
    @property
    def zero_token_id(self) -> int:
        """Special input token contributing exactly zero embedding."""
        return -1

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def text_card(self) -> int:
        """Cardinality of the Qwen text vocabulary (used by LMGen)."""
        return int(self.qwen.config.vocab_size)

    @property
    def ungenerated_token_id(self) -> int:
        """Marker for tokens LMGen should sample rather than read from the prompt."""
        return -2

    def _get_initial_token(self) -> torch.Tensor:
        """Return the initial tokens fed to the very first timestep, [1, K, 1]."""
        device = self.device
        zero = torch.full([1, 1, 1], self.zero_token_id, device=device, dtype=torch.long)
        audio = torch.full_like(zero, self.initial_token_id).expand(-1, self.num_audio_codebooks, -1)
        text = torch.full_like(zero, self.text_initial_token_id)
        return torch.cat([text, audio], dim=1)

    def _text_embedding(self) -> nn.Module:
        if hasattr(self.qwen, "get_input_embeddings"):
            return self.qwen.get_input_embeddings()
        return self.qwen.model.embed_tokens

    def _embed_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed text tokens; the zero token contributes exactly zero."""
        embedding = self._text_embedding()
        safe = tokens.clamp(min=0)
        out = embedding(safe)
        is_zero = tokens == self.zero_token_id
        return torch.where(is_zero[..., None], torch.zeros_like(out), out)

    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sum the embeddings of all 17 streams, as the Helium model did."""
        B, K, S = sequence.shape
        if K != self.num_codebooks:
            raise ValueError(f"Sequence shape {sequence.shape} must match the number of codebooks.")
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self._embed_text(sequence[:, 0])
        return text_emb if input_ is None else input_ + text_emb

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen Qwen decoder; return hidden context and text logits.

        When ``start_streaming_caches`` was called, successive single-frame calls
        reuse the Hugging Face KV cache so the model keeps dialogue context.
        """
        kwargs = {"output_hidden_states": True, "return_dict": True}
        if self._cache_streaming:
            if self._hf_cache is not None:
                kwargs["past_key_values"] = self._hf_cache
            kwargs["use_cache"] = True
        outputs = self.qwen(inputs_embeds=input, **kwargs)
        if self._cache_streaming:
            self._hf_cache = outputs.past_key_values
        hidden = outputs.hidden_states[-1]
        text_logits = self.qwen.lm_head(hidden)
        return hidden, text_logits[:, None]

    def start_streaming_caches(self) -> None:
        """Enable incremental KV-cache decoding for LMGen-style single-frame calls."""
        self._hf_cache = None
        self._cache_streaming = True

    def stop_streaming_caches(self) -> None:
        """Disable cache reuse and drop the cached attention state."""
        self._cache_streaming = False
        self._hf_cache = None

    def forward_codes(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed_codes(sequence))

    def forward_depformer_training(self, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        """Run the depth transformer over all codebooks at once (training mode)."""
        B, K, T = sequence.shape
        Ka = self.dep_q
        if K != self.num_codebooks:
            raise ValueError(f"Codebooks for Depformer training should be passed all at once, got {K,}.")
        depformer_inputs = []
        for cb_index in range(Ka):
            linear_index = cb_index
            if self.depformer_weights_per_step_schedule is not None:
                linear_index = self.depformer_weights_per_step_schedule[cb_index]
            transformer_in = self.depformer_in[linear_index](transformer_out)
            if cb_index == 0:
                token_in = self.text_depth_adapter(self._embed_text(sequence[:, 0]))
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        if logits.dim() != 4:  # [B, Ka, T, card]
            raise RuntimeError(f"unexpected depformer logits shape {logits.shape}")
        return logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming depformer step mirroring ``moshi.models.lm.LMModel.forward_depformer``."""
        B, K, S = sequence.shape
        if K != 1 or S != 1:
            raise ValueError(f"Depformer streaming expects one codebook and one step, got {K}, {S}")
        if transformer_out.shape[1] != 1:
            raise ValueError("transformer_out must be a single step")
        linear_index = depformer_cb_index
        if self.depformer_weights_per_step_schedule is not None:
            linear_index = self.depformer_weights_per_step_schedule[depformer_cb_index]
        depformer_input = self.depformer_in[linear_index](transformer_out)
        if depformer_cb_index == 0:
            last_token_input = self.text_depth_adapter(self._embed_text(sequence[:, 0]))
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        depformer_input = depformer_input + last_token_input
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        if logits.dim() != 4:  # [B, Ka, S, card]
            raise RuntimeError(f"unexpected depformer logits shape {logits.shape}")
        return logits

    def forward_train(self, codes: torch.Tensor) -> QwenLMOutput:
        """Reproduce the PersonaPlex training forward on the Qwen backbone."""
        B, K, T = codes.shape
        if K != self.num_codebooks:
            raise ValueError(f"Sequence shape {codes.shape} must match the number of codebooks.")
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        transformer_out, text_logits = self.forward_codes(delayed_codes[:, :, :-1])
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset : self.audio_offset + self.dep_q], logits, fill_value=float("nan")
        )
        logits_mask &= codes[:, self.audio_offset : self.audio_offset + self.dep_q] != self.zero_token_id
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float("nan"))
        text_logits_mask &= codes[:, :1] != self.zero_token_id
        return QwenLMOutput(logits, logits_mask, text_logits, text_logits_mask)

    def forward(self, codes: torch.Tensor) -> QwenLMOutput:
        return self.forward_train(codes)
