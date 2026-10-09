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
            # (B,1) broadcasts into the (B,delay) head for any batch size.
            line[:, :delay] = padding[:, k : k + 1]
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
            # NaN only exists for floating tensors; int streams keep their
            # rolled values and rely on the cleared mask tail.
            if tensor.is_floating_point():
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


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-free RMS normalization (no learned gain) in float32."""
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)


class StreamGains(nn.Module):
    """Combine text, agent-audio and user-audio embeddings at a learned balance.

    Each audio group sum is RMS-normalized, so its weight in the backbone input
    is set by one learned gain instead of drifting with the embedding tables
    during training. Gains start at the RMS of a text embedding, putting both
    audio groups level with the text stream.
    """

    def __init__(self, text_rms: float) -> None:
        super().__init__()
        self.gains = nn.Parameter(torch.tensor([1.0, text_rms, text_rms], dtype=torch.float32))

    def forward(self, text: torch.Tensor, agent: torch.Tensor, user: torch.Tensor) -> torch.Tensor:
        text_gain, agent_gain, user_gain = self.gains
        mixed = text_gain * text.float() + agent_gain * _rms(agent) + user_gain * _rms(user)
        return mixed.to(text.dtype)


class HiddenLayerMix(nn.Module):
    """Softmax-weighted mix of every backbone hidden state (embeddings + layers).

    The last layer of a text LM specializes in next-token prediction; earlier
    layers keep more acoustic and positional detail. Intermediate states are
    raw residual streams, so each passes through the backbone's own final norm
    (the last state already has it) before mixing. Starts as a uniform average.
    """

    def __init__(self, num_states: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_states, dtype=torch.float32))

    def forward(self, hidden_states, final_norm: nn.Module) -> torch.Tensor:
        if len(hidden_states) != self.weights.numel():
            raise RuntimeError(
                f"backbone returned {len(hidden_states)} hidden states, layer mix expects {self.weights.numel()}"
            )
        probs = torch.softmax(self.weights, dim=0)
        last = len(hidden_states) - 1
        mixed = None
        for index, state in enumerate(hidden_states):
            normed = state if index == last else final_norm(state)
            term = probs[index] * normed.float()
            mixed = term if mixed is None else mixed + term
        return mixed.to(hidden_states[-1].dtype)


def _pca_init(table: torch.Tensor, dim: int) -> torch.Tensor:
    """Project a pretrained embedding table onto its top ``dim`` principal axes.

    Keeps the relative geometry between codes (which codes sound alike) while
    changing the width, then rescales to unit per-element RMS.
    """
    weight = table.detach().float()
    centered = weight - weight.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(dim, vh.shape[0])
    projected = centered @ vh[:rank].T
    out = torch.zeros(weight.shape[0], dim, device=weight.device)
    out[:, :rank] = projected
    return out / out.pow(2).mean().sqrt().clamp_min(1e-12)


class QwenMoshiLM(nn.Module):
    """PersonaPlex-style 17-stream model with a frozen Qwen backbone.

    Streams follow the PersonaPlex layout: index 0 is agent text, 1..8 are the
    agent audio codebooks, 9..16 are the user audio codebooks. The Qwen decoder
    consumes the summed embeddings; its hidden state is mapped into the Helium
    context space (``backbone_proj`` + the grafted Helium ``out_norm``) so the
    grafted ``depformer_in`` and depth transformer see the input they were
    trained on. As in the Cohere TinyAya+Moshi recipe, the agent semantic
    codebook (CB0) is predicted only by ``cb0_head`` on the Qwen hidden state;
    the depth transformer refines the remaining codebooks conditioned on it.
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
        out_norm: Optional[nn.Module] = None,
        context_dim: Optional[int] = None,
        audio_emb_init: Optional[list[torch.Tensor]] = None,
        num_hidden_states: Optional[int] = None,
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
        # Backbone feature of the newest frame; forward_depformer step 0 turns
        # it into CB0 logits once the frame's text token is known.
        self._last_cb0_hidden: Optional[torch.Tensor] = None

        factory_kwargs = {"device": device, "dtype": dtype}
        self.emb = nn.ModuleList(
            [ZeroEmbedding(card + 1, dim, zero_idx=self.zero_token_id, **factory_kwargs) for _ in range(n_q)]
        )
        if audio_emb_init is not None:
            # Reuse the code geometry Helium learned from ~7M hours of audio.
            if len(audio_emb_init) != n_q:
                raise ValueError(f"audio_emb_init needs {n_q} tables, got {len(audio_emb_init)}")
            with torch.no_grad():
                for table, source in zip(self.emb, audio_emb_init):
                    if source.shape[0] != card + 1:
                        raise ValueError(f"audio_emb_init table has {source.shape[0]} rows, expected {card + 1}")
                    table.weight.copy_(_pca_init(source.to(table.weight.device), dim).to(table.weight.dtype))
        # Audio groups are RMS-normalized in embed_codes, so table scale does not
        # set their weight; StreamGains does, starting level with a text token.
        # Strided rows estimate the RMS without a full fp32 copy of the table.
        text_rms = float(self._text_embedding().weight.detach()[::16].float().pow(2).mean().sqrt())
        self.stream_gains = StreamGains(text_rms).to(device)

        if num_hidden_states is None:
            config = getattr(qwen, "config", None)
            layers = getattr(config, "num_hidden_layers", None)
            if layers is None:
                layers = getattr(getattr(config, "text_config", None), "num_hidden_layers", None)
            if layers is None:
                raise ValueError("cannot infer the backbone layer count; pass num_hidden_states")
            num_hidden_states = int(layers) + 1  # embeddings + every decoder layer
        self.layer_mix = HiddenLayerMix(num_hidden_states).to(device)

        if depformer_in is not None:
            context_dim = int(depformer_in[0].in_features)
        elif context_dim is None:
            context_dim = dim
        self.context_dim = context_dim
        if depformer_in is None:
            depformer_in = nn.ModuleList(
                [nn.Linear(context_dim, depformer_dim, bias=False, **factory_kwargs) for _ in range(dep_q)]
            )
        self.depformer_in = depformer_in
        # Orthogonal init keeps the projection well conditioned at any width.
        proj = nn.Linear(dim, context_dim, bias=False, device=device, dtype=torch.float32)
        nn.init.orthogonal_(proj.weight)
        self.backbone_proj = proj.to(dtype)
        if out_norm is None:
            out_norm = nn.RMSNorm(context_dim, **factory_kwargs)
        self.out_norm = out_norm
        self.cb0_head = nn.Linear(dim, card, bias=False, **factory_kwargs)
        # Conditions CB0 on the frame's text token, as the native depformer step 0
        # did. Zero init: training starts from the unconditioned head.
        self.cb0_text_adapter = nn.Linear(dim, dim, bias=False, **factory_kwargs)
        nn.init.zeros_(self.cb0_text_adapter.weight)
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
        """Combine the 17 streams: text + normalized agent sum + normalized user sum."""
        B, K, S = sequence.shape
        if K != self.num_codebooks:
            raise ValueError(f"Sequence shape {sequence.shape} must match the number of codebooks.")
        agent_codebooks = self.num_audio_codebooks // 2
        groups = []
        for start, end in ((0, agent_codebooks), (agent_codebooks, self.num_audio_codebooks)):
            total = None
            for cb_index in range(start, end):
                audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
                total = audio_emb if total is None else total + audio_emb
            groups.append(total)
        return self.stream_gains(self._embed_text(sequence[:, 0]), *groups)

    def _decoder(self) -> nn.Module:
        """The Qwen decoder stack (without ``lm_head``)."""
        if hasattr(self.qwen, "get_decoder"):
            return self.qwen.get_decoder()
        return self.qwen.model

    def _final_norm(self) -> nn.Module:
        """The backbone's own output norm, reused to scale intermediate states."""
        norm = getattr(self._decoder(), "norm", None)
        return norm if norm is not None else nn.Identity()

    def _backbone(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen Qwen decoder; return the audio feature and text logits.

        Text logits come from the last layer through ``lm_head``; the audio
        feature (for ``cb0_head`` and the depth context) is the learned mix of
        every hidden state.

        When ``start_streaming_caches`` was called, successive single-frame calls
        reuse the Hugging Face KV cache so the model keeps dialogue context.
        """
        kwargs = {"output_hidden_states": True, "return_dict": True}
        if self._cache_streaming:
            if self._hf_cache is not None:
                kwargs["past_key_values"] = self._hf_cache
            kwargs["use_cache"] = True
        # Call the decoder directly: its last_hidden_state is always the final
        # normalized output, whatever a transformers version puts last in
        # ``hidden_states``.
        outputs = self._decoder()(inputs_embeds=input, **kwargs)
        if self._cache_streaming:
            self._hf_cache = outputs.past_key_values
        last = outputs.last_hidden_state
        text_logits = self.qwen.lm_head(last)
        states = (*outputs.hidden_states[:-1], last)
        hidden = self.layer_mix(states, self._final_norm())
        return hidden, text_logits[:, None]

    def cb0_logits(self, hidden: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Agent CB0 logits from the backbone feature and the frame's text token."""
        return self.cb0_head(hidden + self.cb0_text_adapter(self._embed_text(text_tokens)))

    def depth_context(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map the Qwen hidden state into the normalized Helium context space."""
        return self.out_norm(self.backbone_proj(hidden))

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LMGen entry point: depth context (Helium space) and text logits.

        LMGen only forwards the depth context to the depformer, so the newest
        frame's feature is kept for ``forward_depformer`` step 0, which also
        receives that frame's text token.
        """
        hidden, text_logits = self._backbone(input)
        self._last_cb0_hidden = hidden[:, -1:]
        return self.depth_context(hidden), text_logits

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

    def forward_depformer_training(
        self, sequence: torch.Tensor, transformer_out: torch.Tensor, cb0_logits: torch.Tensor
    ) -> torch.Tensor:
        """Run the depth transformer over all codebooks at once (training mode).

        Step 0 still runs (later steps attend to it), but its prediction is
        ``cb0_logits`` from the backbone head instead of ``linears[0]``.
        """
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
        all_logits = [cb0_logits]
        for cb_index in range(1, Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack([step.to(cb0_logits.dtype) for step in all_logits], 1)
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
        # Always run the depformer step: later codebooks attend to its state.
        dep_output = self.depformer(depformer_input)
        if depformer_cb_index == 0:
            if self._last_cb0_hidden is None:
                raise RuntimeError("forward_embeddings must run before depformer step 0 (CB0 comes from cb0_head)")
            logits = self.cb0_logits(self._last_cb0_hidden, sequence[:, 0])
        else:
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

        hidden, text_logits = self._backbone(self.embed_codes(delayed_codes[:, :, :-1]))
        targets_in = delayed_codes[:, :, 1:]  # text has no delay: targets_in[:, 0] is the frame's text
        logits = self.forward_depformer_training(
            targets_in, self.depth_context(hidden), self.cb0_logits(hidden, targets_in[:, 0])
        )

        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset : self.audio_offset + self.dep_q], logits, fill_value=float("nan")
        )
        logits_mask &= codes[:, self.audio_offset : self.audio_offset + self.dep_q] != self.zero_token_id
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float("nan"))
        text_logits_mask &= codes[:, :1] != self.zero_token_id
        return QwenLMOutput(logits, logits_mask, text_logits, text_logits_mask)

    def forward(self, codes: torch.Tensor) -> QwenLMOutput:
        return self.forward_train(codes)
