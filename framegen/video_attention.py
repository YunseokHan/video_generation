from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as torch_f

try:
    from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
    from diffusers.models.attention_processor import Attention
except ImportError:  # pragma: no cover - import-time guard for minimal test envs.
    BasicTransformerBlock = None
    Attention = None
    _chunked_feed_forward = None


@dataclass
class VideoAttentionAdapterConfig:
    enabled: bool = False
    active: bool = True
    train: bool = True
    placement: str = "all"
    use_temporal_self_attention: bool = True
    use_temporal_cross_attention: bool = True
    use_temporal_ffn: bool = True
    temporal_ffn_rank: int = 0
    temporal_ffn_kernel_size: int = 3
    temporal_ffn_gamma_init: float = 0.0
    include_prompt_tokens: bool = True
    # Persistent anchor conditioning (Agenda A1 v1).
    anchor_enabled: bool = False
    anchor_mode: str = "spatial"
    anchor_latent_channels: int = 4
    anchor_projector_hidden: int = 128
    anchor_gate_init: float = 0.0
    anchor_gate_per_channel: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "VideoAttentionAdapterConfig":
        values = dict(config or {})
        anchor = dict(values.get("anchor_conditioning") or {})
        return cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            active=bool(values.get("active", cls.active)),
            train=bool(values.get("train", cls.train)),
            placement=str(values.get("placement", cls.placement) or cls.placement),
            use_temporal_self_attention=bool(
                values.get("use_temporal_self_attention", cls.use_temporal_self_attention)
            ),
            use_temporal_cross_attention=bool(
                values.get("use_temporal_cross_attention", cls.use_temporal_cross_attention)
            ),
            use_temporal_ffn=bool(values.get("use_temporal_ffn", cls.use_temporal_ffn)),
            temporal_ffn_rank=int(values.get("temporal_ffn_rank", cls.temporal_ffn_rank)),
            temporal_ffn_kernel_size=int(
                values.get("temporal_ffn_kernel_size", cls.temporal_ffn_kernel_size)
            ),
            temporal_ffn_gamma_init=float(
                values.get("temporal_ffn_gamma_init", cls.temporal_ffn_gamma_init)
            ),
            include_prompt_tokens=bool(values.get("include_prompt_tokens", cls.include_prompt_tokens)),
            anchor_enabled=bool(anchor.get("enabled", cls.anchor_enabled)),
            anchor_mode=str(anchor.get("mode", cls.anchor_mode) or cls.anchor_mode),
            anchor_latent_channels=int(anchor.get("latent_channels", cls.anchor_latent_channels)),
            anchor_projector_hidden=int(anchor.get("projector_hidden", cls.anchor_projector_hidden)),
            anchor_gate_init=float(anchor.get("gate_init", cls.anchor_gate_init)),
            anchor_gate_per_channel=bool(anchor.get("gate_per_channel", cls.anchor_gate_per_channel)),
        )


def resolve_unet_placement_sections(placement: str | None) -> set[str] | None:
    """Map a placement string to the set of top-level UNet sections to adapt.

    Returns None when all sections should be adapted (the default), or a subset
    of {"down", "mid", "up"}. Accepts forms like "all", "mid_up", "mid+up",
    "up", "down_mid_up".
    """
    text = str(placement or "all").strip().lower()
    if text in {"", "all", "downmidup", "down_mid_up", "down-mid-up"}:
        return None
    tokens = {tok for tok in text.replace("+", "_").replace("-", "_").split("_") if tok}
    sections = {tok for tok in tokens if tok in {"down", "mid", "up"}}
    if not sections:
        raise ValueError(
            f"Unrecognized placement {placement!r}; use 'all' or a combination of down/mid/up."
        )
    return sections


def _zero_attention_output(attention: nn.Module) -> None:
    with torch.no_grad():
        output = attention.to_out[0]
        output.weight.zero_()
        if output.bias is not None:
            output.bias.zero_()


class VideoBasicTransformerBlock(nn.Module):
    """BasicTransformerBlock-compatible adapter with temporal attention paths."""

    def __init__(
        self,
        base: nn.Module,
        use_temporal_self_attention: bool = True,
        use_temporal_cross_attention: bool = True,
        use_temporal_ffn: bool = True,
        temporal_ffn_rank: int = 0,
        temporal_ffn_kernel_size: int = 3,
        temporal_ffn_gamma_init: float = 0.0,
        include_prompt_tokens: bool = True,
        active: bool = True,
        use_anchor_conditioning: bool = False,
        anchor_latent_channels: int = 4,
        anchor_projector_hidden: int = 128,
        anchor_gate_init: float = 0.0,
        anchor_gate_per_channel: bool = False,
    ) -> None:
        super().__init__()
        if Attention is None:
            raise ImportError("diffusers is required to create video attention adapters.")

        self.norm_type = base.norm_type
        self.only_cross_attention = base.only_cross_attention
        self.use_ada_layer_norm = base.use_ada_layer_norm
        self.use_ada_layer_norm_zero = base.use_ada_layer_norm_zero
        self.use_ada_layer_norm_single = base.use_ada_layer_norm_single
        self.use_ada_layer_norm_continuous = base.use_ada_layer_norm_continuous
        self.use_layer_norm = base.use_layer_norm

        self.norm1 = base.norm1
        self.attn1 = base.attn1
        self.norm2 = base.norm2
        self.attn2 = base.attn2
        self.norm3 = base.norm3
        self.ff = base.ff
        self.pos_embed = base.pos_embed
        self.fuser = getattr(base, "fuser", None)
        self._chunk_size = getattr(base, "_chunk_size", None)
        self._chunk_dim = getattr(base, "_chunk_dim", 0)
        if hasattr(base, "scale_shift_table"):
            self.scale_shift_table = base.scale_shift_table

        dim = int(self.attn1.to_q.in_features)
        heads = int(self.attn1.heads)
        dim_head = int(self.attn1.inner_dim // heads)
        attention_bias = self.attn1.to_q.bias is not None
        attention_out_bias = self.attn1.to_out[0].bias is not None
        cross_attention_dim = int(self.attn2.to_k.in_features) if self.attn2 is not None else dim

        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_self_attn = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )
        self.temporal_cross_norm = nn.LayerNorm(dim)
        self.temporal_cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )
        _zero_attention_output(self.temporal_self_attn)
        _zero_attention_output(self.temporal_cross_attn)

        temporal_ffn_rank = int(temporal_ffn_rank)
        if temporal_ffn_rank <= 0:
            temporal_ffn_rank = max(dim // 4, 1)
        temporal_ffn_kernel_size = int(temporal_ffn_kernel_size)
        if temporal_ffn_kernel_size <= 0 or temporal_ffn_kernel_size % 2 == 0:
            raise ValueError("temporal_ffn_kernel_size must be a positive odd integer.")
        self.temporal_ffn_norm = nn.LayerNorm(dim)
        self.temporal_ffn_in = nn.Linear(dim, temporal_ffn_rank)
        self.temporal_ffn_conv = nn.Conv1d(
            temporal_ffn_rank,
            temporal_ffn_rank,
            kernel_size=temporal_ffn_kernel_size,
            padding=temporal_ffn_kernel_size // 2,
        )
        self.temporal_ffn_out = nn.Linear(temporal_ffn_rank, dim)
        self.temporal_ffn_gamma = nn.Parameter(torch.tensor(float(temporal_ffn_gamma_init)))
        nn.init.zeros_(self.temporal_ffn_out.weight)
        nn.init.zeros_(self.temporal_ffn_out.bias)

        self.use_anchor_conditioning = bool(use_anchor_conditioning)
        if self.use_anchor_conditioning:
            self.anchor_proj_in = nn.Conv2d(int(anchor_latent_channels), int(anchor_projector_hidden), kernel_size=1)
            self.anchor_proj_out = nn.Conv2d(int(anchor_projector_hidden), dim, kernel_size=1)
            self.anchor_norm = nn.LayerNorm(dim)
            self.anchor_attn = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=heads,
                dim_head=dim_head,
                bias=attention_bias,
                out_bias=attention_out_bias,
            )
            # NOTE: do NOT zero-init anchor_attn output here. Identity at init is
            # provided by the zero gate alone. If both the gate and the attn
            # output were zero, the gate's gradient (upstream * anchor_out) would
            # be zero and the branch could never open — a dead-branch deadlock.
            if bool(anchor_gate_per_channel):
                self.anchor_gate = nn.Parameter(torch.full((dim,), float(anchor_gate_init)))
            else:
                self.anchor_gate = nn.Parameter(torch.tensor(float(anchor_gate_init)))

        self.use_temporal_self_attention = bool(use_temporal_self_attention)
        self.use_temporal_cross_attention = bool(use_temporal_cross_attention)
        self.use_temporal_ffn = bool(use_temporal_ffn)
        self.include_prompt_tokens = bool(include_prompt_tokens)
        self.active = bool(active)
        self.video_num_frames: int | None = None
        self.video_frame_tokens: torch.Tensor | None = None
        self.video_anchor_latents: torch.Tensor | None = None

        # When a use_* flag is False the corresponding sub-module is still
        # instantiated (so state_dict layout stays compatible with checkpoints
        # trained under the default config), but its parameters are frozen so
        # DDP does not expect gradients for them and the trainable-param
        # counter excludes them.
        if not self.use_temporal_self_attention:
            for parameter in self.temporal_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in self.temporal_self_attn.parameters():
                parameter.requires_grad_(False)
        if not self.use_temporal_cross_attention:
            for parameter in self.temporal_cross_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in self.temporal_cross_attn.parameters():
                parameter.requires_grad_(False)
        if not self.use_temporal_ffn:
            for parameter in self.temporal_ffn_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in self.temporal_ffn_in.parameters():
                parameter.requires_grad_(False)
            for parameter in self.temporal_ffn_conv.parameters():
                parameter.requires_grad_(False)
            for parameter in self.temporal_ffn_out.parameters():
                parameter.requires_grad_(False)
            self.temporal_ffn_gamma.requires_grad_(False)

    @property
    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        # Only yield parameters from sub-modules whose use_* flag is on. This
        # keeps disabled sub-modules out of the train=True grad-enablement path
        # so DDP does not see them as trainable parameters.
        if self.use_temporal_self_attention:
            yield from self.temporal_norm.parameters()
            yield from self.temporal_self_attn.parameters()
        if self.use_temporal_cross_attention:
            yield from self.temporal_cross_norm.parameters()
            yield from self.temporal_cross_attn.parameters()
        if self.use_temporal_ffn:
            yield from self.temporal_ffn_norm.parameters()
            yield from self.temporal_ffn_in.parameters()
            yield from self.temporal_ffn_conv.parameters()
            yield from self.temporal_ffn_out.parameters()
            yield self.temporal_ffn_gamma
        if self.use_anchor_conditioning:
            yield from self.anchor_proj_in.parameters()
            yield from self.anchor_proj_out.parameters()
            yield from self.anchor_norm.parameters()
            yield from self.anchor_attn.parameters()
            yield self.anchor_gate

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int = 0) -> None:
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def set_video_context(
        self,
        num_frames: int | None,
        frame_tokens: torch.Tensor | None = None,
        anchor_latents: torch.Tensor | None = None,
    ) -> None:
        self.video_num_frames = int(num_frames) if num_frames is not None else None
        self.video_frame_tokens = frame_tokens
        self.video_anchor_latents = anchor_latents

    def clear_video_context(self) -> None:
        self.video_num_frames = None
        self.video_frame_tokens = None
        self.video_anchor_latents = None

    def _to_temporal_tokens(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        num_frames = int(self.video_num_frames or 1)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        batch_frames, spatial_tokens, channels = hidden_states.shape
        if batch_frames % num_frames != 0:
            raise ValueError(
                "Flattened frame batch is not divisible by num_frames: "
                f"{batch_frames} vs {num_frames}."
            )
        batch_size = batch_frames // num_frames
        temporal = hidden_states.reshape(batch_size, num_frames, spatial_tokens, channels)
        temporal = temporal.permute(0, 2, 1, 3).contiguous()
        temporal = temporal.reshape(batch_size * spatial_tokens, num_frames, channels)
        return temporal, batch_size, spatial_tokens

    def _from_temporal_tokens(
        self,
        temporal: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        num_frames = int(self.video_num_frames or 1)
        channels = temporal.shape[-1]
        hidden_states = temporal.reshape(batch_size, spatial_tokens, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        return hidden_states.reshape(batch_size * num_frames, spatial_tokens, channels)

    def _prompt_context(
        self,
        encoder_hidden_states: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if encoder_hidden_states is None or not self.include_prompt_tokens:
            return None
        num_frames = int(self.video_num_frames or 1)
        if encoder_hidden_states.shape[0] % num_frames != 0:
            raise ValueError(
                "Prompt batch is not divisible by num_frames: "
                f"{encoder_hidden_states.shape[0]} vs {num_frames}."
            )
        prompt_batch = encoder_hidden_states.shape[0] // num_frames
        if prompt_batch != batch_size:
            raise ValueError(f"Prompt batch {prompt_batch} does not match latent batch {batch_size}.")
        prompt = encoder_hidden_states.reshape(
            batch_size,
            num_frames,
            encoder_hidden_states.shape[1],
            encoder_hidden_states.shape[2],
        )
        return prompt[:, 0]

    def _frame_context(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.video_frame_tokens is None:
            return None
        num_frames = int(self.video_num_frames or 1)
        frame_tokens = self.video_frame_tokens.to(device=device, dtype=dtype)
        if frame_tokens.ndim == 3:
            if frame_tokens.shape[0] == batch_size * num_frames:
                frame_tokens = frame_tokens.reshape(
                    batch_size,
                    num_frames,
                    frame_tokens.shape[1],
                    frame_tokens.shape[2],
                )
            elif frame_tokens.shape[0] == batch_size:
                frame_tokens = frame_tokens[:, :, None, :]
            elif frame_tokens.shape[0] == num_frames and batch_size % 1 == 0:
                frame_tokens = frame_tokens[None, :, :, :].expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(
                    "Unsupported frame token shape for temporal cross-attention: "
                    f"{tuple(frame_tokens.shape)}."
                )
        elif frame_tokens.ndim == 4:
            if frame_tokens.shape[1] != num_frames:
                raise ValueError(
                    "Frame tokens must have shape [B, F, T, D], got "
                    f"{tuple(frame_tokens.shape)} for B={batch_size}, F={num_frames}."
                )
            if frame_tokens.shape[0] != batch_size:
                if batch_size % frame_tokens.shape[0] != 0:
                    raise ValueError(
                        "Frame token batch must match or divide hidden batch, got "
                        f"{frame_tokens.shape[0]} vs {batch_size}."
                    )
                frame_tokens = frame_tokens.repeat(batch_size // frame_tokens.shape[0], 1, 1, 1)
        else:
            raise ValueError(
                "Frame tokens must be [B*F, T, D], [B, F, D], or [B, F, T, D], got "
                f"{tuple(frame_tokens.shape)}."
            )
        return frame_tokens.reshape(
            batch_size,
            num_frames * frame_tokens.shape[-2],
            frame_tokens.shape[-1],
        )

    def _temporal_self_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.active or not self.use_temporal_self_attention:
            return hidden_states
        norm_hidden_states = self.temporal_norm(hidden_states)
        temporal, batch_size, spatial_tokens = self._to_temporal_tokens(norm_hidden_states)
        temporal_output = self.temporal_self_attn(temporal)
        hidden_output = self._from_temporal_tokens(temporal_output, batch_size, spatial_tokens)
        return hidden_states + hidden_output

    def _temporal_anchor_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Spatial-aligned anchor conditioning (Agenda A1 v1).

        At each spatial site the F frame tokens attend to a single anchor token
        derived from the clean anchor latent projected to this block's grid.
        Gated by a zero-init parameter so an enabled-but-untrained branch is an
        exact identity (checkpoint-compatible).
        """
        if not self.active or not self.use_anchor_conditioning:
            return hidden_states
        if self.video_anchor_latents is None:
            return hidden_states
        norm_hidden_states = self.anchor_norm(hidden_states)
        temporal, batch_size, spatial_tokens = self._to_temporal_tokens(norm_hidden_states)
        side = int(round(spatial_tokens ** 0.5))
        if side * side != spatial_tokens:
            # Non-square spatial grid: spatial alignment is undefined, skip.
            return hidden_states
        anchor = self.video_anchor_latents.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if anchor.ndim != 4:
            return hidden_states
        if anchor.shape[0] != batch_size:
            if batch_size % anchor.shape[0] == 0:
                anchor = anchor.repeat_interleave(batch_size // anchor.shape[0], dim=0)
            else:
                return hidden_states
        anchor = torch_f.interpolate(anchor, size=(side, side), mode="bilinear", align_corners=False)
        anchor = self.anchor_proj_out(torch_f.silu(self.anchor_proj_in(anchor)))
        anchor_tokens = anchor.flatten(2).transpose(1, 2)  # [B, S, C]
        kv = anchor_tokens.reshape(batch_size * spatial_tokens, 1, anchor_tokens.shape[-1])
        anchor_output = self.anchor_attn(temporal, encoder_hidden_states=kv)
        hidden_output = self._from_temporal_tokens(anchor_output, batch_size, spatial_tokens)
        gate = self.anchor_gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if gate.ndim == 1:
            gate = gate.view(1, 1, -1)
        return hidden_states + gate * hidden_output

    def _temporal_cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.active or not self.use_temporal_cross_attention:
            return hidden_states
        norm_hidden_states = self.temporal_cross_norm(hidden_states)
        temporal, batch_size, spatial_tokens = self._to_temporal_tokens(norm_hidden_states)
        prompt_context = self._prompt_context(encoder_hidden_states, batch_size)
        frame_context = self._frame_context(
            batch_size=batch_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        contexts = [context for context in [prompt_context, frame_context] if context is not None]
        if not contexts:
            return hidden_states
        context = torch.cat(contexts, dim=1) if len(contexts) > 1 else contexts[0]
        temporal_queries = temporal.reshape(batch_size, spatial_tokens, -1, temporal.shape[-1])
        temporal_queries = temporal_queries.reshape(batch_size, spatial_tokens * temporal.shape[1], temporal.shape[-1])
        temporal_output = self.temporal_cross_attn(
            temporal_queries,
            encoder_hidden_states=context,
        )
        temporal_output = temporal_output.reshape(batch_size, spatial_tokens, temporal.shape[1], temporal.shape[-1])
        temporal_output = temporal_output.reshape(batch_size * spatial_tokens, temporal.shape[1], temporal.shape[-1])
        hidden_output = self._from_temporal_tokens(temporal_output, batch_size, spatial_tokens)
        return hidden_states + hidden_output

    def _temporal_feed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.active or not self.use_temporal_ffn:
            return hidden_states
        norm_hidden_states = self.temporal_ffn_norm(hidden_states)
        temporal, batch_size, spatial_tokens = self._to_temporal_tokens(norm_hidden_states)
        temporal = self.temporal_ffn_in(temporal)
        temporal = self.temporal_ffn_conv(temporal.transpose(1, 2)).transpose(1, 2)
        temporal = torch_f.silu(temporal)
        temporal = self.temporal_ffn_out(temporal)
        hidden_output = self._from_temporal_tokens(temporal, batch_size, spatial_tokens)
        gamma = self.temporal_ffn_gamma.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + gamma * hidden_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] = None,
        class_labels: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        hidden_states = self._temporal_self_attention(hidden_states)
        hidden_states = self._temporal_anchor_attention(hidden_states)

        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        hidden_states = self._temporal_cross_attention(hidden_states, encoder_hidden_states)

        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        hidden_states = self._temporal_feed_forward(hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


def _target_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _placement_roots(module: nn.Module, sections: set[str] | None) -> list[nn.Module]:
    """Return the subtrees to inject adapters into for a given placement.

    None means the whole module (default). A subset of {down, mid, up} restricts
    injection to those top-level UNet sections; sections absent on the module
    (e.g. a VAE decoder has no down_blocks) are simply skipped.
    """
    if sections is None:
        return [module]
    roots: list[nn.Module] = []
    if "down" in sections and getattr(module, "down_blocks", None) is not None:
        roots.append(module.down_blocks)
    if "mid" in sections and getattr(module, "mid_block", None) is not None:
        roots.append(module.mid_block)
    if "up" in sections and getattr(module, "up_blocks", None) is not None:
        roots.append(module.up_blocks)
    return roots


def iter_video_attention_blocks(module: nn.Module) -> Iterator[VideoBasicTransformerBlock]:
    module = _target_module(module)
    for child in module.modules():
        if isinstance(child, VideoBasicTransformerBlock):
            yield child


def inject_video_attention_adapters(
    module: nn.Module,
    config: VideoAttentionAdapterConfig | dict[str, Any] | None,
) -> int:
    adapter_config = (
        config
        if isinstance(config, VideoAttentionAdapterConfig)
        else VideoAttentionAdapterConfig.from_config(config)
    )
    if not adapter_config.enabled:
        return 0
    if BasicTransformerBlock is None:
        raise ImportError("diffusers is required to inject video attention adapters.")

    module = _target_module(module)
    replaced = 0

    def replace_children(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, VideoBasicTransformerBlock):
                continue
            if isinstance(child, BasicTransformerBlock):
                setattr(
                    parent,
                    name,
                    VideoBasicTransformerBlock(
                        child,
                        use_temporal_self_attention=adapter_config.use_temporal_self_attention,
                        use_temporal_cross_attention=adapter_config.use_temporal_cross_attention,
                        use_temporal_ffn=adapter_config.use_temporal_ffn,
                        temporal_ffn_rank=adapter_config.temporal_ffn_rank,
                        temporal_ffn_kernel_size=adapter_config.temporal_ffn_kernel_size,
                        temporal_ffn_gamma_init=adapter_config.temporal_ffn_gamma_init,
                        include_prompt_tokens=adapter_config.include_prompt_tokens,
                        active=adapter_config.active,
                        use_anchor_conditioning=adapter_config.anchor_enabled,
                        anchor_latent_channels=adapter_config.anchor_latent_channels,
                        anchor_projector_hidden=adapter_config.anchor_projector_hidden,
                        anchor_gate_init=adapter_config.anchor_gate_init,
                        anchor_gate_per_channel=adapter_config.anchor_gate_per_channel,
                    ),
                )
                replaced += 1
            else:
                replace_children(child)

    sections = resolve_unet_placement_sections(adapter_config.placement)
    for root in _placement_roots(module, sections):
        replace_children(root)
    return replaced


def set_video_attention_adapters_active(module: nn.Module, active: bool) -> None:
    for block in iter_video_attention_blocks(module):
        block.active = bool(active)


def set_video_attention_adapter_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for block in iter_video_attention_blocks(module):
        for parameter in block.adapter_parameters:
            parameter.requires_grad_(requires_grad)


def set_video_attention_context(
    module: nn.Module,
    num_frames: int | None,
    frame_tokens: torch.Tensor | None = None,
    anchor_latents: torch.Tensor | None = None,
) -> None:
    for block in iter_video_attention_blocks(module):
        block.set_video_context(
            num_frames=num_frames,
            frame_tokens=frame_tokens,
            anchor_latents=anchor_latents,
        )


def clear_video_attention_context(module: nn.Module) -> None:
    for block in iter_video_attention_blocks(module):
        block.clear_video_context()


def sync_video_attention_adapter_device_dtype(module: nn.Module) -> None:
    module = _target_module(module)
    try:
        reference = next(module.parameters())
    except StopIteration:
        return
    for block in iter_video_attention_blocks(module):
        block.temporal_norm.to(device=reference.device, dtype=reference.dtype)
        block.temporal_self_attn.to(device=reference.device, dtype=reference.dtype)
        block.temporal_cross_norm.to(device=reference.device, dtype=reference.dtype)
        block.temporal_cross_attn.to(device=reference.device, dtype=reference.dtype)
        block.temporal_ffn_norm.to(device=reference.device, dtype=reference.dtype)
        block.temporal_ffn_in.to(device=reference.device, dtype=reference.dtype)
        block.temporal_ffn_conv.to(device=reference.device, dtype=reference.dtype)
        block.temporal_ffn_out.to(device=reference.device, dtype=reference.dtype)
        block.temporal_ffn_gamma.data = block.temporal_ffn_gamma.data.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if getattr(block, "use_anchor_conditioning", False):
            block.anchor_proj_in.to(device=reference.device, dtype=reference.dtype)
            block.anchor_proj_out.to(device=reference.device, dtype=reference.dtype)
            block.anchor_norm.to(device=reference.device, dtype=reference.dtype)
            block.anchor_attn.to(device=reference.device, dtype=reference.dtype)
            block.anchor_gate.data = block.anchor_gate.data.to(
                device=reference.device,
                dtype=reference.dtype,
            )


def video_attention_adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    module = _target_module(module)
    adapter_names = (
        ".temporal_norm.",
        ".temporal_self_attn.",
        ".temporal_cross_norm.",
        ".temporal_cross_attn.",
        ".temporal_ffn_norm.",
        ".temporal_ffn_in.",
        ".temporal_ffn_conv.",
        ".temporal_ffn_out.",
        ".anchor_proj_in.",
        ".anchor_proj_out.",
        ".anchor_norm.",
        ".anchor_attn.",
    )
    adapter_suffixes = (".temporal_ffn_gamma", ".anchor_gate")
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
        if any(name in key for name in adapter_names) or key.endswith(adapter_suffixes)
    }


def load_video_attention_adapter_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    module = _target_module(module)
    load_result = module.load_state_dict(state_dict, strict=False)
    if strict:
        missing_adapter_keys = [
            key
            for key in video_attention_adapter_state_dict(module)
            if key not in state_dict
        ]
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_adapter_keys or unexpected_keys:
            raise RuntimeError(
                "Video attention adapter state mismatch: "
                f"missing={missing_adapter_keys}, unexpected={unexpected_keys}"
            )
