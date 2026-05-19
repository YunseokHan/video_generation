from __future__ import annotations

import torch
from torch import nn

FRAME_TOKEN_MODE_TEMPORAL_ONLY = "temporal_cross_attention_only"
FRAME_TOKEN_MODE_ADD_TO_TEXT = "add_to_text"
FRAME_TOKEN_MODE_CONCAT_TOKENS = "concat_tokens"
FRAME_TOKEN_MODE_NONE = "none"

FRAME_TOKEN_MODE_ALIASES = {
    "": FRAME_TOKEN_MODE_TEMPORAL_ONLY,
    "legacy": FRAME_TOKEN_MODE_TEMPORAL_ONLY,
    "temporal_only": FRAME_TOKEN_MODE_TEMPORAL_ONLY,
    "temporal_cross_attention": FRAME_TOKEN_MODE_TEMPORAL_ONLY,
    "temporal_cross_attention_only": FRAME_TOKEN_MODE_TEMPORAL_ONLY,
    "add": FRAME_TOKEN_MODE_ADD_TO_TEXT,
    "add_to_text": FRAME_TOKEN_MODE_ADD_TO_TEXT,
    "add_to_text_embeds": FRAME_TOKEN_MODE_ADD_TO_TEXT,
    "add_to_prompt_embeds": FRAME_TOKEN_MODE_ADD_TO_TEXT,
    "concat": FRAME_TOKEN_MODE_CONCAT_TOKENS,
    "concat_tokens": FRAME_TOKEN_MODE_CONCAT_TOKENS,
    "learnable_concat": FRAME_TOKEN_MODE_CONCAT_TOKENS,
    "learnable_token_concat": FRAME_TOKEN_MODE_CONCAT_TOKENS,
    "off": FRAME_TOKEN_MODE_NONE,
    "disabled": FRAME_TOKEN_MODE_NONE,
    "none": FRAME_TOKEN_MODE_NONE,
}


def normalize_frame_token_mode(mode: str | None) -> str:
    normalized = str(mode or FRAME_TOKEN_MODE_TEMPORAL_ONLY).lower()
    normalized = normalized.replace("-", "_")
    if normalized not in FRAME_TOKEN_MODE_ALIASES:
        supported = ", ".join(sorted(set(FRAME_TOKEN_MODE_ALIASES.values())))
        raise ValueError(f"Unsupported frame token embedding mode {mode!r}; expected one of: {supported}.")
    return FRAME_TOKEN_MODE_ALIASES[normalized]


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _flatten_frame_embeddings(
    frame_embeddings: torch.Tensor,
    batch_frames: int,
    num_frames: int,
) -> torch.Tensor:
    if frame_embeddings.ndim == 1:
        frame_embeddings = frame_embeddings.unsqueeze(-1)
    if frame_embeddings.ndim == 2:
        if frame_embeddings.shape[0] == batch_frames:
            return frame_embeddings
        if frame_embeddings.shape[0] == num_frames and batch_frames % num_frames == 0:
            batch_size = batch_frames // num_frames
            return frame_embeddings.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_frames, -1)
    if frame_embeddings.ndim == 3:
        if frame_embeddings.shape[0] * frame_embeddings.shape[1] == batch_frames:
            return frame_embeddings.reshape(batch_frames, frame_embeddings.shape[-1])
        if frame_embeddings.shape[1] == num_frames and batch_frames % num_frames == 0:
            batch_size = batch_frames // num_frames
            if frame_embeddings.shape[0] == 1 and batch_size > 1:
                frame_embeddings = frame_embeddings.expand(batch_size, -1, -1)
            if frame_embeddings.shape[0] == batch_size:
                return frame_embeddings.reshape(batch_frames, frame_embeddings.shape[-1])
    raise ValueError(
        "Frame embeddings must be [B*F, D], [F, D], or [B, F, D], got "
        f"{tuple(frame_embeddings.shape)} for batch_frames={batch_frames}, num_frames={num_frames}."
    )


def _flatten_frame_tokens(
    frame_tokens: torch.Tensor,
    batch_frames: int,
    num_frames: int,
) -> torch.Tensor:
    if frame_tokens.ndim == 2:
        frame_tokens = frame_tokens.unsqueeze(-2)
    if frame_tokens.ndim == 3:
        if frame_tokens.shape[0] == batch_frames:
            return frame_tokens
        if frame_tokens.shape[0] == num_frames and batch_frames % num_frames == 0:
            batch_size = batch_frames // num_frames
            return frame_tokens.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
                batch_frames,
                frame_tokens.shape[-2],
                frame_tokens.shape[-1],
            )
    if frame_tokens.ndim == 4:
        if frame_tokens.shape[0] * frame_tokens.shape[1] == batch_frames:
            return frame_tokens.reshape(batch_frames, frame_tokens.shape[-2], frame_tokens.shape[-1])
        if frame_tokens.shape[1] == num_frames and batch_frames % num_frames == 0:
            batch_size = batch_frames // num_frames
            if frame_tokens.shape[0] == 1 and batch_size > 1:
                frame_tokens = frame_tokens.expand(batch_size, -1, -1, -1)
            if frame_tokens.shape[0] == batch_size:
                return frame_tokens.reshape(batch_frames, frame_tokens.shape[-2], frame_tokens.shape[-1])
    raise ValueError(
        "Frame tokens must be [B*F, T, D], [F, T, D], or [B, F, T, D], got "
        f"{tuple(frame_tokens.shape)} for batch_frames={batch_frames}, num_frames={num_frames}."
    )


def _unflatten_prompt_tokens(prompt_embeds: torch.Tensor, num_frames: int) -> torch.Tensor:
    batch_frames, sequence_length, channels = prompt_embeds.shape
    if batch_frames % num_frames != 0:
        raise ValueError(
            "Prompt batch is not divisible by num_frames: "
            f"{batch_frames} vs {num_frames}."
        )
    return prompt_embeds.reshape(batch_frames // num_frames, num_frames, sequence_length, channels)


def apply_frame_token_conditioning(
    prompt_embeds: torch.Tensor,
    frame_embeddings: torch.Tensor | None,
    frame_tokens: torch.Tensor | None,
    num_frames: int,
    mode: str | None,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the configured frame-token ablation strategy to SDXL text tokens.

    Returns the prompt embeddings to feed into SDXL cross-attention and the
    frame-token bank to feed into the temporal cross-attention adapter.
    """

    token_mode = normalize_frame_token_mode(mode)
    if token_mode == FRAME_TOKEN_MODE_NONE:
        return prompt_embeds, None
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if prompt_embeds.ndim != 3:
        raise ValueError(f"Expected prompt_embeds [B*F, S, D], got {tuple(prompt_embeds.shape)}.")

    batch_frames, _, embedding_dim = prompt_embeds.shape
    scale = float(alpha)

    if token_mode == FRAME_TOKEN_MODE_TEMPORAL_ONLY:
        return prompt_embeds, frame_tokens

    if token_mode == FRAME_TOKEN_MODE_ADD_TO_TEXT:
        if frame_embeddings is None:
            raise ValueError("frame_embeddings are required for token_embedding_mode='add_to_text'.")
        flat_frame_embeddings = _flatten_frame_embeddings(
            frame_embeddings,
            batch_frames=batch_frames,
            num_frames=num_frames,
        ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if flat_frame_embeddings.shape[-1] != embedding_dim:
            raise ValueError(
                "Frame embedding dimension must match prompt embedding dimension for add_to_text: "
                f"{flat_frame_embeddings.shape[-1]} vs {embedding_dim}."
            )
        conditioned_prompt_embeds = prompt_embeds + scale * flat_frame_embeddings[:, None, :]
        temporal_tokens = _unflatten_prompt_tokens(conditioned_prompt_embeds, num_frames)
        return conditioned_prompt_embeds, temporal_tokens

    if token_mode == FRAME_TOKEN_MODE_CONCAT_TOKENS:
        if frame_tokens is None:
            raise ValueError("frame_tokens are required for token_embedding_mode='concat_tokens'.")
        flat_frame_tokens = _flatten_frame_tokens(
            frame_tokens,
            batch_frames=batch_frames,
            num_frames=num_frames,
        ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if flat_frame_tokens.shape[-1] != embedding_dim:
            raise ValueError(
                "Frame token dimension must match prompt embedding dimension for concat_tokens: "
                f"{flat_frame_tokens.shape[-1]} vs {embedding_dim}."
            )
        scaled_frame_tokens = scale * flat_frame_tokens
        conditioned_prompt_embeds = torch.cat([prompt_embeds, scaled_frame_tokens], dim=1)
        temporal_tokens = scaled_frame_tokens.reshape(
            batch_frames // num_frames,
            num_frames,
            scaled_frame_tokens.shape[-2],
            scaled_frame_tokens.shape[-1],
        )
        return conditioned_prompt_embeds, temporal_tokens

    raise AssertionError(f"Unhandled frame token mode {token_mode!r}.")


class FramePositionMLP(nn.Module):
    """Maps normalized scalar frame positions to SDXL pooled-text embedding space."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.net = _make_mlp(1, output_dim, hidden_dim, num_layers)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

    def forward(self, frame_positions: torch.Tensor) -> torch.Tensor:
        if frame_positions.ndim != 1:
            frame_positions = frame_positions.reshape(-1)
        dtype = next(self.parameters()).dtype
        positions = frame_positions.to(dtype=dtype).unsqueeze(-1)
        return self.net(positions)


class FramePositionTokenEncoder(nn.Module):
    """Encodes scalar frame positions into learnable per-frame tokens.

    The token path follows the requested concat-token surgery:
    frame position -> projection, add a learned token basis, then refine each
    token with a lightweight MLP. The pooled output is the position projection
    and can be used by convolutional frame-conditioning branches.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_tokens: int = 3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if pooling != "mean":
            raise ValueError("Only pooling='mean' is implemented.")

        self.position_projection = _make_mlp(1, embedding_dim, hidden_dim, num_layers)
        self.token_basis = nn.Parameter(torch.zeros(num_tokens, embedding_dim))
        self.token_refine_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_tokens = int(num_tokens)
        self.pooling = pooling

    def forward(self, frame_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = tuple(frame_positions.shape)
        if not shape:
            frame_positions = frame_positions.reshape(1)
            shape = tuple(frame_positions.shape)
        dtype = next(self.parameters()).dtype
        positions = frame_positions.to(dtype=dtype).reshape(-1, 1)
        pooled = self.position_projection(positions)
        tokens = pooled[:, None, :] + self.token_basis[None, :, :]
        tokens = self.token_refine_mlp(tokens)
        pooled = pooled.reshape(*shape, self.embedding_dim)
        tokens = tokens.reshape(*shape, self.num_tokens, self.embedding_dim)
        return pooled, tokens


class SinusoidalFramePositionEncoder(nn.Module):
    """Deterministic sinusoidal frame-position encoder."""

    def __init__(
        self,
        embedding_dim: int,
        num_tokens: int = 1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.num_tokens = int(num_tokens)
        half_dim = self.embedding_dim // 2
        if half_dim > 0:
            frequencies = torch.exp(
                -torch.log(torch.tensor(10000.0))
                * torch.arange(half_dim, dtype=torch.float32)
                / max(half_dim - 1, 1)
            )
        else:
            frequencies = torch.empty(0, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, frame_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = tuple(frame_positions.shape)
        if not shape:
            frame_positions = frame_positions.reshape(1)
            shape = tuple(frame_positions.shape)

        positions = frame_positions.to(dtype=torch.float32).reshape(-1, 1)
        if self.frequencies.numel() > 0:
            frequencies = self.frequencies.to(device=positions.device, dtype=torch.float32)
            angles = positions * frequencies.unsqueeze(0)
            embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            embedding = torch.empty(positions.shape[0], 0, device=positions.device)

        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, positions], dim=-1)
        embedding = embedding.reshape(*shape, self.embedding_dim)
        embedding = embedding.to(dtype=frame_positions.dtype if frame_positions.is_floating_point() else torch.float32)
        tokens = embedding.unsqueeze(-2).expand(*shape, self.num_tokens, self.embedding_dim).contiguous()
        return embedding, tokens


def build_frame_position_encoder(config: dict) -> nn.Module:
    encoder_type = str(config.get("type", "learned_tokens")).lower()
    if encoder_type in {
        "learned_tokens",
        "learnable_tokens",
        "learnable",
        "learnable_token_concat",
        "token_basis",
        "mlp_tokens",
        "mlp",
    }:
        return FramePositionTokenEncoder(
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            num_tokens=int(config["num_tokens"]),
            pooling=config.get("pooling", "mean"),
        )
    if encoder_type in {"sinusoidal", "sin", "sincos"}:
        return SinusoidalFramePositionEncoder(
            embedding_dim=int(config["embedding_dim"]),
            num_tokens=int(config.get("num_tokens", 1)),
        )
    raise ValueError(f"Unsupported frame_position_encoder.type={encoder_type!r}.")
