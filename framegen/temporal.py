from __future__ import annotations

import torch
from torch import nn


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

        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
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
    """Encodes scalar frame positions into per-frame tokens and pooled embeddings."""

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

        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_tokens * embedding_dim))
        self.net = nn.Sequential(*layers)
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
        tokens = self.net(positions)
        tokens = tokens.reshape(*shape, self.num_tokens, self.embedding_dim)
        pooled = tokens.mean(dim=-2)
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

    def forward(self, frame_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = tuple(frame_positions.shape)
        if not shape:
            frame_positions = frame_positions.reshape(1)
            shape = tuple(frame_positions.shape)

        positions = frame_positions.to(dtype=torch.float32).reshape(-1, 1)
        half_dim = self.embedding_dim // 2
        if half_dim > 0:
            frequencies = torch.exp(
                -torch.log(torch.tensor(10000.0, device=positions.device))
                * torch.arange(half_dim, device=positions.device, dtype=torch.float32)
                / max(half_dim - 1, 1)
            )
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
    if encoder_type in {"learned_tokens", "mlp_tokens", "mlp"}:
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
