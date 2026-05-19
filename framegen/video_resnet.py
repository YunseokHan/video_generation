from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as torch_f

try:
    from diffusers.models.resnet import ResnetBlock2D
except ImportError:  # pragma: no cover - import-time guard for minimal test envs.
    ResnetBlock2D = None


@dataclass
class VideoResnetAdapterConfig:
    enabled: bool = False
    active: bool = True
    train: bool = True
    frame_embedding_dim: int = 1
    use_temporal_conv1: bool = True
    use_temporal_conv2: bool = True
    use_frame_conditioning: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "VideoResnetAdapterConfig":
        values = dict(config or {})
        return cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            active=bool(values.get("active", cls.active)),
            train=bool(values.get("train", cls.train)),
            frame_embedding_dim=int(values.get("frame_embedding_dim", cls.frame_embedding_dim)),
            use_temporal_conv1=bool(values.get("use_temporal_conv1", cls.use_temporal_conv1)),
            use_temporal_conv2=bool(values.get("use_temporal_conv2", cls.use_temporal_conv2)),
            use_frame_conditioning=bool(
                values.get("use_frame_conditioning", cls.use_frame_conditioning)
            ),
        )


class IdentityTemporalConv3D(nn.Module):
    """Temporal-only 3D convolution initialized as identity over flattened video frames."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = int(channels)
        self.conv = nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        with torch.no_grad():
            self.conv.weight.zero_()
            diagonal = torch.arange(self.channels)
            self.conv.weight[diagonal, diagonal, 1, 0, 0] = 1.0
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, hidden_states: torch.Tensor, num_frames: int | None) -> torch.Tensor:
        if num_frames is None:
            num_frames = 1
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        sample_count, channels, height, width = hidden_states.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        if sample_count % num_frames != 0:
            raise ValueError(
                "Flattened frame batch is not divisible by num_frames: "
                f"{sample_count} vs {num_frames}."
            )

        batch_size = sample_count // num_frames
        video = hidden_states.reshape(batch_size, num_frames, channels, height, width)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = self.conv(video)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video.reshape(sample_count, channels, height, width)


class VideoResnetBlock2D(nn.Module):
    """A ResnetBlock2D-compatible block with optional temporal and frame-conditioning adapters."""

    def __init__(
        self,
        base: nn.Module,
        frame_embedding_dim: int,
        use_temporal_conv1: bool = True,
        use_temporal_conv2: bool = True,
        use_frame_conditioning: bool = True,
        active: bool = True,
    ) -> None:
        super().__init__()
        if frame_embedding_dim <= 0:
            raise ValueError("frame_embedding_dim must be positive.")

        self.pre_norm = getattr(base, "pre_norm", True)
        self.in_channels = int(base.in_channels)
        self.out_channels = int(base.out_channels)
        self.use_conv_shortcut = bool(getattr(base, "use_conv_shortcut", False))
        self.up = bool(getattr(base, "up", False))
        self.down = bool(getattr(base, "down", False))
        self.output_scale_factor = float(getattr(base, "output_scale_factor", 1.0))
        self.time_embedding_norm = getattr(base, "time_embedding_norm", "default")
        self.skip_time_act = bool(getattr(base, "skip_time_act", False))

        self.norm1 = base.norm1
        self.conv1 = base.conv1
        self.time_emb_proj = base.time_emb_proj
        self.norm2 = base.norm2
        self.dropout = base.dropout
        self.conv2 = base.conv2
        self.nonlinearity = base.nonlinearity
        self.upsample = base.upsample
        self.downsample = base.downsample
        self.use_in_shortcut = bool(getattr(base, "use_in_shortcut", False))
        self.conv_shortcut = base.conv_shortcut

        self.frame_embedding_dim = int(frame_embedding_dim)
        self.use_temporal_conv1 = bool(use_temporal_conv1)
        self.use_temporal_conv2 = bool(use_temporal_conv2)
        self.use_frame_conditioning = bool(use_frame_conditioning)
        self.active = bool(active)

        frame_projection_dim = (
            self.time_emb_proj.out_features
            if self.time_emb_proj is not None
            else self.conv1.out_channels
        )
        self.temporal_conv1 = IdentityTemporalConv3D(self.conv1.out_channels)
        self.temporal_conv2 = IdentityTemporalConv3D(self.conv2.out_channels)
        self.frame_emb_proj = (
            nn.Linear(self.frame_embedding_dim, frame_projection_dim, bias=False)
            if self.use_frame_conditioning
            else None
        )
        if self.frame_emb_proj is not None:
            nn.init.zeros_(self.frame_emb_proj.weight)

        self.video_num_frames: int | None = None
        self.video_frame_embeddings: torch.Tensor | None = None
        self.video_frame_positions: torch.Tensor | None = None

    @property
    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.temporal_conv1.parameters()
        yield from self.temporal_conv2.parameters()
        if self.frame_emb_proj is not None:
            yield from self.frame_emb_proj.parameters()

    def set_video_context(
        self,
        num_frames: int | None,
        frame_positions: torch.Tensor | None = None,
        frame_embeddings: torch.Tensor | None = None,
    ) -> None:
        self.video_num_frames = int(num_frames) if num_frames is not None else None
        self.video_frame_positions = frame_positions
        self.video_frame_embeddings = frame_embeddings

    def clear_video_context(self) -> None:
        self.video_num_frames = None
        self.video_frame_positions = None
        self.video_frame_embeddings = None

    def _apply_temporal_conv(
        self,
        hidden_states: torch.Tensor,
        temporal_conv: IdentityTemporalConv3D,
        enabled: bool,
    ) -> torch.Tensor:
        if not self.active or not enabled:
            return hidden_states
        return temporal_conv(hidden_states, self.video_num_frames)

    def _expand_condition_to_batch(
        self,
        values: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = values.to(device=device, dtype=dtype)
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        values = values.reshape(values.shape[0], -1)
        if values.shape[0] == batch_size:
            return values
        if batch_size % values.shape[0] == 0:
            return values.repeat(batch_size // values.shape[0], 1)
        raise ValueError(
            "Frame conditioning batch does not match hidden batch: "
            f"{values.shape[0]} vs {batch_size}."
        )

    def _make_placeholder_frame_embeddings(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.video_frame_embeddings is not None:
            embeddings = self._expand_condition_to_batch(
                self.video_frame_embeddings,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        elif self.video_frame_positions is not None:
            embeddings = self._expand_condition_to_batch(
                self.video_frame_positions,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        elif self.video_num_frames is not None:
            num_frames = int(self.video_num_frames)
            if batch_size % num_frames != 0:
                raise ValueError(
                    "Cannot infer placeholder frame positions because hidden batch is "
                    f"not divisible by num_frames: {batch_size} vs {num_frames}."
                )
            positions = torch.linspace(0.0, 1.0, steps=num_frames, device=device, dtype=dtype)
            embeddings = positions.repeat(batch_size // num_frames).unsqueeze(-1)
        else:
            embeddings = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        if embeddings.shape[-1] == self.frame_embedding_dim:
            return embeddings
        projected = torch.zeros(
            embeddings.shape[0],
            self.frame_embedding_dim,
            device=device,
            dtype=dtype,
        )
        used_dim = min(self.frame_embedding_dim, embeddings.shape[-1])
        projected[:, :used_dim] = embeddings[:, :used_dim]
        return projected

    def _frame_temb(self, temb: torch.Tensor, out_channels: int) -> torch.Tensor:
        if not self.active or not self.use_frame_conditioning or self.frame_emb_proj is None:
            raise RuntimeError("Frame conditioning was requested but frame_emb_proj is unavailable.")
        frame_embeddings = self._make_placeholder_frame_embeddings(
            batch_size=temb.shape[0],
            device=temb.device,
            dtype=temb.dtype,
        )
        return self.frame_emb_proj(torch_f.silu(frame_embeddings))

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self._apply_temporal_conv(
            hidden_states,
            temporal_conv=self.temporal_conv1,
            enabled=self.use_temporal_conv1,
        )

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            projected_temb = self.time_emb_proj(temb)
            if self.active and self.use_frame_conditioning:
                projected_temb = projected_temb + self._frame_temb(
                    temb=temb,
                    out_channels=projected_temb.shape[-1],
                )
            projected_temb = projected_temb[:, :, None, None]
        else:
            projected_temb = None

        if self.time_embedding_norm == "default":
            if projected_temb is not None:
                hidden_states = hidden_states + projected_temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if projected_temb is None:
                raise ValueError(
                    f"`temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(projected_temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self._apply_temporal_conv(
            hidden_states,
            temporal_conv=self.temporal_conv2,
            enabled=self.use_temporal_conv2,
        )

        if self.conv_shortcut is not None:
            if self.training:
                input_tensor = input_tensor.contiguous()
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


def _target_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def iter_video_resnet_blocks(module: nn.Module) -> Iterator[VideoResnetBlock2D]:
    module = _target_module(module)
    for child in module.modules():
        if isinstance(child, VideoResnetBlock2D):
            yield child


def inject_video_resnet_adapters(
    module: nn.Module,
    config: VideoResnetAdapterConfig | dict[str, Any] | None,
) -> int:
    adapter_config = (
        config
        if isinstance(config, VideoResnetAdapterConfig)
        else VideoResnetAdapterConfig.from_config(config)
    )
    if not adapter_config.enabled:
        return 0
    if ResnetBlock2D is None:
        raise ImportError("diffusers is required to inject video Resnet adapters.")

    module = _target_module(module)
    replaced = 0

    def replace_children(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, VideoResnetBlock2D):
                continue
            if isinstance(child, ResnetBlock2D):
                setattr(
                    parent,
                    name,
                    VideoResnetBlock2D(
                        child,
                        frame_embedding_dim=adapter_config.frame_embedding_dim,
                        use_temporal_conv1=adapter_config.use_temporal_conv1,
                        use_temporal_conv2=adapter_config.use_temporal_conv2,
                        use_frame_conditioning=adapter_config.use_frame_conditioning,
                        active=adapter_config.active,
                    ),
                )
                replaced += 1
            else:
                replace_children(child)

    replace_children(module)
    return replaced


def set_video_resnet_adapters_active(module: nn.Module, active: bool) -> None:
    for block in iter_video_resnet_blocks(module):
        block.active = bool(active)


def set_video_resnet_adapter_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for block in iter_video_resnet_blocks(module):
        for parameter in block.adapter_parameters:
            parameter.requires_grad_(requires_grad)


def set_video_resnet_context(
    module: nn.Module,
    num_frames: int | None,
    frame_positions: torch.Tensor | None = None,
    frame_embeddings: torch.Tensor | None = None,
) -> None:
    for block in iter_video_resnet_blocks(module):
        block.set_video_context(
            num_frames=num_frames,
            frame_positions=frame_positions,
            frame_embeddings=frame_embeddings,
        )


def clear_video_resnet_context(module: nn.Module) -> None:
    for block in iter_video_resnet_blocks(module):
        block.clear_video_context()


def sync_video_resnet_adapter_device_dtype(module: nn.Module) -> None:
    module = _target_module(module)
    try:
        reference = next(module.parameters())
    except StopIteration:
        return
    for block in iter_video_resnet_blocks(module):
        block.temporal_conv1.to(device=reference.device, dtype=reference.dtype)
        block.temporal_conv2.to(device=reference.device, dtype=reference.dtype)
        if block.frame_emb_proj is not None:
            block.frame_emb_proj.to(device=reference.device, dtype=reference.dtype)


def video_resnet_adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    module = _target_module(module)
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
        if ".temporal_conv" in key or ".frame_emb_proj." in key
    }


def load_video_resnet_adapter_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    module = _target_module(module)
    load_result = module.load_state_dict(state_dict, strict=False)
    if strict:
        missing_adapter_keys = [
            key
            for key in video_resnet_adapter_state_dict(module)
            if key not in state_dict
        ]
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_adapter_keys or unexpected_keys:
            raise RuntimeError(
                "Video Resnet adapter state mismatch: "
                f"missing={missing_adapter_keys}, unexpected={unexpected_keys}"
            )
