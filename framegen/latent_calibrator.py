from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as torch_f


@dataclass
class LatentCalibratorConfig:
    enabled: bool = False
    train: bool = True
    apply_mode: str = "switch_only"
    architecture: str = "temporal_conv"
    latent_channels: int = 4
    hidden_channels: int = 64
    num_blocks: int = 2
    zero_init_output: bool = True
    timestep_embedding_dim: int = 256
    prompt_embedding_dim: int = 1280
    use_timestep: bool = True
    use_frame_pos: bool = True
    use_pooled_prompt: bool = True
    use_anchor_latent: bool = True
    use_bridge_gate: bool = False
    use_log_snr: bool = False
    input_mode: str = "concat_anchor_delta"
    residual_scale_mode: str = "clipped_snr"
    residual_max_scale: float = 0.5
    residual_norm_cap: float = 1.0
    map_weight: float = 0.0
    map_lowfreq_only: bool = True
    map_downsample_factor: int = 4
    norm_weight: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "LatentCalibratorConfig":
        values = dict(config or {})
        conditioning = dict(values.get("conditioning", {}))
        residual = dict(values.get("residual", {}))
        auxiliary = dict(values.get("auxiliary_loss", {}))
        return cls(
            enabled=bool(values.get("enabled", cls.enabled)),
            train=bool(values.get("train", cls.train)),
            apply_mode=str(values.get("apply_mode", cls.apply_mode)),
            architecture=str(values.get("architecture", cls.architecture)),
            latent_channels=int(values.get("latent_channels", cls.latent_channels)),
            hidden_channels=int(values.get("hidden_channels", cls.hidden_channels)),
            num_blocks=int(values.get("num_blocks", cls.num_blocks)),
            zero_init_output=bool(values.get("zero_init_output", cls.zero_init_output)),
            timestep_embedding_dim=int(
                values.get("timestep_embedding_dim", cls.timestep_embedding_dim)
            ),
            prompt_embedding_dim=int(
                values.get("prompt_embedding_dim", cls.prompt_embedding_dim)
            ),
            use_timestep=bool(conditioning.get("use_timestep", cls.use_timestep)),
            use_frame_pos=bool(conditioning.get("use_frame_pos", cls.use_frame_pos)),
            use_pooled_prompt=bool(conditioning.get("use_pooled_prompt", cls.use_pooled_prompt)),
            use_anchor_latent=bool(conditioning.get("use_anchor_latent", cls.use_anchor_latent)),
            use_bridge_gate=bool(conditioning.get("use_bridge_gate", cls.use_bridge_gate)),
            use_log_snr=bool(conditioning.get("use_log_snr", cls.use_log_snr)),
            input_mode=str(conditioning.get("input_mode", cls.input_mode)),
            residual_scale_mode=str(residual.get("scale_mode", cls.residual_scale_mode)),
            residual_max_scale=float(residual.get("max_scale", cls.residual_max_scale)),
            residual_norm_cap=float(residual.get("norm_cap", cls.residual_norm_cap)),
            map_weight=float(auxiliary.get("map_weight", cls.map_weight)),
            map_lowfreq_only=bool(auxiliary.get("map_lowfreq_only", cls.map_lowfreq_only)),
            map_downsample_factor=int(
                auxiliary.get("map_downsample_factor", cls.map_downsample_factor)
            ),
            norm_weight=float(auxiliary.get("norm_weight", cls.norm_weight)),
        )


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        timesteps = timesteps.reshape(-1)
    half_dim = embedding_dim // 2
    if half_dim <= 0:
        return timesteps.float().unsqueeze(-1)
    exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device))
    frequencies = torch.exp(
        exponent * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / max(half_dim - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FiLMResBlock2D(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(torch_f.silu(self.norm1(hidden_states)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1.0 + scale) + shift
        hidden_states = self.conv2(torch_f.silu(hidden_states))
        return residual + hidden_states


class TemporalConvLatentCalibrator(nn.Module):
    """Zero-init residual mapper from anchor-expanded latents to video-like latents."""

    def __init__(self, config: LatentCalibratorConfig | dict[str, Any] | None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, LatentCalibratorConfig)
            else LatentCalibratorConfig.from_config(config)
        )
        if self.config.architecture != "temporal_conv":
            raise ValueError("Only latent_calibrator.architecture='temporal_conv' is implemented.")
        if self.config.apply_mode not in {"switch_only", "training", "gate_scaled"}:
            raise ValueError(
                "latent_calibrator.apply_mode must be switch_only, training, or gate_scaled."
            )
        if self.config.latent_channels <= 0:
            raise ValueError("latent_calibrator.latent_channels must be positive.")
        if self.config.hidden_channels <= 0:
            raise ValueError("latent_calibrator.hidden_channels must be positive.")
        if self.config.num_blocks <= 0:
            raise ValueError("latent_calibrator.num_blocks must be positive.")
        if not self.config.zero_init_output:
            raise ValueError("Only latent_calibrator.zero_init_output=true is implemented.")

        input_channels = self.config.latent_channels
        if self.config.use_anchor_latent:
            if self.config.input_mode != "concat_anchor_delta":
                raise ValueError(
                    "Only latent_calibrator.conditioning.input_mode='concat_anchor_delta' is implemented."
                )
            input_channels = self.config.latent_channels * 3

        hidden_channels = self.config.hidden_channels
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.temporal_conv1 = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=hidden_channels,
        )
        self.temporal_conv2 = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=hidden_channels,
        )

        cond_dim = hidden_channels
        self.time_proj = (
            nn.Sequential(
                nn.Linear(self.config.timestep_embedding_dim, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if self.config.use_timestep
            else None
        )
        self.frame_proj = (
            nn.Sequential(
                nn.Linear(1, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if self.config.use_frame_pos
            else None
        )
        self.prompt_proj = (
            nn.Sequential(
                nn.Linear(self.config.prompt_embedding_dim, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            if self.config.use_pooled_prompt
            else None
        )
        # Scalar conditioning on the smooth-SNR bridge gate g(t) and/or log-SNR,
        # so a gate_scaled calibrator knows how much anchor bias survived the blend.
        self.gate_proj = (
            nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
            if self.config.use_bridge_gate
            else None
        )
        self.logsnr_proj = (
            nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
            if self.config.use_log_snr
            else None
        )
        self.blocks = nn.ModuleList(
            [FiLMResBlock2D(hidden_channels, cond_dim) for _ in range(self.config.num_blocks)]
        )
        self.output_proj = _zero_module(
            nn.Conv2d(hidden_channels, self.config.latent_channels, kernel_size=3, padding=1)
        )

    def _temporal_conv(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        conv: nn.Conv3d,
    ) -> torch.Tensor:
        batch_frames, channels, height, width = hidden_states.shape
        if batch_frames % int(num_frames) != 0:
            raise ValueError(
                "Latent calibrator batch is not divisible by num_frames: "
                f"{batch_frames} vs {num_frames}."
            )
        batch_size = batch_frames // int(num_frames)
        video = hidden_states.reshape(batch_size, int(num_frames), channels, height, width)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = conv(video)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video.reshape(batch_frames, channels, height, width)

    def _conditioning(
        self,
        timesteps: torch.Tensor,
        frame_positions: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        dtype: torch.dtype,
        bridge_gate: torch.Tensor | None = None,
        log_snr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_frames = timesteps.reshape(-1).shape[0]
        cond = torch.zeros(
            batch_frames,
            self.config.hidden_channels,
            device=timesteps.device,
            dtype=dtype,
        )
        if self.time_proj is not None:
            time_embeds = timestep_embedding(
                timesteps,
                self.config.timestep_embedding_dim,
            ).to(device=timesteps.device, dtype=dtype)
            cond = cond + self.time_proj(time_embeds)
        if self.frame_proj is not None:
            frame_values = frame_positions.reshape(-1, 1).to(device=timesteps.device, dtype=dtype)
            cond = cond + self.frame_proj(frame_values)
        if self.prompt_proj is not None:
            if pooled_prompt_embeds.shape[-1] != self.config.prompt_embedding_dim:
                raise ValueError(
                    "Latent calibrator pooled prompt dim mismatch: "
                    f"{pooled_prompt_embeds.shape[-1]} vs {self.config.prompt_embedding_dim}."
                )
            cond = cond + self.prompt_proj(pooled_prompt_embeds.to(device=timesteps.device, dtype=dtype))
        if self.gate_proj is not None and bridge_gate is not None:
            gate_values = bridge_gate.reshape(-1, 1).to(device=timesteps.device, dtype=dtype)
            cond = cond + self.gate_proj(gate_values)
        if self.logsnr_proj is not None and log_snr is not None:
            logsnr_values = log_snr.reshape(-1, 1).to(device=timesteps.device, dtype=dtype)
            cond = cond + self.logsnr_proj(logsnr_values)
        return cond

    def _residual_scale(self, timesteps: torch.Tensor, noise_scheduler, dtype: torch.dtype) -> torch.Tensor:
        mode = str(self.config.residual_scale_mode).lower()
        if mode == "constant":
            scale = torch.full_like(timesteps.float(), float(self.config.residual_max_scale))
        elif mode == "clipped_snr":
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
            alpha = alphas_cumprod[timesteps.reshape(-1).long()]
            snr = alpha / (1.0 - alpha).clamp_min(1.0e-12)
            scale = snr.sqrt().clamp(max=float(self.config.residual_max_scale))
        else:
            raise ValueError(
                "latent_calibrator.residual.scale_mode must be constant or clipped_snr, "
                f"got {self.config.residual_scale_mode!r}."
            )
        while scale.ndim < 4:
            scale = scale.unsqueeze(-1)
        return scale.to(dtype=dtype)

    def _log_snr(self, timesteps: torch.Tensor, noise_scheduler, dtype: torch.dtype) -> torch.Tensor:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
        alpha = alphas_cumprod[timesteps.reshape(-1).long()]
        snr = alpha / (1.0 - alpha).clamp_min(1.0e-12)
        return snr.clamp_min(1.0e-12).log().to(dtype=dtype)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        anchor_latents: torch.Tensor,
        timesteps: torch.Tensor,
        frame_positions: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        num_frames: int,
        noise_scheduler,
        bridge_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noisy_latents.shape != anchor_latents.shape:
            raise ValueError(
                "Latent calibrator noisy and anchor latents must match, got "
                f"{tuple(noisy_latents.shape)} vs {tuple(anchor_latents.shape)}."
            )
        if noisy_latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, got {noisy_latents.shape[1]}."
            )

        output_dtype = noisy_latents.dtype
        compute_dtype = self.input_proj.weight.dtype
        noisy_latents_compute = noisy_latents.to(dtype=compute_dtype)
        anchor_latents_compute = anchor_latents.to(dtype=compute_dtype)
        log_snr = (
            self._log_snr(timesteps, noise_scheduler, compute_dtype)
            if self.logsnr_proj is not None
            else None
        )
        cond = self._conditioning(
            timesteps=timesteps,
            frame_positions=frame_positions,
            pooled_prompt_embeds=pooled_prompt_embeds,
            dtype=compute_dtype,
            bridge_gate=bridge_gate,
            log_snr=log_snr,
        )
        if self.config.use_anchor_latent:
            inputs = torch.cat(
                [
                    noisy_latents_compute,
                    anchor_latents_compute,
                    noisy_latents_compute - anchor_latents_compute,
                ],
                dim=1,
            )
        else:
            inputs = noisy_latents_compute
        hidden_states = self.input_proj(inputs)
        hidden_states = hidden_states + self._temporal_conv(hidden_states, num_frames, self.temporal_conv1)
        for block in self.blocks:
            hidden_states = block(hidden_states, cond)
        hidden_states = hidden_states + self._temporal_conv(hidden_states, num_frames, self.temporal_conv2)
        raw_delta = self.output_proj(torch_f.silu(hidden_states))

        norm_cap = float(self.config.residual_norm_cap)
        if norm_cap > 0:
            raw_delta = norm_cap * torch.tanh(raw_delta / norm_cap)
        scale = self._residual_scale(timesteps, noise_scheduler, dtype=compute_dtype)
        # gate_scaled: also scale the residual by the smooth-SNR bridge gate g(t),
        # so the calibrator only corrects in proportion to surviving anchor bias and
        # smoothly vanishes where g(t)->0 (no hard switch boundary).
        if self.config.apply_mode == "gate_scaled" and bridge_gate is not None:
            gate = bridge_gate.reshape(-1).to(device=scale.device, dtype=compute_dtype)
            while gate.ndim < scale.ndim:
                gate = gate.unsqueeze(-1)
            scale = scale * gate
        delta = scale * raw_delta
        delta = delta.to(dtype=output_dtype)
        scale = scale.to(dtype=output_dtype)
        return noisy_latents + delta, delta, scale


def build_latent_calibrator(config: dict[str, Any] | None) -> TemporalConvLatentCalibrator | None:
    calibrator_config = LatentCalibratorConfig.from_config(config)
    if not calibrator_config.enabled:
        return None
    return TemporalConvLatentCalibrator(calibrator_config)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weighted = values * mask
    return weighted.sum() / mask.expand_as(values).sum().clamp_min(1.0)


def latent_calibrator_alignment_loss(
    calibrated_latents: torch.Tensor,
    target_latents: torch.Tensor,
    mask: torch.Tensor,
    lowfreq_only: bool = True,
    downsample_factor: int = 4,
) -> torch.Tensor:
    if calibrated_latents.shape != target_latents.shape:
        raise ValueError(
            "Latent calibrator alignment tensors must match, got "
            f"{tuple(calibrated_latents.shape)} vs {tuple(target_latents.shape)}."
        )
    if lowfreq_only and int(downsample_factor) > 1:
        kernel = min(int(downsample_factor), calibrated_latents.shape[-1], calibrated_latents.shape[-2])
        if kernel > 1:
            calibrated_latents = torch_f.avg_pool2d(calibrated_latents, kernel_size=kernel, stride=kernel)
            target_latents = torch_f.avg_pool2d(target_latents, kernel_size=kernel, stride=kernel)
    return _masked_mean((calibrated_latents.float() - target_latents.float()).square(), mask)


def latent_calibrator_norm_loss(delta: torch.Tensor, mask: torch.Tensor, norm_cap: float) -> torch.Tensor:
    if float(norm_cap) <= 0:
        return delta.new_zeros(())
    per_frame_rms = delta.float().square().mean(dim=(1, 2, 3)).clamp_min(1.0e-12).sqrt()
    excess = torch_f.relu(per_frame_rms - float(norm_cap))
    return _masked_mean(excess.square(), mask)
