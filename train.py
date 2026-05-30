from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file

load_env_file(PROJECT_ROOT / ".env", override=False)

import torch
import torch.nn.functional as torch_f
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in the training env.
    tqdm = None

from framegen.checkpointing import (
    load_video_attention_adapter_checkpoint,
    load_video_resnet_adapter_checkpoint,
    save_checkpoint,
)
from framegen.config import as_project_path, get_torch_dtype, load_config, save_config
from framegen.data import build_dataset, flatten_video_batch, video_collate_fn
from framegen.generation import generate_video_frames, guidance_scale_label
from framegen.image_first_generation import (
    image_first_output_dir,
    generate_image_first_video_frames,
    t1_ratio_label,
)
from framegen.latent_calibrator import (
    LatentCalibratorConfig,
    build_latent_calibrator,
    latent_calibrator_alignment_loss,
    latent_calibrator_norm_loss,
)
from framegen.sdxl import (
    add_temporal_embedding_to_pooled_prompt_embeds,
    compute_sdxl_time_ids,
    encode_prompts_with_sdxl_logic,
    validate_sdxl_added_conditioning_dimensions,
)
from framegen.temporal import (
    FramePositionMLP,
    apply_frame_token_conditioning,
    build_frame_position_encoder,
    normalize_frame_token_mode,
)
from framegen.utils import count_trainable_and_frozen, format_param_count, set_requires_grad
from framegen.video_attention import (
    VideoAttentionAdapterConfig,
    clear_video_attention_context,
    inject_video_attention_adapters,
    iter_video_attention_blocks,
    set_video_attention_adapter_requires_grad,
    set_video_attention_adapters_active,
    set_video_attention_context,
    sync_video_attention_adapter_device_dtype,
)
from framegen.video_resnet import (
    VideoResnetAdapterConfig,
    clear_video_resnet_context,
    inject_video_resnet_adapters,
    iter_video_resnet_blocks,
    set_video_resnet_adapter_requires_grad,
    set_video_resnet_adapters_active,
    set_video_resnet_context,
    sync_video_resnet_adapter_device_dtype,
)


LATENT_INIT_VIDEO_GT = "video_gt"
LATENT_INIT_FIRST_FRAME_REPEAT = "first_frame_repeat"
IMAGE_FIRST_NOISE_INDEPENDENT = "independent"
IMAGE_FIRST_NOISE_SHARED = "shared"
IMAGE_FIRST_NOISE_MIXED = "mixed"
IMAGE_FIRST_BRIDGE_ALWAYS = "always"
IMAGE_FIRST_BRIDGE_SNR = "snr"
IMAGE_FIRST_BRIDGE_SMOOTH_SNR = "smooth_snr"
IMAGE_FIRST_BRIDGE_ROLLOUT = "rollout"
IMAGE_FIRST_BRIDGE_ROLLOUT_SNR = "rollout_snr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SDXL frame-position-conditioned generator.")
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--env_file", default=".env", help="Path to an environment-variable file.")
    return parser.parse_args()


def normalize_latent_init_mode(mode: str | None) -> str:
    normalized = str(mode or LATENT_INIT_VIDEO_GT).lower().replace("-", "_")
    aliases = {
        "": LATENT_INIT_VIDEO_GT,
        "default": LATENT_INIT_VIDEO_GT,
        "video": LATENT_INIT_VIDEO_GT,
        "video_gt": LATENT_INIT_VIDEO_GT,
        "gt_video": LATENT_INIT_VIDEO_GT,
        "standard": LATENT_INIT_VIDEO_GT,
        "first": LATENT_INIT_FIRST_FRAME_REPEAT,
        "first_frame": LATENT_INIT_FIRST_FRAME_REPEAT,
        "first_frame_repeat": LATENT_INIT_FIRST_FRAME_REPEAT,
        "image_first": LATENT_INIT_FIRST_FRAME_REPEAT,
        "anchor_first_frame": LATENT_INIT_FIRST_FRAME_REPEAT,
    }
    if normalized not in aliases:
        raise ValueError(
            "training.latent_init_mode must be one of video_gt or first_frame_repeat, "
            f"got {mode!r}."
        )
    return aliases[normalized]


def normalize_image_first_noise_mode(mode: str | None) -> str:
    normalized = str(mode or IMAGE_FIRST_NOISE_INDEPENDENT).lower().replace("-", "_")
    aliases = {
        "": IMAGE_FIRST_NOISE_INDEPENDENT,
        "default": IMAGE_FIRST_NOISE_INDEPENDENT,
        "independent": IMAGE_FIRST_NOISE_INDEPENDENT,
        "frame_independent": IMAGE_FIRST_NOISE_INDEPENDENT,
        "shared": IMAGE_FIRST_NOISE_SHARED,
        "same": IMAGE_FIRST_NOISE_SHARED,
        "same_noise": IMAGE_FIRST_NOISE_SHARED,
        "mixed": IMAGE_FIRST_NOISE_MIXED,
        "mix": IMAGE_FIRST_NOISE_MIXED,
    }
    if normalized not in aliases:
        raise ValueError(
            "training.image_first_noise_mode must be one of independent, shared, or mixed, "
            f"got {mode!r}."
        )
    return aliases[normalized]


def normalize_image_first_bridge_mode(mode: str | None) -> str:
    normalized = str(mode or IMAGE_FIRST_BRIDGE_ALWAYS).lower().replace("-", "_")
    aliases = {
        "": IMAGE_FIRST_BRIDGE_ALWAYS,
        "default": IMAGE_FIRST_BRIDGE_ALWAYS,
        "always": IMAGE_FIRST_BRIDGE_ALWAYS,
        "anchor": IMAGE_FIRST_BRIDGE_ALWAYS,
        "anchor_repeat": IMAGE_FIRST_BRIDGE_ALWAYS,
        "first_frame_repeat": IMAGE_FIRST_BRIDGE_ALWAYS,
        "snr": IMAGE_FIRST_BRIDGE_SNR,
        "snr_hybrid": IMAGE_FIRST_BRIDGE_SNR,
        "snr_gated": IMAGE_FIRST_BRIDGE_SNR,
        "smooth_snr": IMAGE_FIRST_BRIDGE_SMOOTH_SNR,
        "soft_snr": IMAGE_FIRST_BRIDGE_SMOOTH_SNR,
        "cosine_snr": IMAGE_FIRST_BRIDGE_SMOOTH_SNR,
        "smooth_snr_bridge": IMAGE_FIRST_BRIDGE_SMOOTH_SNR,
        "rollout": IMAGE_FIRST_BRIDGE_ROLLOUT,
        "image_rollout": IMAGE_FIRST_BRIDGE_ROLLOUT,
        "anchor_rollout": IMAGE_FIRST_BRIDGE_ROLLOUT,
        "rollout_snr": IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
        "snr_rollout": IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
        "rollout_snr_hybrid": IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
        "rollout_snr_gated": IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
    }
    if normalized not in aliases:
        raise ValueError(
            "training.image_first_bridge_mode must be one of always, snr, smooth_snr, "
            "rollout, or rollout_snr, "
            f"got {mode!r}."
        )
    return aliases[normalized]


def maybe_load_pipeline(model_config: dict, torch_dtype: torch.dtype) -> StableDiffusionXLPipeline:
    kwargs = {"torch_dtype": torch_dtype}
    if model_config.get("revision") is not None:
        kwargs["revision"] = model_config["revision"]
    if model_config.get("variant") is not None:
        kwargs["variant"] = model_config["variant"]
    cache_dir = get_hf_cache_dir(model_config)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = get_hf_token(model_config)
    if token is not None:
        kwargs["token"] = token
    return StableDiffusionXLPipeline.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        **kwargs,
    )


def load_sdxl_vae(model_config: dict, vae_dtype: torch.dtype) -> AutoencoderKL:
    vae_path = model_config.get("pretrained_vae_model_name_or_path")
    if vae_path is None:
        vae_path = model_config["pretrained_model_name_or_path"]
        subfolder = "vae"
    else:
        subfolder = None

    kwargs = {"torch_dtype": vae_dtype}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    if model_config.get("revision") is not None:
        kwargs["revision"] = model_config["revision"]
    if model_config.get("variant") is not None:
        kwargs["variant"] = model_config["variant"]
    cache_dir = get_hf_cache_dir(model_config)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = get_hf_token(model_config)
    if token is not None:
        kwargs["token"] = token
    return AutoencoderKL.from_pretrained(vae_path, **kwargs)


def load_noise_scheduler(model_config: dict) -> DDPMScheduler:
    kwargs = {}
    if model_config.get("revision") is not None:
        kwargs["revision"] = model_config["revision"]
    cache_dir = get_hf_cache_dir(model_config)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = get_hf_token(model_config)
    if token is not None:
        kwargs["token"] = token
    return DDPMScheduler.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        subfolder="scheduler",
        **kwargs,
    )


def build_optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    if bool(training_config.get("use_8bit_adam", False)):
        optimizer_type = "adamw8bit"
    if optimizer_type == "adamw":
        optimizer_class = torch.optim.AdamW
    elif optimizer_type in {"adamw8bit", "8bit_adam", "adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer.type='adamw8bit' or training.use_8bit_adam=true requires bitsandbytes."
            ) from exc
        optimizer_class = bnb.optim.AdamW8bit
    else:
        raise ValueError(f"Unsupported optimizer.type={optimizer_type!r}.")

    return optimizer_class(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1.0e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 1.0e-2)),
    )


def build_lr_scheduler(optimizer, training_config: dict, max_train_steps: int):
    scheduler_name = training_config.get("lr_scheduler", "constant")
    warmup_steps = int(training_config.get("lr_warmup_steps", 0))
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=int(max_train_steps),
    )


def prepare_trainable_models(accelerator, named_models, optimizer, dataloader, lr_scheduler):
    names = [name for name, model in named_models]
    models = [model for name, model in named_models]
    prepared = accelerator.prepare(*models, optimizer, dataloader, lr_scheduler)
    prepared_models = dict(zip(names, prepared[: len(models)]))
    return prepared_models, prepared[-3], prepared[-2], prepared[-1]


def generate_timestep_weights(training_config: dict, num_timesteps: int) -> torch.Tensor:
    weights = torch.ones(int(num_timesteps), dtype=torch.float32)
    strategy = str(training_config.get("timestep_bias_strategy", "none")).lower()
    if strategy in {"", "none", "null", "false"}:
        return weights

    portion = float(training_config.get("timestep_bias_portion", 0.25))
    multiplier = float(training_config.get("timestep_bias_multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("training.timestep_bias_multiplier must be positive.")
    num_to_bias = max(1, int(portion * int(num_timesteps)))

    if strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif strategy == "range":
        begin = int(training_config.get("timestep_bias_begin", 0))
        end = int(training_config.get("timestep_bias_end", int(num_timesteps)))
        if begin < 0 or end > int(num_timesteps) or begin >= end:
            raise ValueError(
                "Invalid timestep bias range: expected 0 <= begin < end <= num_train_timesteps."
            )
        bias_indices = slice(begin, end)
    else:
        raise ValueError(
            "training.timestep_bias_strategy must be one of none, earlier, later, or range."
        )

    weights[bias_indices] *= multiplier
    weights /= weights.sum()
    return weights


def sample_video_timesteps(
    batch_size: int,
    num_frames: int,
    num_train_timesteps: int,
    device: torch.device,
    timestep_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive.")
    if timestep_weights is None:
        video_timesteps = torch.randint(
            0,
            int(num_train_timesteps),
            (int(batch_size),),
            device=device,
        ).long()
    else:
        weights = timestep_weights.to(device=device, dtype=torch.float32)
        if weights.shape[0] != int(num_train_timesteps):
            raise ValueError(
                "timestep_weights length must match num_train_timesteps: "
                f"{weights.shape[0]} vs {num_train_timesteps}."
            )
        weights = weights / weights.sum()
        video_timesteps = torch.multinomial(weights, int(batch_size), replacement=True).long()
    return video_timesteps.repeat_interleave(int(num_frames))


def add_noise_offset(noise: torch.Tensor, offset: float) -> torch.Tensor:
    if float(offset) == 0.0:
        return noise
    offset_noise = torch.randn(
        (noise.shape[0], noise.shape[1], 1, 1),
        device=noise.device,
        dtype=noise.dtype,
    )
    return noise + float(offset) * offset_noise


def repeat_first_frame_latents(latents: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    if latents.shape[0] != int(batch_size) * int(num_frames):
        raise ValueError(
            "latents must be flattened [B*F, C, H, W], got "
            f"{tuple(latents.shape)} for B={batch_size}, F={num_frames}."
        )
    video_latents = latents.reshape(int(batch_size), int(num_frames), *latents.shape[1:])
    anchor = video_latents[:, :1].expand(-1, int(num_frames), -1, -1, -1)
    return anchor.reshape_as(latents)


def _scalar(value) -> float:
    """Convert a python float or a (possibly GPU) tensor to a python float.

    Used only in logging paths so the per-micro-step hot loop can keep these
    diagnostics as device tensors and avoid a CUDA sync on every step.
    """
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def sample_image_first_noise(
    reference: torch.Tensor,
    batch_size: int,
    num_frames: int,
    noise_mode: str,
    shared_noise_prob: float,
    noise_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape[0] != int(batch_size) * int(num_frames):
        raise ValueError(
            "reference must be flattened [B*F, C, H, W], got "
            f"{tuple(reference.shape)} for B={batch_size}, F={num_frames}."
        )
    mode = normalize_image_first_noise_mode(noise_mode)
    probability = float(shared_noise_prob)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("training.image_first_shared_noise_prob must be in [0, 1].")

    batch_size = int(batch_size)
    num_frames = int(num_frames)
    shape = reference.shape[1:]
    independent = torch.randn_like(reference).reshape(batch_size, num_frames, *shape)
    shared = torch.randn(
        (batch_size, 1, *shape),
        device=reference.device,
        dtype=reference.dtype,
    ).expand(-1, num_frames, -1, -1, -1)

    if mode == IMAGE_FIRST_NOISE_SHARED:
        mask = torch.ones(batch_size, 1, 1, 1, 1, device=reference.device, dtype=torch.bool)
    elif mode == IMAGE_FIRST_NOISE_MIXED:
        mask = (
            torch.rand(batch_size, 1, 1, 1, 1, device=reference.device)
            < probability
        )
    else:
        mask = torch.zeros(batch_size, 1, 1, 1, 1, device=reference.device, dtype=torch.bool)

    noise = torch.where(mask, shared, independent)
    if float(noise_offset) != 0.0:
        independent_offset = torch.randn(
            (batch_size, num_frames, shape[0], 1, 1),
            device=reference.device,
            dtype=reference.dtype,
        )
        shared_offset = torch.randn(
            (batch_size, 1, shape[0], 1, 1),
            device=reference.device,
            dtype=reference.dtype,
        ).expand(-1, num_frames, -1, -1, -1)
        offset_noise = torch.where(mask, shared_offset, independent_offset)
        noise = noise + float(noise_offset) * offset_noise

    return noise.reshape_as(reference), mask.float().mean()


def _optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def compute_image_first_bridge_mask(
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    bridge_mode: str,
    snr_min: float | None,
    snr_max: float | None,
) -> torch.Tensor:
    mode = normalize_image_first_bridge_mode(bridge_mode)
    if mode not in {IMAGE_FIRST_BRIDGE_SNR, IMAGE_FIRST_BRIDGE_ROLLOUT_SNR}:
        return torch.ones_like(timesteps, dtype=torch.bool)

    snr = compute_snr(noise_scheduler, timesteps)
    mask = torch.ones_like(timesteps, dtype=torch.bool)
    if snr_min is not None:
        mask = mask & (snr >= float(snr_min))
    if snr_max is not None:
        mask = mask & (snr <= float(snr_max))
    return mask


def compute_image_first_smooth_snr_gate(
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    snr_full: float,
    snr_zero: float,
    gate_mode: str = "cosine",
    domain: str = "log_snr",
) -> torch.Tensor:
    """Return a soft bridge gate that is 1 at low SNR and 0 at high SNR."""
    full = float(snr_full)
    zero = float(snr_zero)
    if full <= 0.0 or zero <= 0.0:
        raise ValueError("smooth SNR gate thresholds must be positive.")
    if full >= zero:
        raise ValueError("training.image_first_bridge_snr_full must be < image_first_bridge_snr_zero.")

    snr = compute_snr(noise_scheduler, timesteps).float()
    normalized_domain = str(domain or "log_snr").lower().replace("-", "_")
    if normalized_domain in {"log", "log_snr", "logsnr"}:
        start = math.log(max(full, 1.0e-12))
        stop = math.log(max(zero, 1.0e-12))
        position = (snr.clamp_min(1.0e-12).log() - start) / max(stop - start, 1.0e-12)
    elif normalized_domain in {"linear", "snr"}:
        position = (snr - full) / max(zero - full, 1.0e-12)
    else:
        raise ValueError("training.image_first_bridge_gate_domain must be 'log_snr' or 'snr'.")
    position = position.clamp(0.0, 1.0)

    normalized_gate = str(gate_mode or "cosine").lower().replace("-", "_")
    if normalized_gate == "cosine":
        return 0.5 * (1.0 + torch.cos(math.pi * position))
    if normalized_gate == "linear":
        return 1.0 - position
    raise ValueError("training.image_first_bridge_gate must be 'cosine' or 'linear'.")


def timestep_closest_to_snr(
    noise_scheduler: DDPMScheduler,
    target_snr: float,
    device: torch.device,
) -> torch.Tensor:
    if float(target_snr) <= 0.0:
        raise ValueError("target_snr must be positive.")
    alphas_cumprod = noise_scheduler.alphas_cumprod.float()
    snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp_min(1.0e-12)
    target = math.log(max(float(target_snr), 1.0e-12))
    timestep = torch.argmin((snr.clamp_min(1.0e-12).log() - target).abs())
    return timestep.to(device=device, dtype=torch.long)


def predict_clean_latents_from_epsilon(
    noisy_latents: torch.Tensor,
    epsilon: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device,
        dtype=torch.float32,
    )
    alpha_prod_t = alphas_cumprod[timesteps]
    while alpha_prod_t.ndim < noisy_latents.ndim:
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    sqrt_alpha_prod = alpha_prod_t.clamp_min(1.0e-12).sqrt()
    sqrt_one_minus_alpha_prod = (1.0 - alpha_prod_t).clamp_min(1.0e-12).sqrt()
    return (noisy_latents.float() - sqrt_one_minus_alpha_prod * epsilon.float()) / sqrt_alpha_prod


def compute_boundary_renoise_loss(
    pred_clean_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    boundary_noise: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    boundary_timestep: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    lowfreq_only: bool = True,
    downsample_factor: int = 4,
) -> torch.Tensor:
    batch = clean_latents.shape[0]
    timesteps = boundary_timestep.reshape(1).repeat(batch).to(
        device=clean_latents.device,
        dtype=torch.long,
    )
    pred_boundary = noise_scheduler.add_noise(
        pred_clean_latents.float(),
        boundary_noise.float(),
        timesteps,
    )
    target_boundary = noise_scheduler.add_noise(
        clean_latents.float(),
        boundary_noise.float(),
        timesteps,
    ).detach()
    if bool(lowfreq_only) and int(downsample_factor) > 1:
        factor = int(downsample_factor)
        pred_boundary = torch_f.avg_pool2d(pred_boundary, kernel_size=factor, stride=factor)
        target_boundary = torch_f.avg_pool2d(target_boundary, kernel_size=factor, stride=factor)
    per_sample = (pred_boundary - target_boundary).square().mean(dim=list(range(1, pred_boundary.ndim)))
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype).reshape_as(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _expand_latent_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(device=reference.device, dtype=torch.bool)
    while expanded.ndim < reference.ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded


def _expand_rollout_switch_level(
    switch_level: torch.Tensor,
    batch_size: int,
    num_frames: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Repeat per-video switch noise levels over frames for latent broadcasting."""
    expanded = switch_level.to(device=reference.device, dtype=reference.dtype).reshape(
        int(batch_size),
        *([1] * (reference.ndim - 1)),
    )
    expanded = expanded[:, None].expand(
        int(batch_size),
        int(num_frames),
        *expanded.shape[1:],
    )
    return expanded.reshape(int(batch_size) * int(num_frames), *expanded.shape[2:])


def _snapshot_adapter_states(module) -> list[tuple[object, bool]]:
    snapshots: list[tuple[object, bool]] = []
    snapshots.extend((block, bool(block.active)) for block in iter_video_resnet_blocks(module))
    snapshots.extend((block, bool(block.active)) for block in iter_video_attention_blocks(module))
    return snapshots


def _set_adapter_snapshot_active(snapshot: list[tuple[object, bool]], active: bool) -> None:
    for block, _ in snapshot:
        block.active = bool(active)


def _restore_adapter_snapshot(snapshot: list[tuple[object, bool]]) -> None:
    for block, active in snapshot:
        block.active = bool(active)


@torch.no_grad()
def rollout_image_first_anchor_latents(
    unet,
    noise_scheduler: DDPMScheduler,
    anchor_latents: torch.Tensor,
    target_timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    rollout_steps: int,
    noise_offset: float = 0.0,
) -> torch.Tensor:
    """Create inference-like image latents by denoising anchors with adapters off.

    Each sample starts from the clean anchor noised at `target_t + rollout_steps`
    and runs the base SDXL UNet down to `target_t`. The returned tensor is still
    a single image latent per video; callers duplicate it across frames.
    """

    if anchor_latents.ndim != 4:
        raise ValueError(f"Expected anchor_latents [B, C, H, W], got {tuple(anchor_latents.shape)}.")
    batch_size = anchor_latents.shape[0]
    if target_timesteps.shape[0] != batch_size:
        raise ValueError("target_timesteps must contain one timestep per anchor latent.")
    if int(rollout_steps) < 0:
        raise ValueError("training.image_first_rollout_steps must be non-negative.")

    steps = int(rollout_steps)
    max_timestep = int(noise_scheduler.config.num_train_timesteps) - 1
    adapter_snapshot = _snapshot_adapter_states(unet)
    training_states = _set_eval_temporarily(unet)
    outputs: list[torch.Tensor] = []

    try:
        _set_adapter_snapshot_active(adapter_snapshot, active=False)
        for index in range(batch_size):
            target_t = int(target_timesteps[index].item())
            start_t = min(max_timestep, target_t + steps)
            timestep_tensor = torch.tensor(
                [start_t],
                device=anchor_latents.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(anchor_latents[index : index + 1])
            noise = add_noise_offset(noise, float(noise_offset))
            latent = noise_scheduler.add_noise(
                anchor_latents[index : index + 1],
                noise,
                timestep_tensor,
            )

            current_t = start_t
            while current_t > target_t:
                step_timestep = torch.tensor(
                    current_t,
                    device=anchor_latents.device,
                    dtype=torch.long,
                )
                latent_model_input = noise_scheduler.scale_model_input(latent, step_timestep)
                model_pred = unet(
                    latent_model_input,
                    step_timestep,
                    encoder_hidden_states=prompt_embeds[index : index + 1],
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds[index : index + 1],
                        "time_ids": add_time_ids[index : index + 1],
                    },
                ).sample
                latent = noise_scheduler.step(
                    model_pred,
                    current_t,
                    latent,
                    return_dict=False,
                )[0]
                current_t -= 1
            outputs.append(latent[0])
    finally:
        _restore_adapter_snapshot(adapter_snapshot)
        _restore_training_states(training_states)

    return torch.stack(outputs, dim=0)


@torch.no_grad()
def rollout_image_first_anchor_pred_x0(
    unet,
    noise_scheduler: DDPMScheduler,
    anchor_latents: torch.Tensor,
    source_timestep: int,
    rollout_steps: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    noise_offset: float = 0.0,
) -> torch.Tensor:
    """Rollout a clean-x0 estimate of the anchor image (Agenda U3 E4).

    Noises the clean first-frame latent at a FIXED ``source_timestep`` (codex's
    "K fixed"), runs the base SDXL UNet with video adapters off for
    ``rollout_steps``, then returns the model's predicted x0 at the final step.
    This is an x0-like latent (not a noisy latent), so it can be plugged into the
    smooth-SNR bridge as the anchor source without double-counting noise. Callers
    duplicate it across frames.
    """

    if anchor_latents.ndim != 4:
        raise ValueError(f"Expected anchor_latents [B, C, H, W], got {tuple(anchor_latents.shape)}.")
    batch_size = anchor_latents.shape[0]
    max_timestep = int(noise_scheduler.config.num_train_timesteps) - 1
    start_t = max(0, min(max_timestep, int(source_timestep)))
    steps = max(1, int(rollout_steps))
    stop_t = max(0, start_t - steps)

    adapter_snapshot = _snapshot_adapter_states(unet)
    training_states = _set_eval_temporarily(unet)
    outputs: list[torch.Tensor] = []
    try:
        _set_adapter_snapshot_active(adapter_snapshot, active=False)
        for index in range(batch_size):
            timestep_tensor = torch.tensor([start_t], device=anchor_latents.device, dtype=torch.long)
            noise = torch.randn_like(anchor_latents[index : index + 1])
            noise = add_noise_offset(noise, float(noise_offset))
            latent = noise_scheduler.add_noise(anchor_latents[index : index + 1], noise, timestep_tensor)

            current_t = start_t
            pre_step_latent = latent
            last_eps = None
            last_t = start_t
            while current_t > stop_t:
                step_timestep = torch.tensor(current_t, device=anchor_latents.device, dtype=torch.long)
                latent_model_input = noise_scheduler.scale_model_input(latent, step_timestep)
                model_pred = unet(
                    latent_model_input,
                    step_timestep,
                    encoder_hidden_states=prompt_embeds[index : index + 1],
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds[index : index + 1],
                        "time_ids": add_time_ids[index : index + 1],
                    },
                ).sample
                pre_step_latent = latent
                last_eps = model_pred
                last_t = current_t
                latent = noise_scheduler.step(model_pred, current_t, latent, return_dict=False)[0]
                current_t -= 1

            # Predicted x0 from the final epsilon: pred_x0 = (z_t - sqrt(1-abar)*eps)/sqrt(abar).
            last_t_tensor = torch.tensor([last_t], device=anchor_latents.device, dtype=torch.long)
            pred_x0 = predict_clean_latents_from_epsilon(
                noisy_latents=pre_step_latent,
                epsilon=last_eps,
                noise_scheduler=noise_scheduler,
                timesteps=last_t_tensor,
            )
            outputs.append(pred_x0[0])
    finally:
        _restore_adapter_snapshot(adapter_snapshot)
        _restore_training_states(training_states)

    return torch.stack(outputs, dim=0)


def compute_clean_latent_epsilon_target(
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    alpha_prod_t = alphas_cumprod[timesteps]
    while alpha_prod_t.ndim < noisy_latents.ndim:
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    sqrt_alpha_prod = alpha_prod_t.sqrt()
    sqrt_one_minus_alpha_prod = (1.0 - alpha_prod_t).clamp_min(1.0e-12).sqrt()
    return (noisy_latents - sqrt_alpha_prod * clean_latents) / sqrt_one_minus_alpha_prod


def compute_diffusion_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    snr_gamma: float | None,
) -> torch.Tensor:
    if snr_gamma is None:
        return torch_f.mse_loss(model_pred.float(), target.float(), reduction="mean")

    snr = compute_snr(noise_scheduler, timesteps)
    mse_loss_weights = torch.stack(
        [snr, float(snr_gamma) * torch.ones_like(timesteps)],
        dim=1,
    ).min(dim=1)[0]
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        mse_loss_weights = mse_loss_weights / snr
    elif prediction_type == "v_prediction":
        mse_loss_weights = mse_loss_weights / (snr + 1)

    loss = torch_f.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, loss.ndim))) * mse_loss_weights
    return loss.mean()


def apply_prompt_dropout(captions: list[str], proportion_empty_prompts: float) -> list[str]:
    probability = float(proportion_empty_prompts)
    if probability <= 0:
        return list(captions)
    if probability > 1:
        raise ValueError("training.proportion_empty_prompts must be in the range [0, 1].")
    keep_or_drop = torch.rand(len(captions))
    return ["" if keep_or_drop[index].item() < probability else caption for index, caption in enumerate(captions)]


def compute_batch_sdxl_time_ids(
    batch: dict,
    num_frames: int,
    fallback_resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    original_sizes = batch.get("original_sizes")
    crop_top_lefts = batch.get("crop_top_lefts")
    target_sizes = batch.get("target_sizes")
    if original_sizes is None or crop_top_lefts is None or target_sizes is None:
        return compute_sdxl_time_ids(
            original_size=(fallback_resolution, fallback_resolution),
            crop_coords=(0, 0),
            target_size=(fallback_resolution, fallback_resolution),
            batch_size=len(batch["captions"]) * int(num_frames),
            device=device,
            dtype=dtype,
        )

    values = torch.cat([original_sizes, crop_top_lefts, target_sizes], dim=1)
    values = values.repeat_interleave(int(num_frames), dim=0)
    return values.to(device=device, dtype=dtype, non_blocking=True)


def move_video_frames_to_device(
    frames: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        frames = frames.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return frames.mul_(1.0 / 127.5).sub_(1.0)
    return frames.to(device=device, dtype=dtype, non_blocking=non_blocking)


def log_startup(
    accelerator: Accelerator,
    config: dict,
    torch_dtype: torch.dtype,
    temporal_config: dict,
    trainable_count: int,
    frozen_count: int,
) -> None:
    model_config = config["model"]
    data_config = config["data"]
    accelerator.print("SDXL frame generator startup")
    accelerator.print(f"  model: {model_config['pretrained_model_name_or_path']}")
    accelerator.print(f"  dtype: {torch_dtype}")
    accelerator.print(f"  resolution: {model_config['resolution']}")
    accelerator.print(f"  num_frames_per_video: {data_config['num_frames_per_video']}")
    accelerator.print(f"  temporal conditioning type: {temporal_config['type']}")
    accelerator.print(f"  temporal injection mode: {temporal_config['injection_mode']}")
    accelerator.print(f"  temporal alpha: {temporal_config['alpha']}")
    accelerator.print(
        f"  trainable parameters: {trainable_count} ({format_param_count(trainable_count)})"
    )
    accelerator.print(f"  frozen parameters: {frozen_count} ({format_param_count(frozen_count)})")
    accelerator.print(f"  distributed type: {accelerator.distributed_type}")
    accelerator.print(f"  num processes: {accelerator.num_processes}")
    accelerator.print(f"  process index: {accelerator.process_index}")
    accelerator.print(f"  local process index: {accelerator.local_process_index}")
    accelerator.print(f"  device: {accelerator.device}")


def tracker_is_enabled(config: dict) -> bool:
    report_to = str(config.get("logging", {}).get("report_to", "")).lower()
    return report_to not in {"", "none", "null", "false", "disabled"}


def get_report_to(config: dict) -> str | None:
    if not tracker_is_enabled(config):
        return None
    return str(config.get("logging", {}).get("report_to", "wandb"))


def wandb_is_enabled(config: dict) -> bool:
    report_to = str(config.get("logging", {}).get("report_to", "")).lower()
    return tracker_is_enabled(config) and "wandb" in {item.strip() for item in report_to.split(",")}


def init_experiment_trackers(
    accelerator: Accelerator,
    config: dict,
    temporal_config: dict,
    trainable_count: int,
    frozen_count: int,
) -> None:
    if not tracker_is_enabled(config):
        return

    logging_config = config.get("logging", {})
    project_name = (
        logging_config.get("project")
        or os.environ.get("WANDB_PROJECT")
        or "sdxl-frame-generator"
    )
    wandb_kwargs = {}
    for config_key, wandb_key in [
        ("entity", "entity"),
        ("run_name", "name"),
        ("group", "group"),
        ("job_type", "job_type"),
        ("tags", "tags"),
        ("notes", "notes"),
    ]:
        value = logging_config.get(config_key)
        if value is not None and value != "":
            wandb_kwargs[wandb_key] = value

    tracker_config = {
        "model": config.get("model", {}),
        "data": config.get("data", {}),
        "temporal_conditioning": temporal_config,
        "video_adapters": config.get("video_adapters", {}),
        "training": config.get("training", {}),
        "optimizer": config.get("optimizer", {}),
        "validation": config.get("validation", {}),
        "logging": logging_config,
        "distributed": {
            "distributed_type": str(accelerator.distributed_type),
            "num_processes": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
        },
        "parameters": {
            "trainable": trainable_count,
            "frozen": frozen_count,
        },
    }

    init_kwargs = {"wandb": wandb_kwargs} if wandb_kwargs else {}
    accelerator.init_trackers(
        project_name=project_name,
        config=tracker_config,
        init_kwargs=init_kwargs,
    )


def maybe_watch_with_wandb(accelerator: Accelerator, unet, temporal_mlp, config: dict) -> None:
    logging_config = config.get("logging", {})
    if not tracker_is_enabled(config) or not logging_config.get("wandb_watch", False):
        return
    if not accelerator.is_main_process:
        return
    try:
        tracker = accelerator.get_tracker("wandb").tracker
    except Exception as exc:
        accelerator.print(f"Could not attach wandb.watch: {exc}")
        return

    tracker.watch(
        [accelerator.unwrap_model(unet), accelerator.unwrap_model(temporal_mlp)],
        log=logging_config.get("wandb_watch_log", "gradients"),
        log_freq=int(logging_config.get("wandb_watch_log_freq", 100)),
    )


def log_validation_media(
    accelerator: Accelerator,
    config: dict,
    validation_dir: Path,
    global_step: int,
) -> None:
    if not config.get("logging", {}).get("log_validation_media", True):
        return
    log_media_directory(
        accelerator=accelerator,
        config=config,
        media_dir=validation_dir,
        global_step=global_step,
        prefix="validation",
    )


def log_media_directory(
    accelerator: Accelerator,
    config: dict,
    media_dir: Path,
    global_step: int,
    prefix: str,
    caption: str | None = None,
    fps: int = 8,
) -> None:
    if not wandb_is_enabled(config):
        return
    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except ImportError:
        accelerator.print(f"wandb is not installed; skipped {prefix} media logging.")
        return

    payload = {}
    media_dir = Path(media_dir)
    grid_path = media_dir / "grid.png"
    video_path = media_dir / "video.mp4"
    if grid_path.exists():
        payload[f"{prefix}/grid"] = wandb.Image(
            str(grid_path),
            caption=caption or f"step {global_step}",
        )
    if video_path.exists():
        payload[f"{prefix}/video"] = wandb.Video(
            str(video_path),
            fps=fps,
            format="mp4",
            caption=caption,
        )
    if payload:
        accelerator.log(payload, step=global_step)


def config_float_list(value, default: list[float] | None = None) -> list[float]:
    if value is None:
        return list(default or [])
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    result = []
    for item in value:
        if isinstance(item, str) and "," in item:
            result.extend(config_float_list(item))
        else:
            result.append(float(item))
    return result


def _set_eval_temporarily(*modules) -> list[tuple[torch.nn.Module, bool]]:
    states = []
    for module in modules:
        if module is None:
            continue
        states.append((module, bool(module.training)))
        module.eval()
    return states


def _restore_training_states(states: list[tuple[torch.nn.Module, bool]]) -> None:
    for module, was_training in states:
        module.train(was_training)


def should_log_training_caption_inference(config: dict, global_step: int) -> bool:
    if not wandb_is_enabled(config):
        return False
    logging_config = config.get("logging", {})
    if not logging_config.get("log_training_caption_inference", False):
        return False
    interval = int(logging_config.get("training_caption_inference_steps", 0) or 0)
    return interval > 0 and global_step > 0 and global_step % interval == 0


def log_training_caption_inference(
    accelerator: Accelerator,
    config: dict,
    pipe,
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    temporal_mlp,
    frame_position_encoder,
    prompt: str,
    global_step: int,
    output_dir: Path,
    resolution: int,
    temporal_alpha: float,
    injection_mode: str,
    frame_encoder_config: dict,
) -> None:
    if not accelerator.is_main_process:
        return

    logging_config = config.get("logging", {})
    media_root = output_dir / (
        logging_config.get("training_caption_output_dir")
        or logging_config.get("training_caption_inference_output_dir")
        or "training_caption_samples"
    )
    media_dir = media_root / f"step_{global_step:04d}"
    num_frames = int(
        logging_config.get(
            "training_caption_num_frames",
            config.get("data", {}).get("num_frames_per_video", 8),
        )
    )
    seed = logging_config.get("training_caption_seed")
    if seed is None and config.get("training", {}).get("seed") is not None:
        seed = int(config["training"]["seed"]) + int(global_step)
    generation_resolution = logging_config.get("training_caption_resolution") or resolution

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_vae = accelerator.unwrap_model(vae)
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    unwrapped_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
    unwrapped_temporal_mlp = accelerator.unwrap_model(temporal_mlp)
    unwrapped_frame_position_encoder = (
        accelerator.unwrap_model(frame_position_encoder)
        if frame_position_encoder is not None
        else None
    )

    module_states = _set_eval_temporarily(
        unwrapped_unet,
        unwrapped_vae,
        unwrapped_text_encoder,
        unwrapped_text_encoder_2,
        unwrapped_temporal_mlp,
        unwrapped_frame_position_encoder,
    )
    try:
        pipe.unet = unwrapped_unet
        pipe.vae = unwrapped_vae
        pipe.text_encoder = unwrapped_text_encoder
        pipe.text_encoder_2 = unwrapped_text_encoder_2
        pipe.to(accelerator.device)
        generate_video_frames(
            pipe=pipe,
            temporal_mlp=unwrapped_temporal_mlp,
            frame_position_encoder=unwrapped_frame_position_encoder,
            prompt=prompt,
            num_frames=num_frames,
            output_dir=media_dir,
            resolution=int(generation_resolution),
            temporal_alpha=temporal_alpha,
            injection_mode=injection_mode,
            frame_token_embedding_mode=frame_encoder_config["token_embedding_mode"],
            frame_token_alpha=float(frame_encoder_config.get("alpha", 1.0)),
            num_inference_steps=int(logging_config.get("training_caption_num_inference_steps", 20)),
            guidance_scale=float(logging_config.get("training_caption_guidance_scale", 7.5)),
            seed=seed,
            batch_size=logging_config.get("training_caption_batch_size"),
            save_grid=bool(logging_config.get("training_caption_save_grid", True)),
            save_video=bool(logging_config.get("training_caption_save_mp4", False)),
        )
        log_media_directory(
            accelerator=accelerator,
            config=config,
            media_dir=media_dir,
            global_step=global_step,
            prefix="train_caption",
            caption=prompt,
            fps=int(logging_config.get("training_caption_fps", 8)),
        )
        accelerator.log(
            {
                "train_caption/generated_frames": num_frames,
                "train_caption/prompt_length": len(prompt),
            },
            step=global_step,
        )
    finally:
        _restore_training_states(module_states)


def get_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def write_status(accelerator: Accelerator, message: str, progress_bar=None) -> None:
    if not accelerator.is_main_process:
        return
    if progress_bar is not None:
        progress_bar.write(message)
    else:
        accelerator.print(message)


def log_first_step_shapes(
    accelerator: Accelerator,
    shapes: dict[str, object],
    progress_bar=None,
) -> None:
    lines = ["First training-step tensor shapes"]
    lines.extend(f"  {key}: {value}" for key, value in shapes.items())
    write_status(accelerator, "\n".join(lines), progress_bar=progress_bar)


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def find_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> Path | None:
    if resume_from_checkpoint in {None, "", "none", "null", "false"}:
        return None
    if resume_from_checkpoint == "latest":
        checkpoints = [
            path for path in output_dir.iterdir()
            if path.is_dir() and _checkpoint_step(path) is not None
        ] if output_dir.exists() else []
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: _checkpoint_step(path) or -1)
    path = Path(str(resume_from_checkpoint))
    if not path.is_absolute():
        path = output_dir / path
    return path


def read_checkpoint_step(checkpoint_dir: Path) -> int:
    parsed = _checkpoint_step(checkpoint_dir)
    if parsed is not None:
        return parsed
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return int(state.get("global_step", 0))
    temporal_path = checkpoint_dir / "temporal_mlp.pt"
    if temporal_path.exists():
        payload = torch.load(temporal_path, map_location="cpu")
        return int(payload.get("global_step", 0))
    return 0


def rotate_numbered_checkpoints(output_dir: Path, total_limit: int | None) -> None:
    if total_limit is None or int(total_limit) <= 0:
        return
    checkpoints = [
        path for path in output_dir.iterdir()
        if path.is_dir() and _checkpoint_step(path) is not None
    ]
    checkpoints = sorted(checkpoints, key=lambda path: _checkpoint_step(path) or -1)
    excess = len(checkpoints) - int(total_limit)
    for checkpoint_dir in checkpoints[: max(0, excess)]:
        shutil.rmtree(checkpoint_dir)


def save_training_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    step: int,
    unet,
    vae,
    temporal_mlp,
    frame_position_encoder,
    latent_calibrator,
    config: dict,
    temporal_config: dict,
    progress_bar=None,
) -> None:
    save_checkpoint(
        output_dir=output_dir,
        step=step,
        unet=unet,
        vae=vae,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        latent_calibrator=latent_calibrator,
        config=config,
        temporal_config=temporal_config,
        accelerator=accelerator,
    )
    accelerator.wait_for_everyone()

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    state_dir = checkpoint_dir / "accelerator_state"
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump({"global_step": int(step)}, handle, indent=2)
        last_dir = output_dir / "checkpoint-last"
        if last_dir.exists():
            shutil.rmtree(last_dir)
        shutil.copytree(checkpoint_dir, last_dir)
        rotate_numbered_checkpoints(
            output_dir,
            config.get("training", {}).get("checkpoints_total_limit"),
        )
    accelerator.wait_for_everyone()
    write_status(accelerator, f"saved checkpoint at step {step}", progress_bar=progress_bar)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    project_root = PROJECT_ROOT
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = project_root / env_file
    load_env_file(env_file, override=True)

    training_config = config["training"]
    model_config = config["model"]
    model_config.setdefault("pretrained_vae_model_name_or_path", None)
    model_config.setdefault("revision", None)
    model_config.setdefault("variant", None)
    model_config.setdefault("vae_dtype", "fp32")
    data_config_for_defaults = config.setdefault("data", {})
    data_config_for_defaults.setdefault("center_crop", False)
    data_config_for_defaults.setdefault("random_flip", False)
    data_config_for_defaults.setdefault("image_interpolation_mode", "lanczos")
    training_config.setdefault("num_train_epochs", 100)
    training_config.setdefault("max_train_steps", None)
    training_config.setdefault("checkpointing_steps", 500)
    training_config.setdefault("checkpoints_total_limit", None)
    training_config.setdefault("resume_from_checkpoint", None)
    training_config.setdefault("init_from_checkpoint", None)
    training_config.setdefault("gradient_accumulation_steps", 1)
    training_config.setdefault("gradient_checkpointing", False)
    training_config.setdefault("learning_rate", 1.0e-4)
    training_config.setdefault("scale_lr", False)
    training_config.setdefault("lr_scheduler", "constant")
    training_config.setdefault("lr_warmup_steps", 500)
    training_config.setdefault("timestep_bias_strategy", "none")
    training_config.setdefault("timestep_bias_multiplier", 1.0)
    training_config.setdefault("timestep_bias_begin", 0)
    training_config.setdefault("timestep_bias_end", 1000)
    training_config.setdefault("timestep_bias_portion", 0.25)
    training_config.setdefault("snr_gamma", None)
    training_config.setdefault("allow_tf32", False)
    training_config.setdefault("enable_xformers_memory_efficient_attention", False)
    training_config.setdefault("use_8bit_adam", False)
    training_config.setdefault("max_grad_norm", 1.0)
    training_config.setdefault("prediction_type", None)
    training_config.setdefault("noise_offset", 0.0)
    training_config.setdefault("latent_init_mode", LATENT_INIT_VIDEO_GT)
    training_config.setdefault("image_first_noise_mode", IMAGE_FIRST_NOISE_INDEPENDENT)
    training_config.setdefault("image_first_shared_noise_prob", 0.5)
    training_config.setdefault("image_first_bridge_mode", IMAGE_FIRST_BRIDGE_ALWAYS)
    training_config.setdefault("image_first_bridge_snr_min", None)
    training_config.setdefault("image_first_bridge_snr_max", None)
    training_config.setdefault("image_first_bridge_snr_full", 1.0)
    training_config.setdefault("image_first_bridge_snr_zero", 5.0)
    training_config.setdefault("image_first_bridge_gate", "cosine")
    training_config.setdefault("image_first_bridge_gate_domain", "log_snr")
    training_config.setdefault("image_first_boundary_loss_weight", 0.0)
    training_config.setdefault("image_first_boundary_snr", None)
    training_config.setdefault("image_first_boundary_lowfreq_only", True)
    training_config.setdefault("image_first_boundary_downsample_factor", 4)
    training_config.setdefault("image_first_smooth_anchor_source", "clean_first_frame")
    training_config.setdefault("image_first_rollout_source_timestep", 600)
    training_config.setdefault("image_first_rollout_source_steps", 4)
    training_config.setdefault("image_first_rollout_steps", 0)
    training_config.setdefault("image_first_rollout_noise_offset", 0.0)
    training_config.setdefault("image_first_rollout_switch_noise_scale", 0.0)
    training_config.setdefault("proportion_empty_prompts", 0.0)
    training_config.setdefault("logging_dir", "logs")
    temporal_config = dict(config["temporal_conditioning"])
    video_adapter_config = config.setdefault("video_adapters", {})
    latent_calibrator_config = dict(config.setdefault("latent_calibrator", {}))
    latent_calibrator_config.setdefault("enabled", False)
    latent_calibrator_config.setdefault("train", True)
    latent_calibrator_config.setdefault("apply_mode", "switch_only")
    latent_calibrator_config.setdefault("architecture", "temporal_conv")
    latent_calibrator_config.setdefault("latent_channels", 4)
    latent_calibrator_config.setdefault("hidden_channels", 64)
    latent_calibrator_config.setdefault("num_blocks", 2)
    latent_calibrator_config.setdefault("zero_init_output", True)
    latent_calibrator_config.setdefault("timestep_embedding_dim", 256)
    latent_calibrator_config.setdefault("prompt_embedding_dim", 1280)
    latent_calibrator_config.setdefault("conditioning", {})
    latent_calibrator_config["conditioning"].setdefault("use_timestep", True)
    latent_calibrator_config["conditioning"].setdefault("use_frame_pos", True)
    latent_calibrator_config["conditioning"].setdefault("use_pooled_prompt", True)
    latent_calibrator_config["conditioning"].setdefault("use_anchor_latent", True)
    latent_calibrator_config["conditioning"].setdefault("input_mode", "concat_anchor_delta")
    latent_calibrator_config.setdefault("residual", {})
    latent_calibrator_config["residual"].setdefault("scale_mode", "clipped_snr")
    latent_calibrator_config["residual"].setdefault("max_scale", 0.5)
    latent_calibrator_config["residual"].setdefault("norm_cap", 1.0)
    latent_calibrator_config.setdefault("auxiliary_loss", {})
    latent_calibrator_config["auxiliary_loss"].setdefault("map_weight", 0.0)
    latent_calibrator_config["auxiliary_loss"].setdefault("map_lowfreq_only", True)
    latent_calibrator_config["auxiliary_loss"].setdefault("map_downsample_factor", 4)
    latent_calibrator_config["auxiliary_loss"].setdefault("norm_weight", 0.0)
    config["latent_calibrator"] = latent_calibrator_config
    frame_encoder_config = dict(video_adapter_config.get("frame_position_encoder", {}))
    frame_encoder_config.setdefault("enabled", False)
    frame_encoder_config.setdefault("type", "sinusoidal")
    frame_encoder_config.setdefault("train", True)
    frame_encoder_config.setdefault("embedding_dim", 2048)
    frame_encoder_config.setdefault("hidden_dim", 1024)
    frame_encoder_config.setdefault("num_layers", 2)
    frame_encoder_config.setdefault("num_tokens", 1)
    frame_encoder_config.setdefault("pooling", "mean")
    frame_encoder_config.setdefault("token_embedding_mode", "add_to_text")
    frame_encoder_config.setdefault("alpha", 1.0)
    frame_encoder_config["token_embedding_mode"] = normalize_frame_token_mode(
        frame_encoder_config["token_embedding_mode"]
    )
    video_adapter_config["frame_position_encoder"] = frame_encoder_config
    video_resnet_config = VideoResnetAdapterConfig.from_config(
        video_adapter_config.get("resnet")
    )
    vae_decoder_resnet_config = VideoResnetAdapterConfig.from_config(
        video_adapter_config.get("vae_decoder_resnet")
    )
    if frame_encoder_config["enabled"]:
        video_resnet_config.frame_embedding_dim = int(frame_encoder_config["embedding_dim"])
        vae_decoder_resnet_config.frame_embedding_dim = int(frame_encoder_config["embedding_dim"])
        video_adapter_config.setdefault("resnet", {})["frame_embedding_dim"] = int(
            frame_encoder_config["embedding_dim"]
        )
        video_adapter_config.setdefault("vae_decoder_resnet", {})["frame_embedding_dim"] = int(
            frame_encoder_config["embedding_dim"]
        )
    video_attention_config = VideoAttentionAdapterConfig.from_config(
        video_adapter_config.get("attention")
    )
    image_first_anchor_enabled = bool(video_attention_config.anchor_enabled)
    torch_dtype = get_torch_dtype(model_config.get("dtype", "bf16"))
    vae_dtype = get_torch_dtype(model_config.get("vae_dtype", "fp32"))
    output_dir = as_project_path(training_config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / str(training_config.get("logging_dir", "logs"))
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        mixed_precision=training_config.get("mixed_precision", "no"),
        log_with=get_report_to(config),
        project_config=accelerator_project_config,
    )
    if training_config.get("seed") is not None:
        set_seed(int(training_config["seed"]))
    if bool(training_config.get("allow_tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        save_config(config, output_dir / "config.yaml")

    pipe = maybe_load_pipeline(model_config, torch_dtype)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    original_pipeline_vae = pipe.vae
    vae = load_sdxl_vae(model_config, vae_dtype)
    pipe.vae = vae
    del original_pipeline_vae
    unet = pipe.unet
    vae_decoder = getattr(vae, "decoder", vae)
    video_resnet_blocks = inject_video_resnet_adapters(unet, video_resnet_config)
    vae_decoder_resnet_blocks = inject_video_resnet_adapters(
        vae_decoder,
        vae_decoder_resnet_config,
    )
    video_attention_blocks = inject_video_attention_adapters(unet, video_attention_config)
    sync_video_resnet_adapter_device_dtype(unet)
    sync_video_resnet_adapter_device_dtype(vae_decoder)
    sync_video_attention_adapter_device_dtype(unet)

    noise_scheduler = load_noise_scheduler(model_config)
    if training_config.get("prediction_type") is not None:
        noise_scheduler.register_to_config(prediction_type=training_config["prediction_type"])
    latent_init_mode = normalize_latent_init_mode(training_config.get("latent_init_mode"))
    training_config["latent_init_mode"] = latent_init_mode
    image_first_noise_mode = normalize_image_first_noise_mode(
        training_config.get("image_first_noise_mode")
    )
    training_config["image_first_noise_mode"] = image_first_noise_mode
    image_first_bridge_mode = normalize_image_first_bridge_mode(
        training_config.get("image_first_bridge_mode")
    )
    training_config["image_first_bridge_mode"] = image_first_bridge_mode
    image_first_uses_rollout_bridge = image_first_bridge_mode in {
        IMAGE_FIRST_BRIDGE_ROLLOUT,
        IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
    }
    image_first_uses_snr_gate = image_first_bridge_mode in {
        IMAGE_FIRST_BRIDGE_SNR,
        IMAGE_FIRST_BRIDGE_ROLLOUT_SNR,
    }
    image_first_uses_smooth_snr_bridge = image_first_bridge_mode == IMAGE_FIRST_BRIDGE_SMOOTH_SNR
    image_first_smooth_anchor_source = str(
        training_config.get("image_first_smooth_anchor_source", "clean_first_frame")
    )
    if image_first_smooth_anchor_source not in {"clean_first_frame", "rollout_pred_x0"}:
        raise ValueError(
            "training.image_first_smooth_anchor_source must be 'clean_first_frame' or 'rollout_pred_x0'."
        )
    image_first_smooth_uses_rollout_source = (
        image_first_uses_smooth_snr_bridge
        and image_first_smooth_anchor_source == "rollout_pred_x0"
    )
    image_first_rollout_source_timestep = int(training_config.get("image_first_rollout_source_timestep", 600))
    image_first_rollout_source_steps = int(training_config.get("image_first_rollout_source_steps", 4))
    image_first_shared_noise_prob = float(training_config.get("image_first_shared_noise_prob", 0.5))
    if not 0.0 <= image_first_shared_noise_prob <= 1.0:
        raise ValueError("training.image_first_shared_noise_prob must be in [0, 1].")
    image_first_bridge_snr_min = _optional_float(training_config.get("image_first_bridge_snr_min"))
    image_first_bridge_snr_max = _optional_float(training_config.get("image_first_bridge_snr_max"))
    if (
        image_first_bridge_snr_min is not None
        and image_first_bridge_snr_max is not None
        and image_first_bridge_snr_min > image_first_bridge_snr_max
    ):
        raise ValueError("training.image_first_bridge_snr_min must be <= image_first_bridge_snr_max.")
    image_first_bridge_snr_full = _optional_float(training_config.get("image_first_bridge_snr_full"))
    image_first_bridge_snr_zero = _optional_float(training_config.get("image_first_bridge_snr_zero"))
    image_first_bridge_gate = str(training_config.get("image_first_bridge_gate", "cosine"))
    image_first_bridge_gate_domain = str(training_config.get("image_first_bridge_gate_domain", "log_snr"))
    if image_first_uses_smooth_snr_bridge:
        if image_first_bridge_snr_full is None or image_first_bridge_snr_zero is None:
            raise ValueError(
                "training.image_first_bridge_snr_full and image_first_bridge_snr_zero are required "
                "for image_first_bridge_mode='smooth_snr'."
            )
        if image_first_bridge_snr_full >= image_first_bridge_snr_zero:
            raise ValueError(
                "training.image_first_bridge_snr_full must be < image_first_bridge_snr_zero."
            )
        compute_image_first_smooth_snr_gate(
            noise_scheduler=noise_scheduler,
            timesteps=torch.tensor([0], dtype=torch.long),
            snr_full=image_first_bridge_snr_full,
            snr_zero=image_first_bridge_snr_zero,
            gate_mode=image_first_bridge_gate,
            domain=image_first_bridge_gate_domain,
        )
    image_first_boundary_loss_weight = float(training_config.get("image_first_boundary_loss_weight", 0.0))
    if image_first_boundary_loss_weight < 0.0:
        raise ValueError("training.image_first_boundary_loss_weight must be non-negative.")
    image_first_boundary_snr = _optional_float(training_config.get("image_first_boundary_snr"))
    if image_first_boundary_snr is None:
        image_first_boundary_snr = image_first_bridge_snr_zero
    if image_first_boundary_loss_weight > 0.0 and image_first_boundary_snr is None:
        raise ValueError("training.image_first_boundary_snr is required when boundary loss is enabled.")
    image_first_boundary_lowfreq_only = bool(training_config.get("image_first_boundary_lowfreq_only", True))
    image_first_boundary_downsample_factor = int(
        training_config.get("image_first_boundary_downsample_factor", 4)
    )
    if image_first_boundary_downsample_factor < 1:
        raise ValueError("training.image_first_boundary_downsample_factor must be >= 1.")
    image_first_boundary_timestep = None
    if image_first_boundary_loss_weight > 0.0:
        image_first_boundary_timestep = timestep_closest_to_snr(
            noise_scheduler=noise_scheduler,
            target_snr=float(image_first_boundary_snr),
            device=accelerator.device,
        )
    image_first_rollout_steps = int(training_config.get("image_first_rollout_steps", 0))
    if image_first_rollout_steps < 0:
        raise ValueError("training.image_first_rollout_steps must be non-negative.")
    image_first_rollout_switch_noise_scale = float(
        training_config.get("image_first_rollout_switch_noise_scale", 0.0)
    )
    if image_first_rollout_switch_noise_scale < 0:
        raise ValueError("training.image_first_rollout_switch_noise_scale must be non-negative.")
    if (
        latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
        and getattr(noise_scheduler.config, "prediction_type", "epsilon") != "epsilon"
    ):
        raise ValueError(
            "training.latent_init_mode='first_frame_repeat' currently requires "
            "scheduler prediction_type='epsilon'."
        )

    train_temporal_embedding = bool(training_config.get("train_temporal_embedding", True))
    train_unet = bool(training_config.get("train_unet", False))
    train_video_resnet_adapters = video_resnet_config.enabled and video_resnet_config.train
    train_vae_decoder_resnet_adapters = (
        vae_decoder_resnet_config.enabled and vae_decoder_resnet_config.train
    )
    train_video_attention_adapters = video_attention_config.enabled and video_attention_config.train
    train_frame_position_encoder = bool(
        frame_encoder_config["enabled"] and frame_encoder_config.get("train", True)
    )
    train_latent_calibrator = bool(
        latent_calibrator_config["enabled"] and latent_calibrator_config.get("train", True)
    )
    train_text_encoder = bool(training_config.get("train_text_encoder", False))
    train_vae = bool(training_config.get("train_vae", False))

    set_requires_grad(unet, train_unet)
    if video_resnet_config.enabled:
        set_video_resnet_adapter_requires_grad(unet, train_video_resnet_adapters)
        set_video_resnet_adapters_active(unet, video_resnet_config.active)
    if video_attention_config.enabled:
        set_video_attention_adapter_requires_grad(unet, train_video_attention_adapters)
        set_video_attention_adapters_active(unet, video_attention_config.active)
    set_requires_grad(text_encoder, train_text_encoder)
    set_requires_grad(text_encoder_2, train_text_encoder)
    set_requires_grad(vae, train_vae)
    if vae_decoder_resnet_config.enabled:
        set_video_resnet_adapter_requires_grad(vae_decoder, train_vae_decoder_resnet_adapters)
        set_video_resnet_adapters_active(vae_decoder, vae_decoder_resnet_config.active)

    if train_unet or train_video_resnet_adapters or train_video_attention_adapters:
        unet.train()
    else:
        unet.eval()
    if train_text_encoder:
        text_encoder.train()
        text_encoder_2.train()
    else:
        text_encoder.eval()
        text_encoder_2.eval()
    if train_vae:
        vae.train()
    else:
        vae.eval()

    if bool(training_config.get("gradient_checkpointing", False)):
        unet.enable_gradient_checkpointing()
    if bool(training_config.get("enable_xformers_memory_efficient_attention", False)):
        if not is_xformers_available():
            raise ValueError(
                "training.enable_xformers_memory_efficient_attention=true requires xformers."
            )
        unet.enable_xformers_memory_efficient_attention()

    device = accelerator.device
    if not train_unet:
        unet.to(device=device, dtype=torch_dtype)
    text_encoder.to(device=device, dtype=torch_dtype)
    text_encoder_2.to(device=device, dtype=torch_dtype)
    vae.to(device=device, dtype=vae_dtype)

    with torch.no_grad():
        _, pooled_for_dim = encode_prompts_with_sdxl_logic(
            [config["data"].get("placeholder_caption", "")],
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            device=device,
            dtype=torch_dtype,
        )
    inferred_output_dim = int(pooled_for_dim.shape[-1])
    temporal_config["output_dim"] = temporal_config.get("output_dim") or inferred_output_dim
    latent_calibrator_config["prompt_embedding_dim"] = (
        latent_calibrator_config.get("prompt_embedding_dim") or inferred_output_dim
    )
    if int(latent_calibrator_config["prompt_embedding_dim"]) != inferred_output_dim:
        raise ValueError(
            "latent_calibrator.prompt_embedding_dim must match SDXL pooled prompt dim "
            f"({latent_calibrator_config['prompt_embedding_dim']} vs {inferred_output_dim})."
        )
    if int(temporal_config["output_dim"]) != inferred_output_dim:
        raise ValueError(
            "temporal_conditioning.output_dim must match SDXL pooled prompt dim "
            f"({temporal_config['output_dim']} vs {inferred_output_dim})."
        )
    validate_sdxl_added_conditioning_dimensions(unet, text_encoder_2)

    temporal_mlp = FramePositionMLP(
        output_dim=int(temporal_config["output_dim"]),
        hidden_dim=int(temporal_config["hidden_dim"]),
        num_layers=int(temporal_config["num_layers"]),
    )
    frame_position_encoder = None
    if frame_encoder_config["enabled"]:
        frame_position_encoder = build_frame_position_encoder(frame_encoder_config)
        if not any(True for _ in frame_position_encoder.parameters()):
            train_frame_position_encoder = False
    latent_calibrator = build_latent_calibrator(latent_calibrator_config)
    set_requires_grad(temporal_mlp, train_temporal_embedding)
    set_requires_grad(frame_position_encoder, train_frame_position_encoder)
    set_requires_grad(latent_calibrator, train_latent_calibrator)
    temporal_mlp.train(train_temporal_embedding)
    if frame_position_encoder is not None:
        frame_position_encoder.train(train_frame_position_encoder)
    if latent_calibrator is not None:
        latent_calibrator.train(train_latent_calibrator)
    if not train_temporal_embedding:
        temporal_mlp.to(device=device, dtype=torch_dtype)
    if frame_position_encoder is not None and not train_frame_position_encoder:
        if any(True for _ in frame_position_encoder.parameters()):
            frame_position_encoder.to(device=device, dtype=torch_dtype)
        else:
            frame_position_encoder.to(device=device)
    if latent_calibrator is not None and not train_latent_calibrator:
        latent_calibrator.to(device=device, dtype=torch_dtype)

    # Weights-only warm start (distinct from resume_from_checkpoint, which restores
    # full accelerator state and requires an identical architecture). Used to seed
    # a new architecture (e.g. E2: anchor branch + mid+up placement) from a prior
    # run's adapter weights. strict=False so matching temporal weights load while
    # missing modules (anchor) stay at their fresh init and extra modules
    # (e.g. down-block adapters from a full-placement checkpoint) are ignored.
    init_from_checkpoint = training_config.get("init_from_checkpoint")
    if init_from_checkpoint and not training_config.get("resume_from_checkpoint"):
        init_dir = as_project_path(str(init_from_checkpoint), project_root)
        if not init_dir.exists():
            raise FileNotFoundError(f"init_from_checkpoint not found: {init_dir}")
        load_video_resnet_adapter_checkpoint(init_dir, unet, strict=False)
        load_video_attention_adapter_checkpoint(init_dir, unet, strict=False)
        temporal_mlp_payload = init_dir / "temporal_mlp.pt"
        if temporal_mlp_payload.exists():
            payload = torch.load(temporal_mlp_payload, map_location="cpu")
            temporal_mlp.load_state_dict(payload["state_dict"], strict=False)
        sync_video_resnet_adapter_device_dtype(unet)
        sync_video_attention_adapter_device_dtype(unet)
        write_status(accelerator, f"warm-started adapter weights from {init_dir} (strict=False)")

    trainable_count, frozen_count = count_trainable_and_frozen(
        [
            unet,
            vae,
            text_encoder,
            text_encoder_2,
            temporal_mlp,
            frame_position_encoder,
            latent_calibrator,
        ]
    )
    log_startup(accelerator, config, torch_dtype, temporal_config, trainable_count, frozen_count)
    accelerator.print(
        "  video resnet adapters: "
        f"enabled={video_resnet_config.enabled}, active={video_resnet_config.active}, "
        f"train={train_video_resnet_adapters}, blocks={video_resnet_blocks}, "
        f"frame_embedding_dim={video_resnet_config.frame_embedding_dim}"
    )
    accelerator.print(
        "  vae decoder resnet adapters: "
        f"enabled={vae_decoder_resnet_config.enabled}, active={vae_decoder_resnet_config.active}, "
        f"train={train_vae_decoder_resnet_adapters}, blocks={vae_decoder_resnet_blocks}, "
        f"frame_embedding_dim={vae_decoder_resnet_config.frame_embedding_dim}"
    )
    accelerator.print(
        "  video attention adapters: "
        f"enabled={video_attention_config.enabled}, active={video_attention_config.active}, "
        f"train={train_video_attention_adapters}, blocks={video_attention_blocks}"
    )
    accelerator.print(
        "  frame position encoder: "
        f"enabled={frame_encoder_config['enabled']}, train={train_frame_position_encoder}, "
        f"type={frame_encoder_config['type']}, tokens={frame_encoder_config['num_tokens']}, "
        f"dim={frame_encoder_config['embedding_dim']}, "
        f"token_mode={frame_encoder_config['token_embedding_mode']}, "
        f"alpha={frame_encoder_config['alpha']}"
    )
    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT:
        accelerator.print(
            "  image-first bridge: "
            f"mode={image_first_bridge_mode}, noise_mode={image_first_noise_mode}, "
            f"snr_min={image_first_bridge_snr_min}, snr_max={image_first_bridge_snr_max}, "
            f"smooth_full={image_first_bridge_snr_full}, smooth_zero={image_first_bridge_snr_zero}, "
            f"smooth_gate={image_first_bridge_gate}, boundary_weight={image_first_boundary_loss_weight}, "
            f"boundary_snr={image_first_boundary_snr}, "
            f"rollout_steps={image_first_rollout_steps}, "
            f"rollout_switch_noise_scale={image_first_rollout_switch_noise_scale}"
        )
    latent_calibrator_settings = LatentCalibratorConfig.from_config(latent_calibrator_config)
    accelerator.print(
        "  latent calibrator: "
        f"enabled={latent_calibrator_settings.enabled}, train={train_latent_calibrator}, "
        f"architecture={latent_calibrator_settings.architecture}, "
        f"hidden={latent_calibrator_settings.hidden_channels}, "
        f"blocks={latent_calibrator_settings.num_blocks}, "
        f"scale={latent_calibrator_settings.residual_scale_mode}, "
        f"map_weight={latent_calibrator_settings.map_weight}, "
        f"norm_weight={latent_calibrator_settings.norm_weight}"
    )
    init_experiment_trackers(
        accelerator=accelerator,
        config=config,
        temporal_config=temporal_config,
        trainable_count=trainable_count,
        frozen_count=frozen_count,
    )
    if tracker_is_enabled(config):
        accelerator.log(
            {
                "params/trainable": trainable_count,
                "params/frozen": frozen_count,
                "distributed/num_processes": accelerator.num_processes,
            },
            step=0,
        )

    dataset = build_dataset(config)
    data_config = config["data"]
    torch_sharing_strategy = data_config.get("torch_multiprocessing_sharing_strategy")
    if torch_sharing_strategy is None:
        torch_sharing_strategy = os.environ.get("TORCH_MP_SHARING_STRATEGY")
    if torch_sharing_strategy is not None:
        torch_sharing_strategy = str(torch_sharing_strategy).strip()
    if torch_sharing_strategy and torch_sharing_strategy.lower() not in {"none", "null", "default"}:
        torch.multiprocessing.set_sharing_strategy(torch_sharing_strategy)
    dataloader_num_workers = int(data_config.get("num_workers", 0))
    dataloader_pin_memory = bool(data_config.get("pin_memory", torch.cuda.is_available()))
    dataloader_kwargs = {
        "batch_size": int(training_config["train_batch_size"]),
        "shuffle": True,
        "num_workers": dataloader_num_workers,
        "collate_fn": video_collate_fn,
        "drop_last": True,
        "pin_memory": dataloader_pin_memory,
    }
    dataloader_timeout = float(data_config.get("dataloader_timeout", 0) or 0)
    if dataloader_timeout > 0:
        dataloader_kwargs["timeout"] = dataloader_timeout
    if dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(data_config.get("persistent_workers", True))
        dataloader_kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    accelerator.print(
        "  dataloader: "
        f"workers={dataloader_num_workers}, pin_memory={dataloader_pin_memory}, "
        f"persistent_workers={dataloader_kwargs.get('persistent_workers', False)}, "
        f"prefetch_factor={dataloader_kwargs.get('prefetch_factor', None)}, "
        f"timeout={dataloader_kwargs.get('timeout', 0)}, "
        f"torch_mp_sharing={torch.multiprocessing.get_sharing_strategy()}"
    )
    gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    num_update_steps_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    overrode_max_train_steps = training_config.get("max_train_steps") is None
    if overrode_max_train_steps:
        training_config["max_train_steps"] = int(training_config["num_train_epochs"]) * num_update_steps_per_epoch
    max_train_steps = int(training_config["max_train_steps"])
    training_config["num_train_epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if bool(training_config.get("scale_lr", False)):
        training_config["learning_rate"] = (
            float(training_config["learning_rate"])
            * gradient_accumulation_steps
            * int(training_config["train_batch_size"])
            * accelerator.num_processes
        )

    parameters = [
        parameter
        for module in [unet, vae, text_encoder, text_encoder_2, temporal_mlp, latent_calibrator]
        + ([frame_position_encoder] if frame_position_encoder is not None else [])
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("No trainable parameters. Enable at least one training flag.")
    optimizer = build_optimizer(parameters, config)
    lr_scheduler = build_lr_scheduler(optimizer, training_config, max_train_steps)

    named_models = []
    if train_unet or train_video_resnet_adapters or train_video_attention_adapters:
        named_models.append(("unet", unet))
    if train_temporal_embedding:
        named_models.append(("temporal_mlp", temporal_mlp))
    if train_frame_position_encoder and frame_position_encoder is not None:
        named_models.append(("frame_position_encoder", frame_position_encoder))
    if train_latent_calibrator and latent_calibrator is not None:
        named_models.append(("latent_calibrator", latent_calibrator))
    if train_text_encoder:
        named_models.extend([("text_encoder", text_encoder), ("text_encoder_2", text_encoder_2)])
    if train_vae or train_vae_decoder_resnet_adapters:
        named_models.append(("vae", vae))

    prepared_models, optimizer, dataloader, lr_scheduler = prepare_trainable_models(
        accelerator,
        named_models,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    unet = prepared_models.get("unet", unet)
    temporal_mlp = prepared_models.get("temporal_mlp", temporal_mlp)
    frame_position_encoder = prepared_models.get("frame_position_encoder", frame_position_encoder)
    latent_calibrator = prepared_models.get("latent_calibrator", latent_calibrator)
    text_encoder = prepared_models.get("text_encoder", text_encoder)
    text_encoder_2 = prepared_models.get("text_encoder_2", text_encoder_2)
    vae = prepared_models.get("vae", vae)
    accumulation_models = [model for _, model in named_models]
    maybe_watch_with_wandb(accelerator, unet, temporal_mlp, config)

    global_step = 0
    logged_shapes = False
    num_update_steps_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    if overrode_max_train_steps:
        max_train_steps = int(training_config["num_train_epochs"]) * num_update_steps_per_epoch
        training_config["max_train_steps"] = max_train_steps
    training_config["num_train_epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)
    resolution = int(model_config["resolution"])
    temporal_alpha = float(temporal_config["alpha"])
    injection_mode = temporal_config["injection_mode"]
    logging_steps = int(training_config.get("logging_steps", 10))
    effective_batch_frames = (
        int(training_config["train_batch_size"])
        * int(config["data"]["num_frames_per_video"])
        * accelerator.num_processes
        * gradient_accumulation_steps
    )
    effective_batch_videos = (
        int(training_config["train_batch_size"])
        * accelerator.num_processes
        * gradient_accumulation_steps
    )
    accumulated_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    accumulated_loss_count = 0
    timestep_weights = None
    if str(training_config.get("timestep_bias_strategy", "none")).lower() != "none":
        timestep_weights = generate_timestep_weights(
            training_config,
            noise_scheduler.config.num_train_timesteps,
        )
    snr_gamma = training_config.get("snr_gamma")
    snr_gamma = None if snr_gamma is None else float(snr_gamma)

    resume_checkpoint = find_resume_checkpoint(output_dir, training_config.get("resume_from_checkpoint"))
    if resume_checkpoint is not None:
        state_dir = resume_checkpoint / "accelerator_state"
        if state_dir.exists():
            write_status(accelerator, f"resuming training state from {resume_checkpoint}")
            accelerator.load_state(str(state_dir))
            global_step = read_checkpoint_step(resume_checkpoint)
        else:
            write_status(
                accelerator,
                f"checkpoint {resume_checkpoint} has no accelerator_state; starting a fresh optimizer state",
            )

    progress_bar = None
    if accelerator.is_main_process and tqdm is not None:
        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            desc="train",
            dynamic_ncols=True,
            leave=True,
            smoothing=0.05,
        )
        progress_bar.set_postfix(
            {
                "loss": "n/a",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "frames": 0,
                "gpu": "n/a",
            }
        )

    try:
        while global_step < max_train_steps:
            for batch in dataloader:
                with accelerator.accumulate(*accumulation_models):
                    frames = move_video_frames_to_device(
                        batch["frames"],
                        device=accelerator.device,
                        dtype=vae_dtype,
                        non_blocking=dataloader_pin_memory,
                    )
                    frame_positions = batch["frame_positions"].to(
                        device=accelerator.device,
                        non_blocking=dataloader_pin_memory,
                    )
                    batch_size, num_frames = frames.shape[:2]
                    frames_flat = frames.flatten(0, 1)
                    frame_positions_flat = frame_positions.flatten(0, 1)

                    vae_context = contextlib.nullcontext() if train_vae else torch.no_grad()
                    with vae_context:
                        latents = vae.encode(frames_flat).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    clean_latents = latents
                    timesteps = sample_video_timesteps(
                        batch_size=batch_size,
                        num_frames=num_frames,
                        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                        device=latents.device,
                        timestep_weights=timestep_weights,
                    )
                    video_timesteps = timesteps.reshape(batch_size, num_frames)[:, 0]

                    text_context = contextlib.nullcontext() if train_text_encoder else torch.no_grad()
                    with text_context:
                        captions = apply_prompt_dropout(
                            batch["captions"],
                            float(training_config.get("proportion_empty_prompts", 0.0)),
                        )
                        prompt_embeds_image = None
                        pooled_prompt_embeds_image = None
                        captions_for_text = (
                            [caption for caption in captions for _ in range(num_frames)]
                            if train_text_encoder
                            else captions
                        )
                        prompt_embeds, pooled_prompt_embeds = encode_prompts_with_sdxl_logic(
                            captions_for_text,
                            tokenizer=tokenizer,
                            tokenizer_2=tokenizer_2,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            device=accelerator.device,
                            dtype=torch_dtype,
                        )
                        if not train_text_encoder:
                            prompt_embeds_image = prompt_embeds
                            pooled_prompt_embeds_image = pooled_prompt_embeds
                            prompt_embeds = prompt_embeds.repeat_interleave(num_frames, dim=0)
                            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_frames, dim=0)
                        elif (
                            latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                            and image_first_uses_rollout_bridge
                        ):
                            prompt_embeds_image, pooled_prompt_embeds_image = encode_prompts_with_sdxl_logic(
                                captions,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                device=accelerator.device,
                                dtype=torch_dtype,
                            )

                    modified_pooled_prompt_embeds, frame_embeds = (
                        add_temporal_embedding_to_pooled_prompt_embeds(
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            frame_positions=frame_positions_flat,
                            frame_position_mlp=temporal_mlp,
                            alpha=temporal_alpha,
                            injection_mode=injection_mode,
                        )
                    )
                    add_time_ids = compute_batch_sdxl_time_ids(
                        batch=batch,
                        num_frames=num_frames,
                        fallback_resolution=resolution,
                        device=accelerator.device,
                        dtype=pooled_prompt_embeds.dtype,
                    )

                    noising_latents = clean_latents
                    standard_noise = None
                    image_first_shared_noise_fraction = 0.0
                    image_first_bridge_fraction = 0.0
                    image_first_bridge_gate_values = None
                    bridge_mask = None
                    latent_calibrator_anchor_latents = None
                    latent_calibrator_alignment_target = None
                    latent_calibrator_mask = None
                    latent_calibrator_map_loss = clean_latents.new_zeros(())
                    latent_calibrator_norm_loss_value = clean_latents.new_zeros(())
                    image_first_boundary_loss_value = clean_latents.new_zeros(())
                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT:
                        if image_first_uses_rollout_bridge:
                            if prompt_embeds_image is None or pooled_prompt_embeds_image is None:
                                raise RuntimeError("Rollout source requires image-level prompt embeddings.")
                            anchor_latents = clean_latents.reshape(
                                batch_size,
                                num_frames,
                                *clean_latents.shape[1:],
                            )[:, 0]
                            rollout_add_time_ids = compute_batch_sdxl_time_ids(
                                batch=batch,
                                num_frames=1,
                                fallback_resolution=resolution,
                                device=accelerator.device,
                                dtype=pooled_prompt_embeds.dtype,
                            )
                            rollout_anchor_latents = rollout_image_first_anchor_latents(
                                unet=unet,
                                noise_scheduler=noise_scheduler,
                                anchor_latents=anchor_latents.to(dtype=torch_dtype),
                                target_timesteps=video_timesteps,
                                prompt_embeds=prompt_embeds_image.detach(),
                                pooled_prompt_embeds=pooled_prompt_embeds_image.detach(),
                                add_time_ids=rollout_add_time_ids,
                                rollout_steps=image_first_rollout_steps,
                                noise_offset=float(
                                    training_config.get("image_first_rollout_noise_offset", 0.0)
                                ),
                            )
                            noisy_latents = rollout_anchor_latents[:, None].expand(
                                -1,
                                num_frames,
                                -1,
                                -1,
                                -1,
                            ).reshape_as(clean_latents)
                            if image_first_rollout_switch_noise_scale > 0:
                                switch_level = (1.0 / (1.0 + compute_snr(noise_scheduler, video_timesteps))).sqrt()
                                while switch_level.ndim < rollout_anchor_latents.ndim:
                                    switch_level = switch_level.unsqueeze(-1)
                                switch_noise = torch.randn(
                                    (batch_size, num_frames, *rollout_anchor_latents.shape[1:]),
                                    device=rollout_anchor_latents.device,
                                    dtype=rollout_anchor_latents.dtype,
                                ).reshape_as(clean_latents)
                                switch_level = _expand_rollout_switch_level(
                                    switch_level=switch_level,
                                    batch_size=batch_size,
                                    num_frames=num_frames,
                                    reference=clean_latents,
                                )
                                noisy_latents = noisy_latents + (
                                    image_first_rollout_switch_noise_scale
                                    * switch_level.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
                                    * switch_noise
                                )
                            noisy_latents = noisy_latents.to(dtype=torch_dtype)
                            noising_latents = noisy_latents
                            if image_first_bridge_mode == IMAGE_FIRST_BRIDGE_ROLLOUT_SNR:
                                bridge_mask = compute_image_first_bridge_mask(
                                    noise_scheduler=noise_scheduler,
                                    timesteps=timesteps,
                                    bridge_mode=image_first_bridge_mode,
                                    snr_min=image_first_bridge_snr_min,
                                    snr_max=image_first_bridge_snr_max,
                                )
                                image_first_bridge_fraction = bridge_mask.float().mean()
                                standard_noise = torch.randn_like(clean_latents)
                                standard_noise = add_noise_offset(
                                    standard_noise,
                                    float(training_config.get("noise_offset", 0.0)),
                                )
                                standard_noisy_latents = noise_scheduler.add_noise(
                                    clean_latents,
                                    standard_noise,
                                    timesteps,
                                ).to(dtype=torch_dtype)
                                mask = _expand_latent_mask(bridge_mask, noisy_latents)
                                noisy_latents = torch.where(mask, noisy_latents, standard_noisy_latents)
                                noising_latents = noisy_latents
                            else:
                                image_first_bridge_fraction = 1.0
                        else:
                            if image_first_smooth_uses_rollout_source:
                                # E4: bridge source = rollout pred_x0 of the anchor image
                                # (x0-like, matches inference stage-1), repeated over frames,
                                # instead of the clean first-frame latent z*_1.
                                if prompt_embeds_image is None or pooled_prompt_embeds_image is None:
                                    raise RuntimeError(
                                        "rollout_pred_x0 source requires image-level prompt embeddings."
                                    )
                                anchor_single = clean_latents.reshape(
                                    batch_size, num_frames, *clean_latents.shape[1:]
                                )[:, 0]
                                rollout_source_time_ids = compute_batch_sdxl_time_ids(
                                    batch=batch,
                                    num_frames=1,
                                    fallback_resolution=resolution,
                                    device=accelerator.device,
                                    dtype=pooled_prompt_embeds.dtype,
                                )
                                rollout_pred_x0 = rollout_image_first_anchor_pred_x0(
                                    unet=unet,
                                    noise_scheduler=noise_scheduler,
                                    anchor_latents=anchor_single.to(dtype=torch_dtype),
                                    source_timestep=image_first_rollout_source_timestep,
                                    rollout_steps=image_first_rollout_source_steps,
                                    prompt_embeds=prompt_embeds_image.detach(),
                                    pooled_prompt_embeds=pooled_prompt_embeds_image.detach(),
                                    add_time_ids=rollout_source_time_ids,
                                    noise_offset=float(training_config.get("noise_offset", 0.0)),
                                )
                                anchor_noising_latents = rollout_pred_x0[:, None].expand(
                                    -1, num_frames, -1, -1, -1
                                ).reshape_as(clean_latents).to(dtype=clean_latents.dtype)
                            else:
                                anchor_noising_latents = repeat_first_frame_latents(
                                    clean_latents,
                                    batch_size=batch_size,
                                    num_frames=num_frames,
                                )
                            bridge_noise, image_first_shared_noise_fraction = sample_image_first_noise(
                                reference=anchor_noising_latents,
                                batch_size=batch_size,
                                num_frames=num_frames,
                                noise_mode=image_first_noise_mode,
                                shared_noise_prob=image_first_shared_noise_prob,
                                noise_offset=float(training_config.get("noise_offset", 0.0)),
                            )
                            bridge_mask = compute_image_first_bridge_mask(
                                noise_scheduler=noise_scheduler,
                                timesteps=timesteps,
                                bridge_mode=image_first_bridge_mode,
                                snr_min=image_first_bridge_snr_min,
                                snr_max=image_first_bridge_snr_max,
                            )
                            image_first_bridge_fraction = bridge_mask.float().mean()
                            if image_first_uses_smooth_snr_bridge:
                                if image_first_bridge_snr_full is None or image_first_bridge_snr_zero is None:
                                    raise RuntimeError("smooth_snr bridge thresholds were not initialized.")
                                image_first_bridge_gate_values = compute_image_first_smooth_snr_gate(
                                    noise_scheduler=noise_scheduler,
                                    timesteps=timesteps,
                                    snr_full=image_first_bridge_snr_full,
                                    snr_zero=image_first_bridge_snr_zero,
                                    gate_mode=image_first_bridge_gate,
                                    domain=image_first_bridge_gate_domain,
                                ).to(device=clean_latents.device, dtype=clean_latents.dtype)
                                image_first_bridge_fraction = (
                                    image_first_bridge_gate_values.detach().float().mean()
                                )
                                gate = image_first_bridge_gate_values
                                while gate.ndim < clean_latents.ndim:
                                    gate = gate.unsqueeze(-1)
                                noising_latents = (
                                    gate * anchor_noising_latents
                                    + (1.0 - gate) * clean_latents
                                )
                                noise = bridge_noise
                                bridge_mask = image_first_bridge_gate_values.detach() > 0.0
                            else:
                                mask = _expand_latent_mask(bridge_mask, clean_latents)
                                if image_first_bridge_mode == IMAGE_FIRST_BRIDGE_SNR:
                                    standard_noise = torch.randn_like(clean_latents)
                                    standard_noise = add_noise_offset(
                                        standard_noise,
                                        float(training_config.get("noise_offset", 0.0)),
                                    )
                                    noising_latents = torch.where(mask, anchor_noising_latents, clean_latents)
                                    noise = torch.where(mask, bridge_noise, standard_noise)
                                else:
                                    noising_latents = anchor_noising_latents
                                    noise = bridge_noise
                            noisy_latents = noise_scheduler.add_noise(
                                noising_latents,
                                noise,
                                timesteps,
                            ).to(dtype=torch_dtype)
                            latent_calibrator_anchor_latents = noising_latents.to(dtype=torch_dtype)
                            latent_calibrator_alignment_target = noise_scheduler.add_noise(
                                clean_latents,
                                bridge_noise,
                                timesteps,
                            ).to(dtype=torch_dtype)
                            latent_calibrator_mask = bridge_mask
                    else:
                        noise = torch.randn_like(noising_latents)
                        noise = add_noise_offset(noise, float(training_config.get("noise_offset", 0.0)))
                        noisy_latents = noise_scheduler.add_noise(noising_latents, noise, timesteps).to(dtype=torch_dtype)

                    if (
                        latent_calibrator is not None
                        and latent_calibrator_anchor_latents is not None
                        and latent_calibrator_mask is not None
                    ):
                        # gate_scaled: apply the calibrator everywhere (the gate inside
                        # scales the residual continuously, ~0 where g(t)->0) and weight
                        # the auxiliary losses by the soft gate. switch_only keeps the
                        # original hard-masked behavior.
                        calibrator_gate_scaled = (
                            latent_calibrator_settings.apply_mode == "gate_scaled"
                            and image_first_bridge_gate_values is not None
                        )
                        calibrated_latents, latent_calibrator_delta, _ = latent_calibrator(
                            noisy_latents=noisy_latents,
                            anchor_latents=latent_calibrator_anchor_latents,
                            timesteps=timesteps,
                            frame_positions=frame_positions_flat,
                            pooled_prompt_embeds=modified_pooled_prompt_embeds,
                            num_frames=num_frames,
                            noise_scheduler=noise_scheduler,
                            bridge_gate=(
                                image_first_bridge_gate_values if calibrator_gate_scaled else None
                            ),
                        )
                        if calibrator_gate_scaled:
                            noisy_latents = calibrated_latents
                            aux_weight = image_first_bridge_gate_values.detach()
                        else:
                            calibrator_mask = _expand_latent_mask(latent_calibrator_mask, noisy_latents)
                            noisy_latents = torch.where(calibrator_mask, calibrated_latents, noisy_latents)
                            aux_weight = latent_calibrator_mask
                        if latent_calibrator_alignment_target is not None:
                            latent_calibrator_map_loss = latent_calibrator_alignment_loss(
                                calibrated_latents=noisy_latents,
                                target_latents=latent_calibrator_alignment_target,
                                mask=aux_weight,
                                lowfreq_only=latent_calibrator_settings.map_lowfreq_only,
                                downsample_factor=latent_calibrator_settings.map_downsample_factor,
                            )
                        latent_calibrator_norm_loss_value = latent_calibrator_norm_loss(
                            delta=latent_calibrator_delta,
                            mask=aux_weight,
                            norm_cap=latent_calibrator_settings.residual_norm_cap,
                        )

                    assert frame_embeds.shape == pooled_prompt_embeds.shape
                    assert modified_pooled_prompt_embeds.shape == pooled_prompt_embeds.shape
                    frame_adapter_pooled_flat = None
                    frame_adapter_tokens = None
                    frame_attention_tokens = None
                    if frame_position_encoder is not None:
                        frame_adapter_pooled, frame_adapter_tokens = frame_position_encoder(frame_positions)
                        frame_adapter_pooled_flat = frame_adapter_pooled.reshape(frames_flat.shape[0], -1)
                        prompt_embeds, frame_attention_tokens = apply_frame_token_conditioning(
                            prompt_embeds=prompt_embeds,
                            frame_embeddings=frame_adapter_pooled,
                            frame_tokens=frame_adapter_tokens,
                            num_frames=num_frames,
                            mode=frame_encoder_config["token_embedding_mode"],
                            alpha=float(frame_encoder_config.get("alpha", 1.0)),
                        )

                    if not logged_shapes:
                        log_first_step_shapes(
                            accelerator,
                            {
                                "frames": tuple(frames.shape),
                                "frames_flat": tuple(frames_flat.shape),
                                "frame_positions": tuple(frame_positions.shape),
                                "frame_positions_flat": tuple(frame_positions_flat.shape),
                                "latents": tuple(latents.shape),
                                "noising_latents": tuple(noising_latents.shape),
                                "latent_init_mode": latent_init_mode,
                                "image_first_noise_mode": (
                                    image_first_noise_mode
                                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                                    else None
                                ),
                                "image_first_bridge_mode": (
                                    image_first_bridge_mode
                                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                                    else None
                                ),
                                "image_first_shared_noise_fraction": (
                                    _scalar(image_first_shared_noise_fraction)
                                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                                    else None
                                ),
                                "image_first_bridge_fraction": (
                                    _scalar(image_first_bridge_fraction)
                                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                                    else None
                                ),
                                "image_first_bridge_gate_mean": (
                                    float(image_first_bridge_gate_values.detach().float().mean().item())
                                    if image_first_bridge_gate_values is not None
                                    else None
                                ),
                                "image_first_boundary_loss": (
                                    float(image_first_boundary_loss_value.detach().float().item())
                                    if image_first_boundary_loss_weight > 0.0
                                    else None
                                ),
                                "image_first_rollout_steps": (
                                    image_first_rollout_steps
                                    if (
                                        latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                                        and image_first_uses_rollout_bridge
                                    )
                                    else None
                                ),
                                "latent_calibrator_enabled": latent_calibrator is not None,
                                "latent_calibrator_map_loss": (
                                    float(latent_calibrator_map_loss.detach().float().item())
                                    if latent_calibrator is not None
                                    else None
                                ),
                                "latent_calibrator_norm_loss": (
                                    float(latent_calibrator_norm_loss_value.detach().float().item())
                                    if latent_calibrator is not None
                                    else None
                                ),
                                "video_timesteps": tuple(timesteps.reshape(batch_size, num_frames)[:, 0].shape),
                                "timesteps": tuple(timesteps.shape),
                                "prompt_embeds": tuple(prompt_embeds.shape),
                                "pooled_prompt_embeds": tuple(pooled_prompt_embeds.shape),
                                "frame_embeds": tuple(frame_embeds.shape),
                                "frame_adapter_pooled": (
                                    tuple(frame_adapter_pooled.shape)
                                    if frame_position_encoder is not None
                                    else None
                                ),
                                "frame_adapter_tokens": (
                                    tuple(frame_adapter_tokens.shape)
                                    if frame_position_encoder is not None
                                    else None
                                ),
                                "frame_attention_tokens": (
                                    tuple(frame_attention_tokens.shape)
                                    if frame_attention_tokens is not None
                                    else None
                                ),
                                "modified_pooled_prompt_embeds": tuple(
                                    modified_pooled_prompt_embeds.shape
                                ),
                                "add_time_ids": tuple(add_time_ids.shape),
                            },
                            progress_bar=progress_bar,
                        )
                        logged_shapes = True

                    set_video_resnet_context(
                        unet,
                        num_frames=num_frames,
                        frame_positions=frame_positions_flat,
                        frame_embeddings=frame_adapter_pooled_flat,
                    )
                    image_first_anchor_latents = None
                    if (
                        image_first_anchor_enabled
                        and latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                    ):
                        image_first_anchor_latents = clean_latents.reshape(
                            batch_size,
                            num_frames,
                            *clean_latents.shape[1:],
                        )[:, 0].to(dtype=torch_dtype)
                    set_video_attention_context(
                        unet,
                        num_frames=num_frames,
                        frame_tokens=frame_attention_tokens,
                        anchor_latents=image_first_anchor_latents,
                    )
                    try:
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={
                                "text_embeds": modified_pooled_prompt_embeds,
                                "time_ids": add_time_ids,
                            },
                        ).sample
                    finally:
                        clear_video_resnet_context(unet)
                        clear_video_attention_context(unet)

                    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
                    if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT:
                        target = compute_clean_latent_epsilon_target(
                            noisy_latents=noisy_latents,
                            clean_latents=clean_latents.to(dtype=noisy_latents.dtype),
                            noise_scheduler=noise_scheduler,
                            timesteps=timesteps,
                        )
                        if (
                            image_first_uses_snr_gate
                            and bridge_mask is not None
                            and standard_noise is not None
                        ):
                            mask = _expand_latent_mask(bridge_mask, target)
                            target = torch.where(
                                mask,
                                target,
                                standard_noise.to(device=target.device, dtype=target.dtype),
                            )
                    elif prediction_type == "epsilon":
                        target = noise
                    elif prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(clean_latents, noise, timesteps)
                    elif prediction_type == "sample":
                        target = clean_latents
                        model_pred = model_pred - noise
                    else:
                        raise ValueError(f"Unsupported prediction_type={prediction_type!r}.")

                    if (
                        image_first_boundary_loss_weight > 0.0
                        and latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT
                        and image_first_boundary_timestep is not None
                        and image_first_bridge_gate_values is not None
                    ):
                        pred_clean_latents = predict_clean_latents_from_epsilon(
                            noisy_latents=noisy_latents,
                            epsilon=model_pred,
                            noise_scheduler=noise_scheduler,
                            timesteps=timesteps,
                        )
                        image_first_boundary_loss_value = compute_boundary_renoise_loss(
                            pred_clean_latents=pred_clean_latents,
                            clean_latents=clean_latents,
                            boundary_noise=noise.detach(),
                            noise_scheduler=noise_scheduler,
                            boundary_timestep=image_first_boundary_timestep,
                            sample_weights=image_first_bridge_gate_values.detach().float(),
                            lowfreq_only=image_first_boundary_lowfreq_only,
                            downsample_factor=image_first_boundary_downsample_factor,
                        )

                    loss = compute_diffusion_loss(
                        model_pred=model_pred,
                        target=target,
                        noise_scheduler=noise_scheduler,
                        timesteps=timesteps,
                        snr_gamma=snr_gamma,
                    )
                    if image_first_boundary_loss_weight > 0.0:
                        loss = loss + image_first_boundary_loss_weight * image_first_boundary_loss_value
                    if latent_calibrator is not None:
                        loss = loss + float(latent_calibrator_settings.map_weight) * latent_calibrator_map_loss
                        loss = loss + (
                            float(latent_calibrator_settings.norm_weight)
                            * latent_calibrator_norm_loss_value
                        )
                    accumulated_loss = accumulated_loss + loss.detach().float()
                    accumulated_loss_count += 1
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
                        if max_grad_norm > 0:
                            accelerator.clip_grad_norm_(parameters, max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    local_loss = accumulated_loss / max(accumulated_loss_count, 1)
                    avg_loss = accelerator.gather(local_loss.reshape(1)).mean().item()
                    accumulated_loss = torch.zeros(
                        (),
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    accumulated_loss_count = 0
                    gpu_memory_mb = get_gpu_memory_mb()
                    frames_seen = global_step * effective_batch_frames
                    epoch_progress = global_step / num_update_steps_per_epoch
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                                "epoch": f"{epoch_progress:.2f}",
                                "frames": frames_seen,
                                "gpu": (
                                    f"{gpu_memory_mb / 1024.0:.1f}G"
                                    if gpu_memory_mb is not None
                                    else "n/a"
                                ),
                            }
                        )

                    if global_step % logging_steps == 0:
                        metrics = {
                            "train/loss": avg_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch_progress,
                            "train/global_step": global_step,
                            "train/samples_seen_frames": frames_seen,
                            "train/effective_batch_videos": effective_batch_videos,
                            "train/effective_batch_frames": effective_batch_frames,
                        }
                        if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT:
                            metrics["train/image_first_shared_noise_fraction"] = (
                                _scalar(image_first_shared_noise_fraction)
                            )
                            metrics["train/image_first_shared_noise_prob"] = image_first_shared_noise_prob
                            metrics["train/image_first_bridge_fraction"] = _scalar(image_first_bridge_fraction)
                            if image_first_bridge_gate_values is not None:
                                metrics["train/image_first_bridge_gate_mean"] = (
                                    image_first_bridge_gate_values.detach().float().mean().item()
                                )
                            if image_first_boundary_loss_weight > 0.0:
                                metrics["train/image_first_boundary_loss"] = (
                                    image_first_boundary_loss_value.detach().float().item()
                                )
                                metrics["train/image_first_boundary_loss_weight"] = (
                                    image_first_boundary_loss_weight
                                )
                            if image_first_uses_rollout_bridge:
                                metrics["train/image_first_rollout_steps"] = image_first_rollout_steps
                        if latent_calibrator is not None:
                            metrics["train/latent_calibrator_map_loss"] = (
                                latent_calibrator_map_loss.detach().float().item()
                            )
                            metrics["train/latent_calibrator_norm_loss"] = (
                                latent_calibrator_norm_loss_value.detach().float().item()
                            )
                            metrics["train/latent_calibrator_map_weight"] = (
                                float(latent_calibrator_settings.map_weight)
                            )
                        if gpu_memory_mb is not None:
                            metrics["system/gpu_max_memory_allocated_mb"] = gpu_memory_mb
                        if tracker_is_enabled(config):
                            accelerator.log(metrics, step=global_step)
                        if progress_bar is None:
                            accelerator.print(f"step {global_step}: loss={avg_loss:.6f}")

                    if should_log_training_caption_inference(config, global_step):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            training_prompt = batch["captions"][0]
                            log_training_caption_inference(
                                accelerator=accelerator,
                                config=config,
                                pipe=pipe,
                                unet=unet,
                                vae=vae,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                temporal_mlp=temporal_mlp,
                                frame_position_encoder=frame_position_encoder,
                                prompt=training_prompt,
                                global_step=global_step,
                                output_dir=output_dir,
                                resolution=resolution,
                                temporal_alpha=temporal_alpha,
                                injection_mode=injection_mode,
                                frame_encoder_config=frame_encoder_config,
                            )
                            write_status(
                                accelerator,
                                "logged training-caption inference sample "
                                f"at step {global_step}: {training_prompt[:120]}",
                                progress_bar=progress_bar,
                            )
                        accelerator.wait_for_everyone()

                    if (
                        int(training_config.get("checkpointing_steps", 0)) > 0
                        and global_step % int(training_config["checkpointing_steps"]) == 0
                    ):
                        save_training_checkpoint(
                            accelerator=accelerator,
                            output_dir=output_dir,
                            step=global_step,
                            unet=unet,
                            vae=vae,
                            temporal_mlp=temporal_mlp,
                            frame_position_encoder=frame_position_encoder,
                            latent_calibrator=latent_calibrator,
                            config=config,
                            temporal_config=temporal_config,
                            progress_bar=progress_bar,
                        )
                        if tracker_is_enabled(config):
                            accelerator.log({"checkpoint/saved_step": global_step}, step=global_step)

                    validation_config = config.get("validation", {})
                    if (
                        validation_config.get("enabled", False)
                        and int(training_config.get("validation_steps", 0)) > 0
                        and global_step % int(training_config["validation_steps"]) == 0
                    ):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            pipe.unet = accelerator.unwrap_model(unet)
                            pipe.to(accelerator.device)
                            validation_dir = (
                                output_dir
                                / validation_config.get("output_dir", "validation")
                                / f"step_{global_step:04d}"
                            )
                            validation_temporal_mlp = accelerator.unwrap_model(temporal_mlp)
                            validation_frame_position_encoder = (
                                accelerator.unwrap_model(frame_position_encoder)
                                if frame_position_encoder is not None
                                else None
                            )
                            validation_latent_calibrator = (
                                accelerator.unwrap_model(latent_calibrator)
                                if latent_calibrator is not None
                                else None
                            )
                            if latent_init_mode == LATENT_INIT_FIRST_FRAME_REPEAT:
                                guidance_scale = float(validation_config.get("guidance_scale", 8.0))
                                switch_noise_scale = float(validation_config.get("switch_noise_scale", 0.1))
                                switch_mode = str(
                                    validation_config.get("image_first_switch_mode", "repeat_add_noise")
                                )
                                renoise_noise_mode = str(
                                    validation_config.get("image_first_renoise_noise_mode", "independent")
                                )
                                renoise_noise_scale = float(
                                    validation_config.get("image_first_renoise_noise_scale", 1.0)
                                )
                                guidance_label = guidance_scale_label(guidance_scale)
                                t1_ratios = config_float_list(
                                    validation_config.get("t1_ratios", [0.0, 0.25, 0.5, 0.75])
                                )
                                for t1_ratio in t1_ratios:
                                    t1_label = t1_ratio_label(t1_ratio)
                                    guidance_validation_dir = image_first_output_dir(
                                        validation_dir,
                                        t1_ratio=t1_ratio,
                                        guidance_scale=guidance_scale,
                                    )
                                    generate_image_first_video_frames(
                                        pipe=pipe,
                                        temporal_mlp=validation_temporal_mlp,
                                        frame_position_encoder=validation_frame_position_encoder,
                                        latent_calibrator=validation_latent_calibrator,
                                        latent_calibrator_config=latent_calibrator_config,
                                        prompt=validation_config["prompt"],
                                        num_frames=int(validation_config["num_frames"]),
                                        output_dir=guidance_validation_dir,
                                        resolution=resolution,
                                        temporal_alpha=temporal_alpha,
                                        t1_ratio=float(t1_ratio),
                                        guidance_scale=guidance_scale,
                                        injection_mode=injection_mode,
                                        frame_token_embedding_mode=frame_encoder_config["token_embedding_mode"],
                                        frame_token_alpha=float(frame_encoder_config.get("alpha", 1.0)),
                                        num_inference_steps=int(validation_config.get("num_inference_steps", 30)),
                                        seed=training_config.get("seed"),
                                        save_grid=bool(validation_config.get("save_grid", True)),
                                        save_video=bool(validation_config.get("save_mp4", False)),
                                        fps=int(validation_config.get("fps", 8)),
                                        switch_noise_scale=switch_noise_scale,
                                        switch_mode=switch_mode,
                                        renoise_noise_mode=renoise_noise_mode,
                                        renoise_noise_scale=renoise_noise_scale,
                                    )
                                    write_status(
                                        accelerator,
                                        "saved image-first validation "
                                        f"{t1_label}/{guidance_label} frames to {guidance_validation_dir}",
                                        progress_bar=progress_bar,
                                    )
                                    if tracker_is_enabled(config):
                                        accelerator.log(
                                            {
                                                f"validation/{t1_label}/{guidance_label}/generated_frames": int(
                                                    validation_config["num_frames"]
                                                ),
                                                f"validation/{t1_label}/{guidance_label}/guidance_scale": guidance_scale,
                                                f"validation/{t1_label}/{guidance_label}/t1_ratio": float(t1_ratio),
                                                f"validation/{t1_label}/{guidance_label}/switch_noise_scale": (
                                                    switch_noise_scale
                                                ),
                                                f"validation/{t1_label}/{guidance_label}/renoise_noise_scale": (
                                                    renoise_noise_scale
                                                ),
                                            },
                                            step=global_step,
                                        )
                                    log_media_directory(
                                        accelerator=accelerator,
                                        config=config,
                                        media_dir=guidance_validation_dir,
                                        global_step=global_step,
                                        prefix=f"validation/{t1_label}/{guidance_label}",
                                        caption=validation_config["prompt"],
                                        fps=int(validation_config.get("fps", 8)),
                                    )
                            else:
                                guidance_scales = config_float_list(
                                    validation_config.get("guidance_scales", [1.0, 8.0]),
                                )
                                for guidance_scale in guidance_scales:
                                    guidance_label = guidance_scale_label(guidance_scale)
                                    guidance_validation_dir = validation_dir / guidance_label
                                    generate_video_frames(
                                        pipe=pipe,
                                        temporal_mlp=validation_temporal_mlp,
                                        frame_position_encoder=validation_frame_position_encoder,
                                        prompt=validation_config["prompt"],
                                        num_frames=int(validation_config["num_frames"]),
                                        output_dir=guidance_validation_dir,
                                        resolution=resolution,
                                        temporal_alpha=temporal_alpha,
                                        injection_mode=injection_mode,
                                        frame_token_embedding_mode=frame_encoder_config["token_embedding_mode"],
                                        frame_token_alpha=float(frame_encoder_config.get("alpha", 1.0)),
                                        num_inference_steps=int(validation_config.get("num_inference_steps", 30)),
                                        guidance_scale=float(guidance_scale),
                                        seed=training_config.get("seed"),
                                        batch_size=validation_config.get("batch_size"),
                                        save_grid=bool(validation_config.get("save_grid", True)),
                                        save_video=bool(validation_config.get("save_mp4", False)),
                                    )
                                    write_status(
                                        accelerator,
                                        f"saved validation {guidance_label} frames to {guidance_validation_dir}",
                                        progress_bar=progress_bar,
                                    )
                                    if tracker_is_enabled(config):
                                        accelerator.log(
                                            {
                                                f"validation/{guidance_label}/generated_frames": int(
                                                    validation_config["num_frames"]
                                                ),
                                                f"validation/{guidance_label}/guidance_scale": float(guidance_scale),
                                            },
                                            step=global_step,
                                        )
                                    log_media_directory(
                                        accelerator=accelerator,
                                        config=config,
                                        media_dir=guidance_validation_dir,
                                        global_step=global_step,
                                        prefix=f"validation/{guidance_label}",
                                        caption=validation_config["prompt"],
                                    )
                        accelerator.wait_for_everyone()

                    if global_step >= max_train_steps:
                        break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    accelerator.wait_for_everyone()
    save_training_checkpoint(
        accelerator=accelerator,
        output_dir=output_dir,
        step=global_step,
        unet=unet,
        vae=vae,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        latent_calibrator=latent_calibrator,
        config=config,
        temporal_config=temporal_config,
    )
    accelerator.print(f"training complete at step {global_step}; saved checkpoint-last")
    accelerator.end_training()


if __name__ == "__main__":
    main()
