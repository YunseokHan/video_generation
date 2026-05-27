#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import AutoencoderKL, DDPMScheduler

from framegen.config import as_project_path, get_torch_dtype, load_config
from framegen.data import build_dataset
from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DDPM forward residuals between an 8-frame video latent sequence "
            "and a sequence made by repeating one randomly selected frame latent."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/train/image_first_sinusoidal.yaml",
        help="Training config used to load the dataset/model settings.",
    )
    parser.add_argument("--env_file", default=".env", help="Environment file with HF cache/token settings.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset index to analyze.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of uniformly sampled frames.")
    parser.add_argument("--num-timesteps", type=int, default=100, help="Number of DDPM timesteps to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for VAE sampling, anchor selection, and noise.")
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=None,
        help="Frame index to repeat. Defaults to a random index in [0, num_frames).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for VAE encoding and DDPM forward process.",
    )
    parser.add_argument(
        "--output",
        default="outputs/residual_analysis.png",
        help="Path where the residual plot will be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSON path for the numeric metrics used in the plot.",
    )
    return parser.parse_args()


def load_sdxl_vae(model_config: dict[str, Any], vae_dtype: torch.dtype) -> AutoencoderKL:
    vae_path = model_config.get("pretrained_vae_model_name_or_path")
    subfolder = None
    if vae_path is None:
        vae_path = model_config["pretrained_model_name_or_path"]
        subfolder = "vae"

    kwargs: dict[str, Any] = {"torch_dtype": vae_dtype}
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


def load_noise_scheduler(model_config: dict[str, Any]) -> DDPMScheduler:
    kwargs: dict[str, Any] = {}
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


def clone_config_for_analysis(config: dict[str, Any], num_frames: int) -> dict[str, Any]:
    analysis_config = dict(config)
    analysis_config["model"] = dict(config["model"])
    analysis_config["data"] = dict(config["data"])
    analysis_config["temporal_conditioning"] = dict(config.get("temporal_conditioning", {}))
    analysis_config["data"]["num_frames_per_video"] = int(num_frames)
    analysis_config["data"]["frame_sampling"] = "uniform"
    return analysis_config


def encode_video_latents(
    vae: AutoencoderKL,
    frames: torch.Tensor,
    device: torch.device,
    vae_dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    frames = frames.to(device=device, dtype=vae_dtype)
    latent_dist = vae.encode(frames).latent_dist
    latents = latent_dist.sample(generator=generator)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return latents * scaling_factor


def rms_per_frame(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean(dim=(1, 2, 3)).sqrt()


def compute_forward_residuals(
    scheduler: DDPMScheduler,
    latents: torch.Tensor,
    repeated_latents: torch.Tensor,
    num_timestep_points: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if latents.shape != repeated_latents.shape:
        raise ValueError(f"Latent shapes must match: {latents.shape} vs {repeated_latents.shape}")

    total_timesteps = int(scheduler.config.num_train_timesteps)
    timestep_values = torch.linspace(
        0,
        total_timesteps - 1,
        steps=int(num_timestep_points),
        device=latents.device,
    ).round().long()
    timestep_values = torch.unique_consecutive(timestep_values)

    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    per_frame = []
    mean_rms = []
    max_rms = []
    cosine = []

    flat_latents = latents.flatten()
    flat_repeated = repeated_latents.flatten()
    clean_cosine = torch.nn.functional.cosine_similarity(
        flat_latents.float(),
        flat_repeated.float(),
        dim=0,
    )

    for timestep in timestep_values:
        timesteps = torch.full(
            (latents.shape[0],),
            int(timestep.item()),
            device=latents.device,
            dtype=torch.long,
        )
        noised_latents = scheduler.add_noise(latents, noise, timesteps)
        noised_repeated = scheduler.add_noise(repeated_latents, noise, timesteps)
        residual = noised_latents - noised_repeated
        frame_rms = rms_per_frame(residual)
        per_frame.append(frame_rms.detach().cpu())
        mean_rms.append(frame_rms.mean().detach().cpu())
        max_rms.append(frame_rms.max().detach().cpu())
        cosine.append(
            torch.nn.functional.cosine_similarity(
                noised_latents.flatten().float(),
                noised_repeated.flatten().float(),
                dim=0,
            )
            .detach()
            .cpu()
        )

    alphas = scheduler.alphas_cumprod.to(device=latents.device)[timestep_values].float().sqrt().cpu()
    return {
        "timesteps": timestep_values.detach().cpu(),
        "per_frame_rms": torch.stack(per_frame, dim=0),
        "mean_rms": torch.stack(mean_rms),
        "max_rms": torch.stack(max_rms),
        "cosine": torch.stack(cosine),
        "sqrt_alpha_cumprod": alphas,
        "clean_per_frame_rms": rms_per_frame(latents - repeated_latents).detach().cpu(),
        "clean_cosine": clean_cosine.detach().cpu(),
    }


def plot_metrics(
    metrics: dict[str, torch.Tensor],
    output_path: Path,
    anchor_index: int,
    sample_label: str,
) -> None:
    timesteps = metrics["timesteps"].numpy()
    per_frame = metrics["per_frame_rms"].numpy()
    mean_rms = metrics["mean_rms"].numpy()
    max_rms = metrics["max_rms"].numpy()
    sqrt_alpha = metrics["sqrt_alpha_cumprod"].numpy()
    cosine = metrics["cosine"].numpy()

    baseline = float(mean_rms[0]) if float(mean_rms[0]) > 0.0 else 1.0
    normalized_mean = mean_rms / baseline

    caption = textwrap.shorten(sample_label.replace("\n", " "), width=95, placeholder="...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "DDPM forward residual: video latent sequence vs repeated-frame latent sequence\n"
        f"sample: {caption}\nanchor frame: {anchor_index}; shared forward noise",
        fontsize=11,
    )

    ax = axes[0, 0]
    for frame_index in range(per_frame.shape[1]):
        linewidth = 2.2 if frame_index == anchor_index else 1.2
        alpha = 1.0 if frame_index == anchor_index else 0.75
        label = f"frame {frame_index}"
        if frame_index == anchor_index:
            label += " (repeated anchor)"
        ax.plot(timesteps, per_frame[:, frame_index], label=label, linewidth=linewidth, alpha=alpha)
    ax.set_title("Per-frame latent residual RMS")
    ax.set_xlabel("DDPM timestep")
    ax.set_ylabel("RMS(noised_video - noised_repeat)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    ax.plot(timesteps, mean_rms, label="mean frame RMS", linewidth=2.0)
    ax.plot(timesteps, max_rms, label="max frame RMS", linewidth=1.5, linestyle="--")
    ax.set_title("Sequence residual magnitude")
    ax.set_xlabel("DDPM timestep")
    ax.set_ylabel("RMS residual")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(timesteps, normalized_mean, label="normalized mean residual", linewidth=2.0)
    ax.plot(timesteps, sqrt_alpha, label="sqrt(alpha_cumprod)", linewidth=1.5, linestyle="--")
    ax.set_title("Residual decay under shared forward noise")
    ax.set_xlabel("DDPM timestep")
    ax.set_ylabel("relative scale")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(timesteps, cosine, color="tab:green", linewidth=2.0)
    ax.set_title("Noised sequence cosine similarity")
    ax.set_xlabel("DDPM timestep")
    ax.set_ylabel("cosine(noised_video, noised_repeat)")
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def tensor_to_jsonable(value: torch.Tensor) -> Any:
    if value.ndim == 0:
        return float(value.item())
    return value.tolist()


def main() -> None:
    args = parse_args()
    load_env_file(PROJECT_ROOT / ".env", override=False)
    load_env_file(args.env_file, override=True)

    config_path = as_project_path(args.config, PROJECT_ROOT)
    config = load_config(config_path)
    config = clone_config_for_analysis(config, args.num_frames)
    model_config = config["model"]

    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    vae_dtype = get_torch_dtype(model_config.get("vae_dtype", model_config.get("dtype", "fp32")))

    dataset = build_dataset(config)
    sample_index = int(args.sample_index) % len(dataset)
    sample = dataset[sample_index]
    frames = sample["frames"]
    if frames.ndim != 4:
        raise ValueError(f"Expected one video tensor [F, 3, H, W], got {tuple(frames.shape)}")
    if frames.shape[0] != int(args.num_frames):
        raise ValueError(f"Expected {args.num_frames} frames, got {frames.shape[0]}")

    anchor_index = args.anchor_index
    if anchor_index is None:
        anchor_index = random.randrange(int(args.num_frames))
    if not 0 <= int(anchor_index) < int(args.num_frames):
        raise ValueError(f"--anchor-index must be in [0, {args.num_frames}), got {anchor_index}")
    anchor_index = int(anchor_index)

    vae = load_sdxl_vae(model_config, vae_dtype).to(device)
    vae.eval()
    scheduler = load_noise_scheduler(model_config)

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latents = encode_video_latents(vae, frames, device, vae_dtype, generator)
        repeated_latents = latents[anchor_index : anchor_index + 1].expand_as(latents).clone()
        metrics = compute_forward_residuals(
            scheduler=scheduler,
            latents=latents,
            repeated_latents=repeated_latents,
            num_timestep_points=int(args.num_timesteps),
            generator=generator,
        )

    sample_label = str(sample.get("caption", ""))
    if hasattr(dataset, "samples"):
        try:
            sample_info = dataset.samples[sample_index]
            if sample_info.get("path") is not None:
                sample_label = f"{sample_info['path']} | {sample_label}"
        except Exception:
            pass

    output_path = as_project_path(args.output, PROJECT_ROOT)
    plot_metrics(metrics, output_path, anchor_index=anchor_index, sample_label=sample_label)

    if args.metrics_output is not None:
        metrics_path = as_project_path(args.metrics_output, PROJECT_ROOT)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "sample_index": sample_index,
            "sample_label": sample_label,
            "anchor_index": anchor_index,
            "num_frames": int(args.num_frames),
            "metrics": {key: tensor_to_jsonable(value) for key, value in metrics.items()},
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved residual plot to {output_path}")
    print(f"anchor frame index: {anchor_index}")
    print(f"clean mean RMS residual: {float(metrics['clean_per_frame_rms'].mean()):.6f}")
    print(f"clean sequence cosine: {float(metrics['clean_cosine']):.6f}")


if __name__ == "__main__":
    main()
