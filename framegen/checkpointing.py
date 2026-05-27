from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .config import save_config
from .latent_calibrator import build_latent_calibrator
from .temporal import FramePositionMLP, build_frame_position_encoder
from .video_attention import (
    load_video_attention_adapter_state_dict,
    video_attention_adapter_state_dict,
)
from .video_resnet import (
    load_video_resnet_adapter_state_dict,
    video_resnet_adapter_state_dict,
)


VIDEO_RESNET_ADAPTER_FILENAME = "resnet_video_adapter.pt"
VAE_DECODER_RESNET_ADAPTER_FILENAME = "vae_decoder_video_adapter.pt"
VIDEO_ATTENTION_ADAPTER_FILENAME = "attention_video_adapter.pt"
FRAME_POSITION_ENCODER_FILENAME = "frame_position_encoder.pt"
LATENT_CALIBRATOR_FILENAME = "latent_calibrator.pt"


def _is_main_process(accelerator) -> bool:
    return accelerator is None or accelerator.is_main_process


def _unwrap(model, accelerator):
    return accelerator.unwrap_model(model) if accelerator is not None else model


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    unet,
    vae,
    temporal_mlp: FramePositionMLP,
    frame_position_encoder: torch.nn.Module | None,
    latent_calibrator: torch.nn.Module | None,
    config: dict[str, Any],
    temporal_config: dict[str, Any],
    accelerator=None,
) -> None:
    if not _is_main_process(accelerator):
        return

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    last_dir = output_dir / "checkpoint-last"

    _write_checkpoint(
        checkpoint_dir,
        step,
        unet,
        vae,
        temporal_mlp,
        frame_position_encoder,
        latent_calibrator,
        config,
        temporal_config,
        accelerator,
    )
    if last_dir.exists():
        shutil.rmtree(last_dir)
    shutil.copytree(checkpoint_dir, last_dir)


def _write_checkpoint(
    checkpoint_dir: Path,
    step: int,
    unet,
    vae,
    temporal_mlp: FramePositionMLP,
    frame_position_encoder: torch.nn.Module | None,
    latent_calibrator: torch.nn.Module | None,
    config: dict[str, Any],
    temporal_config: dict[str, Any],
    accelerator=None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _unwrap(unet, accelerator).save_pretrained(checkpoint_dir / "unet")
    unwrapped_unet = _unwrap(unet, accelerator)
    video_resnet_state = video_resnet_adapter_state_dict(unwrapped_unet)
    if video_resnet_state:
        torch.save(
            {
                "global_step": step,
                "state_dict": video_resnet_state,
                "config": config.get("video_adapters", {}).get("resnet", {}),
            },
            checkpoint_dir / VIDEO_RESNET_ADAPTER_FILENAME,
        )
    if vae is not None:
        vae_decoder = getattr(_unwrap(vae, accelerator), "decoder", _unwrap(vae, accelerator))
        vae_decoder_resnet_state = video_resnet_adapter_state_dict(vae_decoder)
        if vae_decoder_resnet_state:
            torch.save(
                {
                    "global_step": step,
                    "state_dict": vae_decoder_resnet_state,
                    "config": config.get("video_adapters", {}).get("vae_decoder_resnet", {}),
                },
                checkpoint_dir / VAE_DECODER_RESNET_ADAPTER_FILENAME,
            )
    video_attention_state = video_attention_adapter_state_dict(unwrapped_unet)
    if video_attention_state:
        torch.save(
            {
                "global_step": step,
                "state_dict": video_attention_state,
                "config": config.get("video_adapters", {}).get("attention", {}),
            },
            checkpoint_dir / VIDEO_ATTENTION_ADAPTER_FILENAME,
        )
    temporal_model = _unwrap(temporal_mlp, accelerator)
    torch.save(
        {
            "global_step": step,
            "state_dict": temporal_model.state_dict(),
            "config": temporal_config,
        },
        checkpoint_dir / "temporal_mlp.pt",
    )
    if frame_position_encoder is not None:
        frame_encoder_model = _unwrap(frame_position_encoder, accelerator)
        torch.save(
            {
                "global_step": step,
                "state_dict": frame_encoder_model.state_dict(),
                "config": config.get("video_adapters", {}).get("frame_position_encoder", {}),
            },
            checkpoint_dir / FRAME_POSITION_ENCODER_FILENAME,
        )
    if latent_calibrator is not None:
        calibrator_model = _unwrap(latent_calibrator, accelerator)
        torch.save(
            {
                "global_step": step,
                "state_dict": calibrator_model.state_dict(),
                "config": config.get("latent_calibrator", {}),
            },
            checkpoint_dir / LATENT_CALIBRATOR_FILENAME,
        )
    save_config(config, checkpoint_dir / "config.yaml")


def load_temporal_mlp_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[FramePositionMLP, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir) / "temporal_mlp.pt"
    payload = torch.load(checkpoint_path, map_location=map_location)
    temporal_config = dict(payload["config"])
    mlp = FramePositionMLP(
        output_dim=int(temporal_config["output_dim"]),
        hidden_dim=int(temporal_config["hidden_dim"]),
        num_layers=int(temporal_config["num_layers"]),
    )
    mlp.load_state_dict(payload["state_dict"])
    return mlp, temporal_config


def load_frame_position_encoder_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]] | tuple[None, None]:
    checkpoint_path = Path(checkpoint_dir) / FRAME_POSITION_ENCODER_FILENAME
    if not checkpoint_path.exists():
        return None, None
    payload = torch.load(checkpoint_path, map_location=map_location)
    encoder_config = dict(payload["config"])
    encoder_config.setdefault("type", "learned_tokens")
    encoder = build_frame_position_encoder(encoder_config)
    encoder.load_state_dict(payload["state_dict"])
    return encoder, encoder_config


def load_latent_calibrator_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]] | tuple[None, None]:
    checkpoint_path = Path(checkpoint_dir) / LATENT_CALIBRATOR_FILENAME
    if not checkpoint_path.exists():
        return None, None
    payload = torch.load(checkpoint_path, map_location=map_location)
    calibrator_config = dict(payload["config"])
    calibrator_config["enabled"] = True
    calibrator = build_latent_calibrator(calibrator_config)
    if calibrator is None:
        raise RuntimeError("Latent calibrator checkpoint exists but config did not build a module.")
    calibrator.load_state_dict(payload["state_dict"])
    return calibrator, calibrator_config


def load_video_resnet_adapter_checkpoint(
    checkpoint_dir: str | Path,
    unet,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_dir) / VIDEO_RESNET_ADAPTER_FILENAME
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=map_location)
    load_video_resnet_adapter_state_dict(unet, payload["state_dict"], strict=strict)
    return dict(payload.get("config", {}))


def load_vae_decoder_resnet_adapter_checkpoint(
    checkpoint_dir: str | Path,
    vae_decoder,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_dir) / VAE_DECODER_RESNET_ADAPTER_FILENAME
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=map_location)
    load_video_resnet_adapter_state_dict(vae_decoder, payload["state_dict"], strict=strict)
    return dict(payload.get("config", {}))


def load_video_attention_adapter_checkpoint(
    checkpoint_dir: str | Path,
    unet,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_dir) / VIDEO_ATTENTION_ADAPTER_FILENAME
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=map_location)
    load_video_attention_adapter_state_dict(unet, payload["state_dict"], strict=strict)
    return dict(payload.get("config", {}))
