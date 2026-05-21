from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file

load_env_file(PROJECT_ROOT / ".env", override=False)

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

from framegen.checkpointing import (
    VAE_DECODER_RESNET_ADAPTER_FILENAME,
    VIDEO_ATTENTION_ADAPTER_FILENAME,
    VIDEO_RESNET_ADAPTER_FILENAME,
    load_frame_position_encoder_checkpoint,
    load_temporal_mlp_checkpoint,
    load_vae_decoder_resnet_adapter_checkpoint,
    load_video_attention_adapter_checkpoint,
    load_video_resnet_adapter_checkpoint,
)
from framegen.config import as_project_path, get_torch_dtype, load_config
from framegen.image_first_generation import (
    generate_image_first_video_frames,
    image_first_output_dir,
)
from framegen.temporal import normalize_frame_token_mode
from framegen.video_attention import (
    VideoAttentionAdapterConfig,
    inject_video_attention_adapters,
    set_video_attention_adapters_active,
    sync_video_attention_adapter_device_dtype,
)
from framegen.video_resnet import (
    VideoResnetAdapterConfig,
    inject_video_resnet_adapters,
    set_video_resnet_adapters_active,
    sync_video_resnet_adapter_device_dtype,
)
from infer import load_sdxl_vae, resolve_checkpoint_dir, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate video with image-first SDXL denoising and video adapters after t1."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML. Defaults to checkpoint/config.yaml when available.",
    )
    parser.add_argument("--env_file", default=".env", help="Path to an environment-variable file.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory containing unet/ and adapter checkpoints.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name under outputs/{name}; used with --step to locate checkpoint-{step}.",
    )
    parser.add_argument(
        "--step",
        default=None,
        help="Checkpoint step used with --name. Use an integer step or 'last'.",
    )
    parser.add_argument("--prompt", default=None, help="Video prompt. Defaults to validation.prompt.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate.")
    parser.add_argument(
        "--t1",
        "--t1_ratio",
        dest="t1_ratio",
        type=float,
        required=True,
        help="Denoising ratio spent with adapters off before repeating the image latent.",
    )
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root. Defaults to outputs/infer_image_first/{name}-{step}.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--switch_noise_scale",
        type=float,
        default=None,
        help="Frame-wise perturbation scale applied after duplicating the image latent.",
    )
    parser.add_argument(
        "--save_mp4",
        action="store_true",
        help="Also write video.mp4 when MP4 export support is available.",
    )
    parser.add_argument(
        "--no_mp4",
        action="store_true",
        help="Skip video.mp4 export, including in --name/--step mode.",
    )
    parser.add_argument("--no_grid", action="store_true", help="Skip grid.png export.")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--disable_video_resnet_adapters",
        action="store_true",
        help="Load adapter weights but keep UNet Resnet video adapters inactive.",
    )
    parser.add_argument(
        "--disable_video_attention_adapters",
        action="store_true",
        help="Load adapter weights but keep UNet attention video adapters inactive.",
    )
    parser.add_argument(
        "--disable_vae_decoder_adapters",
        action="store_true",
        help="Load adapter weights but keep VAE decoder video adapters inactive.",
    )
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace, project_root: Path) -> Path:
    if args.output_dir is not None:
        return as_project_path(args.output_dir, project_root)
    if args.name is not None and args.step is not None:
        return project_root / "outputs" / "infer_image_first" / f"{args.name}-{args.step}"
    raise ValueError("Provide --output_dir when --name/--step is not used.")


def load_image_first_pipeline(
    args: argparse.Namespace,
    config: dict,
    checkpoint_dir: Path,
    torch_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    device: torch.device,
):
    model_config = config["model"]
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

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        **kwargs,
    )
    pipe.vae = load_sdxl_vae(model_config, vae_dtype)

    unet_dir = checkpoint_dir / "unet"
    if not unet_dir.exists():
        raise FileNotFoundError(f"Missing trained UNet directory: {unet_dir}")
    pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch_dtype)

    video_resnet_checkpoint = checkpoint_dir / VIDEO_RESNET_ADAPTER_FILENAME
    if video_resnet_checkpoint.exists():
        payload = torch.load(video_resnet_checkpoint, map_location="cpu")
        video_resnet_config = VideoResnetAdapterConfig.from_config(payload.get("config", {}))
        video_resnet_config.enabled = True
    else:
        video_resnet_config = VideoResnetAdapterConfig.from_config(
            config.get("video_adapters", {}).get("resnet")
        )
    video_resnet_blocks = inject_video_resnet_adapters(pipe.unet, video_resnet_config)
    if video_resnet_blocks:
        loaded_config = load_video_resnet_adapter_checkpoint(
            checkpoint_dir,
            pipe.unet,
            map_location="cpu",
            strict=True,
        )
        if loaded_config is not None:
            video_resnet_config = VideoResnetAdapterConfig.from_config(loaded_config)
            video_resnet_config.enabled = True
        set_video_resnet_adapters_active(
            pipe.unet,
            active=video_resnet_config.active and not args.disable_video_resnet_adapters,
        )
        sync_video_resnet_adapter_device_dtype(pipe.unet)

    vae_decoder = getattr(pipe.vae, "decoder", pipe.vae)
    vae_decoder_checkpoint = checkpoint_dir / VAE_DECODER_RESNET_ADAPTER_FILENAME
    if vae_decoder_checkpoint.exists():
        payload = torch.load(vae_decoder_checkpoint, map_location="cpu")
        vae_decoder_resnet_config = VideoResnetAdapterConfig.from_config(payload.get("config", {}))
        vae_decoder_resnet_config.enabled = True
    else:
        vae_decoder_resnet_config = VideoResnetAdapterConfig.from_config(
            config.get("video_adapters", {}).get("vae_decoder_resnet")
        )
    vae_decoder_resnet_blocks = inject_video_resnet_adapters(
        vae_decoder,
        vae_decoder_resnet_config,
    )
    if vae_decoder_resnet_blocks:
        loaded_config = load_vae_decoder_resnet_adapter_checkpoint(
            checkpoint_dir,
            vae_decoder,
            map_location="cpu",
            strict=True,
        )
        if loaded_config is not None:
            vae_decoder_resnet_config = VideoResnetAdapterConfig.from_config(loaded_config)
            vae_decoder_resnet_config.enabled = True
        set_video_resnet_adapters_active(
            vae_decoder,
            active=vae_decoder_resnet_config.active and not args.disable_vae_decoder_adapters,
        )
        sync_video_resnet_adapter_device_dtype(vae_decoder)

    video_attention_checkpoint = checkpoint_dir / VIDEO_ATTENTION_ADAPTER_FILENAME
    if video_attention_checkpoint.exists():
        payload = torch.load(video_attention_checkpoint, map_location="cpu")
        video_attention_config = VideoAttentionAdapterConfig.from_config(payload.get("config", {}))
        video_attention_config.enabled = True
    else:
        video_attention_config = VideoAttentionAdapterConfig.from_config(
            config.get("video_adapters", {}).get("attention")
        )
    video_attention_blocks = inject_video_attention_adapters(pipe.unet, video_attention_config)
    if video_attention_blocks:
        loaded_config = load_video_attention_adapter_checkpoint(
            checkpoint_dir,
            pipe.unet,
            map_location="cpu",
            strict=True,
        )
        if loaded_config is not None:
            video_attention_config = VideoAttentionAdapterConfig.from_config(loaded_config)
            video_attention_config.enabled = True
        set_video_attention_adapters_active(
            pipe.unet,
            active=video_attention_config.active and not args.disable_video_attention_adapters,
        )
        sync_video_attention_adapter_device_dtype(pipe.unet)

    temporal_mlp, temporal_config = load_temporal_mlp_checkpoint(checkpoint_dir, map_location="cpu")
    temporal_mlp.to(device=device, dtype=torch_dtype)
    temporal_mlp.eval()
    frame_position_encoder, frame_encoder_config = load_frame_position_encoder_checkpoint(
        checkpoint_dir,
        map_location="cpu",
    )
    if frame_encoder_config is None:
        frame_encoder_config = dict(config.get("video_adapters", {}).get("frame_position_encoder", {}))
    frame_encoder_config.setdefault("token_embedding_mode", "temporal_cross_attention_only")
    frame_encoder_config.setdefault("alpha", 1.0)
    frame_encoder_config["token_embedding_mode"] = normalize_frame_token_mode(
        frame_encoder_config["token_embedding_mode"]
    )
    if frame_position_encoder is not None:
        if any(True for _ in frame_position_encoder.parameters()):
            frame_position_encoder.to(device=device, dtype=torch_dtype)
        else:
            frame_position_encoder.to(device=device)
        frame_position_encoder.eval()

    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe, temporal_mlp, temporal_config, frame_position_encoder, frame_encoder_config


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = project_root / env_file
    load_env_file(env_file, override=True)

    checkpoint_dir = resolve_checkpoint_dir(args, project_root)
    config_path = resolve_config_path(args, checkpoint_dir, project_root)
    config = load_config(config_path)
    model_config = config["model"]
    validation_config = config.get("validation", {})

    prompt = args.prompt or validation_config.get("prompt")
    if not prompt:
        raise ValueError("Provide --prompt or set validation.prompt in the config.")
    num_frames = args.num_frames
    if num_frames is None:
        num_frames = int(validation_config.get("num_frames", 0))
    if int(num_frames) <= 0:
        raise ValueError("Provide --num_frames or set validation.num_frames > 0 in the config.")
    if not 0.0 <= float(args.t1_ratio) <= 1.0:
        raise ValueError("--t1 must be in [0, 1].")

    torch_dtype = get_torch_dtype(model_config.get("dtype", "bf16"))
    vae_dtype = get_torch_dtype(model_config.get("vae_dtype", model_config.get("dtype", "bf16")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe, temporal_mlp, temporal_config, frame_position_encoder, frame_encoder_config = (
        load_image_first_pipeline(
            args=args,
            config=config,
            checkpoint_dir=checkpoint_dir,
            torch_dtype=torch_dtype,
            vae_dtype=vae_dtype,
            device=device,
        )
    )

    output_root = resolve_output_root(args, project_root)
    output_dir = image_first_output_dir(
        output_root,
        t1_ratio=float(args.t1_ratio),
        guidance_scale=float(args.guidance_scale),
    )
    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else int(validation_config.get("num_inference_steps", 30))
    )
    fps = args.fps if args.fps is not None else int(validation_config.get("fps", 8))
    switch_noise_scale = (
        args.switch_noise_scale
        if args.switch_noise_scale is not None
        else float(validation_config.get("switch_noise_scale", 0.1))
    )
    save_video = (args.save_mp4 or (args.name is not None and args.step is not None)) and not args.no_mp4

    frame_paths = generate_image_first_video_frames(
        pipe=pipe,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        prompt=str(prompt),
        num_frames=int(num_frames),
        output_dir=output_dir,
        resolution=int(model_config["resolution"]),
        temporal_alpha=float(temporal_config.get("alpha", config["temporal_conditioning"].get("alpha", 1.0))),
        t1_ratio=float(args.t1_ratio),
        guidance_scale=float(args.guidance_scale),
        injection_mode=temporal_config.get("injection_mode", "add_to_pooled_prompt_embeds"),
        frame_token_embedding_mode=frame_encoder_config["token_embedding_mode"],
        frame_token_alpha=float(frame_encoder_config.get("alpha", 1.0)),
        num_inference_steps=int(num_inference_steps),
        seed=args.seed,
        save_grid=not args.no_grid,
        save_video=save_video,
        fps=int(fps),
        switch_noise_scale=float(switch_noise_scale),
    )
    print(f"Loaded checkpoint from {checkpoint_dir}")
    print(f"Saved image-first outputs to {output_dir}")
    print(
        f"Saved {len(frame_paths)} frames with t1={float(args.t1_ratio):g}, "
        f"CFG={float(args.guidance_scale):g}, switch_noise_scale={float(switch_noise_scale):g}"
    )


if __name__ == "__main__":
    main()
