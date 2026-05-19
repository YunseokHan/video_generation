from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file

load_env_file(PROJECT_ROOT / ".env", override=False)

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

from framegen.checkpointing import (
    VIDEO_ATTENTION_ADAPTER_FILENAME,
    VIDEO_RESNET_ADAPTER_FILENAME,
    load_frame_position_encoder_checkpoint,
    load_temporal_mlp_checkpoint,
    load_video_attention_adapter_checkpoint,
    load_video_resnet_adapter_checkpoint,
)
from framegen.config import as_project_path, get_torch_dtype, load_config
from framegen.generation import generate_video_frames
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporally conditioned SDXL video frames.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--env_file", default=".env", help="Path to an environment-variable file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory containing unet/ and temporal_mlp.pt.")
    parser.add_argument("--prompt", required=True, help="Video-level prompt shared by all frames.")
    parser.add_argument("--num_frames", type=int, required=True, help="Number of frames to generate.")
    parser.add_argument("--output_dir", required=True, help="Directory for frame_000.png, frame_001.png, ...")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_mp4", action="store_true", help="Also write video.mp4 when imageio is installed.")
    parser.add_argument("--no_grid", action="store_true", help="Skip grid.png export.")
    parser.add_argument(
        "--disable_video_resnet_adapters",
        action="store_true",
        help="Load adapter weights but run the Resnet video adapters in off mode.",
    )
    parser.add_argument(
        "--disable_video_attention_adapters",
        action="store_true",
        help="Load adapter weights but run the attention video adapters in off mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent if config_path.parent.name == "configs" else Path.cwd()
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = project_root / env_file
    load_env_file(env_file, override=True)
    checkpoint_dir = as_project_path(args.checkpoint, project_root)
    output_dir = as_project_path(args.output_dir, project_root)

    model_config = config["model"]
    validation_config = config.get("validation", {})
    torch_dtype = get_torch_dtype(model_config.get("dtype", "bf16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {"torch_dtype": torch_dtype}
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
    unet_dir = checkpoint_dir / "unet"
    if unet_dir.exists():
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch_dtype)
    else:
        raise FileNotFoundError(f"Missing trained UNet directory: {unet_dir}")

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
    frame_position_encoder, _ = load_frame_position_encoder_checkpoint(
        checkpoint_dir,
        map_location="cpu",
    )
    if frame_position_encoder is not None:
        frame_position_encoder.to(device=device, dtype=torch_dtype)
        frame_position_encoder.eval()

    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else int(validation_config.get("num_inference_steps", 30))
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(validation_config.get("guidance_scale", 7.5))
    )

    frame_paths = generate_video_frames(
        pipe=pipe,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        prompt=args.prompt,
        num_frames=args.num_frames,
        output_dir=output_dir,
        resolution=int(model_config["resolution"]),
        temporal_alpha=float(temporal_config.get("alpha", config["temporal_conditioning"].get("alpha", 1.0))),
        injection_mode=temporal_config.get("injection_mode", "add_to_pooled_prompt_embeds"),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=args.seed,
        batch_size=args.batch_size,
        save_grid=not args.no_grid,
        save_video=args.save_mp4,
    )
    print(f"Saved {len(frame_paths)} frames to {output_dir}")


if __name__ == "__main__":
    main()
