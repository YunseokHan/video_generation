from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file

load_env_file(PROJECT_ROOT / ".env", override=False)

import torch
import torch.nn.functional as torch_f
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from torch.utils.data import DataLoader

from framegen.checkpointing import save_checkpoint
from framegen.config import as_project_path, get_torch_dtype, load_config, save_config
from framegen.data import build_dataset, flatten_video_batch, video_collate_fn
from framegen.generation import generate_video_frames
from framegen.sdxl import (
    add_temporal_embedding_to_pooled_prompt_embeds,
    compute_sdxl_time_ids,
    encode_prompts_with_sdxl_logic,
    validate_sdxl_added_conditioning_dimensions,
)
from framegen.temporal import FramePositionMLP, build_frame_position_encoder
from framegen.utils import count_trainable_and_frozen, format_param_count, set_requires_grad
from framegen.video_attention import (
    VideoAttentionAdapterConfig,
    clear_video_attention_context,
    inject_video_attention_adapters,
    set_video_attention_adapter_requires_grad,
    set_video_attention_adapters_active,
    set_video_attention_context,
    sync_video_attention_adapter_device_dtype,
)
from framegen.video_resnet import (
    VideoResnetAdapterConfig,
    clear_video_resnet_context,
    inject_video_resnet_adapters,
    set_video_resnet_adapter_requires_grad,
    set_video_resnet_adapters_active,
    set_video_resnet_context,
    sync_video_resnet_adapter_device_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SDXL frame-position-conditioned generator.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--env_file", default=".env", help="Path to an environment-variable file.")
    return parser.parse_args()


def maybe_load_pipeline(model_config: dict, torch_dtype: torch.dtype) -> StableDiffusionXLPipeline:
    kwargs = {"torch_dtype": torch_dtype}
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


def load_noise_scheduler(model_config: dict) -> DDPMScheduler:
    kwargs = {}
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
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Only AdamW is implemented, got optimizer.type={optimizer_type!r}.")
    return torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1.0e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 1.0e-2)),
    )


def prepare_trainable_models(accelerator, named_models, optimizer, dataloader):
    names = [name for name, model in named_models]
    models = [model for name, model in named_models]
    prepared = accelerator.prepare(*models, optimizer, dataloader)
    prepared_models = dict(zip(names, prepared[: len(models)]))
    return prepared_models, prepared[-2], prepared[-1]


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
        if value is not None:
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
    if not tracker_is_enabled(config):
        return
    if not config.get("logging", {}).get("log_validation_media", True):
        return
    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except ImportError:
        accelerator.print("wandb is not installed; skipped validation media logging.")
        return

    payload = {}
    grid_path = validation_dir / "grid.png"
    video_path = validation_dir / "video.mp4"
    if grid_path.exists():
        payload["validation/grid"] = wandb.Image(str(grid_path), caption=f"step {global_step}")
    if video_path.exists():
        payload["validation/video"] = wandb.Video(str(video_path), fps=8, format="mp4")
    if payload:
        accelerator.log(payload, step=global_step)


def get_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def log_first_step_shapes(accelerator: Accelerator, shapes: dict[str, object]) -> None:
    accelerator.print("First training-step tensor shapes")
    for key, value in shapes.items():
        accelerator.print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent if config_path.parent.name == "configs" else Path.cwd()
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = project_root / env_file
    load_env_file(env_file, override=True)

    training_config = config["training"]
    model_config = config["model"]
    temporal_config = dict(config["temporal_conditioning"])
    video_adapter_config = config.setdefault("video_adapters", {})
    frame_encoder_config = dict(video_adapter_config.get("frame_position_encoder", {}))
    frame_encoder_config.setdefault("enabled", False)
    frame_encoder_config.setdefault("type", "sinusoidal")
    frame_encoder_config.setdefault("train", True)
    frame_encoder_config.setdefault("embedding_dim", 2048)
    frame_encoder_config.setdefault("hidden_dim", 1024)
    frame_encoder_config.setdefault("num_layers", 2)
    frame_encoder_config.setdefault("num_tokens", 1)
    frame_encoder_config.setdefault("pooling", "mean")
    video_adapter_config["frame_position_encoder"] = frame_encoder_config
    video_resnet_config = VideoResnetAdapterConfig.from_config(
        video_adapter_config.get("resnet")
    )
    if frame_encoder_config["enabled"]:
        video_resnet_config.frame_embedding_dim = int(frame_encoder_config["embedding_dim"])
        video_adapter_config.setdefault("resnet", {})["frame_embedding_dim"] = int(
            frame_encoder_config["embedding_dim"]
        )
    video_attention_config = VideoAttentionAdapterConfig.from_config(
        video_adapter_config.get("attention")
    )
    torch_dtype = get_torch_dtype(model_config.get("dtype", "bf16"))
    output_dir = as_project_path(training_config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        mixed_precision=training_config.get("mixed_precision", "no"),
        log_with=get_report_to(config),
        project_dir=str(output_dir),
    )
    if training_config.get("seed") is not None:
        set_seed(int(training_config["seed"]))

    if accelerator.is_main_process:
        save_config(config, output_dir / "config.yaml")

    pipe = maybe_load_pipeline(model_config, torch_dtype)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet
    video_resnet_blocks = inject_video_resnet_adapters(unet, video_resnet_config)
    video_attention_blocks = inject_video_attention_adapters(unet, video_attention_config)
    sync_video_resnet_adapter_device_dtype(unet)
    sync_video_attention_adapter_device_dtype(unet)

    noise_scheduler = load_noise_scheduler(model_config)

    train_temporal_embedding = bool(training_config.get("train_temporal_embedding", True))
    train_unet = bool(training_config.get("train_unet", True))
    train_video_resnet_adapters = video_resnet_config.enabled and video_resnet_config.train
    train_video_attention_adapters = video_attention_config.enabled and video_attention_config.train
    train_frame_position_encoder = bool(
        frame_encoder_config["enabled"] and frame_encoder_config.get("train", True)
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

    device = accelerator.device
    if not train_unet:
        unet.to(device=device, dtype=torch_dtype)
    text_encoder.to(device=device, dtype=torch_dtype)
    text_encoder_2.to(device=device, dtype=torch_dtype)
    if not train_vae:
        vae.to(device=device, dtype=torch_dtype)

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
    set_requires_grad(temporal_mlp, train_temporal_embedding)
    set_requires_grad(frame_position_encoder, train_frame_position_encoder)
    temporal_mlp.train(train_temporal_embedding)
    if frame_position_encoder is not None:
        frame_position_encoder.train(train_frame_position_encoder)
    if not train_temporal_embedding:
        temporal_mlp.to(device=device, dtype=torch_dtype)
    if frame_position_encoder is not None and not train_frame_position_encoder:
        frame_position_encoder.to(device=device, dtype=torch_dtype)

    trainable_count, frozen_count = count_trainable_and_frozen(
        [unet, vae, text_encoder, text_encoder_2, temporal_mlp, frame_position_encoder]
    )
    log_startup(accelerator, config, torch_dtype, temporal_config, trainable_count, frozen_count)
    accelerator.print(
        "  video resnet adapters: "
        f"enabled={video_resnet_config.enabled}, active={video_resnet_config.active}, "
        f"train={train_video_resnet_adapters}, blocks={video_resnet_blocks}, "
        f"frame_embedding_dim={video_resnet_config.frame_embedding_dim}"
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
        f"dim={frame_encoder_config['embedding_dim']}"
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
    dataloader = DataLoader(
        dataset,
        batch_size=int(training_config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=video_collate_fn,
        drop_last=True,
    )

    parameters = [
        parameter
        for module in [unet, vae, text_encoder, text_encoder_2, temporal_mlp]
        + ([frame_position_encoder] if frame_position_encoder is not None else [])
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("No trainable parameters. Enable at least one training flag.")
    optimizer = build_optimizer(parameters, config)

    named_models = []
    if train_unet or train_video_resnet_adapters or train_video_attention_adapters:
        named_models.append(("unet", unet))
    if train_temporal_embedding:
        named_models.append(("temporal_mlp", temporal_mlp))
    if train_frame_position_encoder and frame_position_encoder is not None:
        named_models.append(("frame_position_encoder", frame_position_encoder))
    if train_text_encoder:
        named_models.extend([("text_encoder", text_encoder), ("text_encoder_2", text_encoder_2)])
    if train_vae:
        named_models.append(("vae", vae))

    prepared_models, optimizer, dataloader = prepare_trainable_models(
        accelerator,
        named_models,
        optimizer,
        dataloader,
    )
    unet = prepared_models.get("unet", unet)
    temporal_mlp = prepared_models.get("temporal_mlp", temporal_mlp)
    frame_position_encoder = prepared_models.get("frame_position_encoder", frame_position_encoder)
    text_encoder = prepared_models.get("text_encoder", text_encoder)
    text_encoder_2 = prepared_models.get("text_encoder_2", text_encoder_2)
    vae = prepared_models.get("vae", vae)
    accumulation_models = [model for _, model in named_models]
    maybe_watch_with_wandb(accelerator, unet, temporal_mlp, config)

    global_step = 0
    logged_shapes = False
    max_train_steps = int(training_config["max_train_steps"])
    resolution = int(model_config["resolution"])
    temporal_alpha = float(temporal_config["alpha"])
    injection_mode = temporal_config["injection_mode"]
    logging_steps = int(training_config.get("logging_steps", 10))
    gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    num_update_steps_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    effective_batch_frames = (
        int(training_config["train_batch_size"])
        * int(config["data"]["num_frames_per_video"])
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(*accumulation_models):
                frames = batch["frames"].to(device=accelerator.device, dtype=torch_dtype)
                frame_positions = batch["frame_positions"].to(device=accelerator.device)
                frames_flat, captions_flat, frame_positions_flat = flatten_video_batch(
                    {
                        "frames": frames,
                        "captions": batch["captions"],
                        "frame_positions": frame_positions,
                    }
                )
                frame_positions_flat = frame_positions_flat.to(device=accelerator.device)

                assert len(captions_flat) == frames_flat.shape[0]
                assert frame_positions_flat.shape[0] == frames_flat.shape[0]

                vae_context = contextlib.nullcontext() if train_vae else torch.no_grad()
                with vae_context:
                    latents = vae.encode(frames_flat).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_context = contextlib.nullcontext() if train_text_encoder else torch.no_grad()
                with text_context:
                    prompt_embeds, pooled_prompt_embeds = encode_prompts_with_sdxl_logic(
                        captions_flat,
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
                add_time_ids = compute_sdxl_time_ids(
                    original_size=(resolution, resolution),
                    crop_coords=(0, 0),
                    target_size=(resolution, resolution),
                    batch_size=frames_flat.shape[0],
                    device=accelerator.device,
                    dtype=pooled_prompt_embeds.dtype,
                )

                assert frame_embeds.shape == pooled_prompt_embeds.shape
                assert modified_pooled_prompt_embeds.shape == pooled_prompt_embeds.shape
                frame_adapter_pooled_flat = None
                frame_adapter_tokens = None
                if frame_position_encoder is not None:
                    frame_adapter_pooled, frame_adapter_tokens = frame_position_encoder(frame_positions)
                    frame_adapter_pooled_flat = frame_adapter_pooled.reshape(frames_flat.shape[0], -1)

                if not logged_shapes:
                    log_first_step_shapes(
                        accelerator,
                        {
                            "frames": tuple(frames.shape),
                            "frames_flat": tuple(frames_flat.shape),
                            "frame_positions": tuple(frame_positions.shape),
                            "frame_positions_flat": tuple(frame_positions_flat.shape),
                            "latents": tuple(latents.shape),
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
                            "modified_pooled_prompt_embeds": tuple(
                                modified_pooled_prompt_embeds.shape
                            ),
                            "add_time_ids": tuple(add_time_ids.shape),
                        },
                    )
                    logged_shapes = True

                set_video_resnet_context(
                    unet,
                    num_frames=frames.shape[1],
                    frame_positions=frame_positions_flat,
                    frame_embeddings=frame_adapter_pooled_flat,
                )
                set_video_attention_context(
                    unet,
                    num_frames=frames.shape[1],
                    frame_tokens=frame_adapter_tokens,
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
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction_type={prediction_type!r}.")

                loss = torch_f.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                avg_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                if global_step % logging_steps == 0:
                    metrics = {
                        "train/loss": avg_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": global_step / num_update_steps_per_epoch,
                        "train/global_step": global_step,
                        "train/samples_seen_frames": global_step * effective_batch_frames,
                    }
                    gpu_memory_mb = get_gpu_memory_mb()
                    if gpu_memory_mb is not None:
                        metrics["system/gpu_max_memory_allocated_mb"] = gpu_memory_mb
                    if tracker_is_enabled(config):
                        accelerator.log(metrics, step=global_step)
                    accelerator.print(f"step {global_step}: loss={avg_loss:.6f}")

                if (
                    int(training_config.get("checkpointing_steps", 0)) > 0
                    and global_step % int(training_config["checkpointing_steps"]) == 0
                ):
                    save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        unet=unet,
                        temporal_mlp=temporal_mlp,
                        frame_position_encoder=frame_position_encoder,
                        config=config,
                        temporal_config=temporal_config,
                        accelerator=accelerator,
                    )
                    accelerator.print(f"saved checkpoint at step {global_step}")
                    if tracker_is_enabled(config):
                        accelerator.log({"checkpoint/saved_step": global_step}, step=global_step)
                    accelerator.wait_for_everyone()

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
                            output_dir / validation_config.get("output_dir", "validation") / f"step_{global_step:04d}"
                        )
                        generate_video_frames(
                            pipe=pipe,
                            temporal_mlp=accelerator.unwrap_model(temporal_mlp),
                            frame_position_encoder=(
                                accelerator.unwrap_model(frame_position_encoder)
                                if frame_position_encoder is not None
                                else None
                            ),
                            prompt=validation_config["prompt"],
                            num_frames=int(validation_config["num_frames"]),
                            output_dir=validation_dir,
                            resolution=resolution,
                            temporal_alpha=temporal_alpha,
                            injection_mode=injection_mode,
                            num_inference_steps=int(validation_config.get("num_inference_steps", 30)),
                            guidance_scale=float(validation_config.get("guidance_scale", 7.5)),
                            seed=training_config.get("seed"),
                            batch_size=validation_config.get("batch_size"),
                            save_grid=bool(validation_config.get("save_grid", True)),
                            save_video=bool(validation_config.get("save_mp4", False)),
                        )
                        accelerator.print(f"saved validation frames to {validation_dir}")
                        if tracker_is_enabled(config):
                            accelerator.log(
                                {"validation/generated_frames": int(validation_config["num_frames"])},
                                step=global_step,
                            )
                        log_validation_media(accelerator, config, validation_dir, global_step)
                    accelerator.wait_for_everyone()

                if global_step >= max_train_steps:
                    break

    accelerator.wait_for_everyone()
    save_checkpoint(
        output_dir=output_dir,
        step=global_step,
        unet=unet,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        config=config,
        temporal_config=temporal_config,
        accelerator=accelerator,
    )
    accelerator.print(f"training complete at step {global_step}; saved checkpoint-last")
    accelerator.end_training()


if __name__ == "__main__":
    main()
