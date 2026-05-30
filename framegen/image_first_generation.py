from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .data import make_frame_positions
from .generation import guidance_scale_label, write_caption_file
from .sdxl import add_temporal_embedding_to_pooled_prompt_embeds, compute_sdxl_time_ids
from .temporal import apply_frame_token_conditioning
from .utils import save_image_grid, save_mp4
from .video_attention import (
    clear_video_attention_context,
    iter_video_attention_blocks,
    set_video_attention_context,
)
from .video_resnet import (
    clear_video_resnet_context,
    iter_video_resnet_blocks,
    set_video_resnet_context,
)

IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE = "repeat_add_noise"
IMAGE_FIRST_SWITCH_PRED_X0_RENOISE = "pred_x0_renoise"
IMAGE_FIRST_RENOISE_INDEPENDENT = "independent"
IMAGE_FIRST_RENOISE_SHARED = "shared"


def t1_ratio_label(t1_ratio: float) -> str:
    value = float(t1_ratio)
    if value.is_integer():
        text = str(int(value))
    else:
        text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"t1_{text}"


def _snapshot_active_states(blocks: Iterable) -> list[tuple[object, bool]]:
    return [(block, bool(block.active)) for block in blocks]


def _set_active_states(snapshot: list[tuple[object, bool]], active: bool) -> None:
    for block, _ in snapshot:
        block.active = bool(active)


def _restore_active_states(snapshot: list[tuple[object, bool]]) -> None:
    for block, active in snapshot:
        block.active = bool(active)


def normalize_image_first_switch_mode(mode: str | None) -> str:
    normalized = str(mode or IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE).lower().replace("-", "_")
    aliases = {
        "": IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
        "repeat": IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
        "repeat_add_noise": IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
        "switch_noise": IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
        "legacy": IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
        "pred_x0": IMAGE_FIRST_SWITCH_PRED_X0_RENOISE,
        "pred_x0_renoise": IMAGE_FIRST_SWITCH_PRED_X0_RENOISE,
        "predict_x0_renoise": IMAGE_FIRST_SWITCH_PRED_X0_RENOISE,
        "matched_renoise": IMAGE_FIRST_SWITCH_PRED_X0_RENOISE,
    }
    if normalized not in aliases:
        raise ValueError(
            "validation.image_first_switch_mode must be repeat_add_noise or pred_x0_renoise, "
            f"got {mode!r}."
        )
    return aliases[normalized]


def normalize_image_first_renoise_mode(mode: str | None) -> str:
    normalized = str(mode or IMAGE_FIRST_RENOISE_INDEPENDENT).lower().replace("-", "_")
    aliases = {
        "": IMAGE_FIRST_RENOISE_INDEPENDENT,
        "independent": IMAGE_FIRST_RENOISE_INDEPENDENT,
        "frame_independent": IMAGE_FIRST_RENOISE_INDEPENDENT,
        "shared": IMAGE_FIRST_RENOISE_SHARED,
        "same": IMAGE_FIRST_RENOISE_SHARED,
        "same_noise": IMAGE_FIRST_RENOISE_SHARED,
    }
    if normalized not in aliases:
        raise ValueError(
            "validation.image_first_renoise_noise_mode must be independent or shared, "
            f"got {mode!r}."
        )
    return aliases[normalized]


def _prepare_latents(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
    scheduler,
    vae_scale_factor: int,
) -> torch.Tensor:
    shape = (batch_size, channels, height // vae_scale_factor, width // vae_scale_factor)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents * scheduler.init_noise_sigma


def _scheduler_switch_noise_level(
    scheduler,
    timesteps: torch.Tensor,
    step_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if step_index >= len(timesteps):
        return torch.zeros((), device=device, dtype=dtype)
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is not None:
        sigma_index = min(int(step_index), int(sigmas.shape[0]) - 1)
        return sigmas[sigma_index].to(device=device, dtype=dtype)

    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is not None:
        timestep = timesteps[step_index].to(device=alphas_cumprod.device).long()
        timestep = timestep.clamp(0, alphas_cumprod.shape[0] - 1)
        alpha_prod = alphas_cumprod[timestep].to(device=device, dtype=dtype)
        return (1.0 - alpha_prod).clamp_min(0.0).sqrt()

    return torch.ones((), device=device, dtype=dtype)


def _scheduler_sigma_for_step(
    scheduler,
    timesteps: torch.Tensor,
    step_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or step_index >= len(timesteps):
        return None
    sigma_index = min(int(step_index), int(sigmas.shape[0]) - 1)
    return sigmas[sigma_index].to(device=device, dtype=dtype)


def _predict_original_from_epsilon(
    scheduler,
    sample: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    scheduler_config = getattr(scheduler, "config", {})
    if isinstance(scheduler_config, dict):
        prediction_type = scheduler_config.get("prediction_type", "epsilon")
    else:
        prediction_type = getattr(scheduler_config, "prediction_type", "epsilon")
    if prediction_type != "epsilon":
        raise ValueError("pred_x0_renoise currently requires scheduler prediction_type='epsilon'.")
    sigma = _scheduler_sigma_for_step(
        scheduler=scheduler,
        timesteps=timesteps,
        step_index=step_index,
        device=sample.device,
        dtype=sample.dtype,
    )
    if sigma is not None:
        return sample - sigma * noise_pred

    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        raise ValueError("pred_x0_renoise requires scheduler.sigmas or scheduler.alphas_cumprod.")
    timestep = timesteps[step_index].to(device=alphas_cumprod.device).long()
    timestep = timestep.clamp(0, alphas_cumprod.shape[0] - 1)
    alpha_prod = alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
    while alpha_prod.ndim < sample.ndim:
        alpha_prod = alpha_prod.unsqueeze(-1)
    sqrt_alpha = alpha_prod.clamp_min(1.0e-12).sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_prod).clamp_min(1.0e-12).sqrt()
    return (sample - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha


def _sample_frame_noise(
    reference: torch.Tensor,
    num_frames: int,
    generator: torch.Generator | None,
    mode: str,
) -> torch.Tensor:
    normalized = normalize_image_first_renoise_mode(mode)
    if normalized == IMAGE_FIRST_RENOISE_SHARED:
        noise = torch.randn(
            (1, *reference.shape[1:]),
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        return noise.expand(int(num_frames), -1, -1, -1).contiguous()
    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )


def _renoise_clean_latents_for_step(
    scheduler,
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    step_index: int,
    noise_scale: float,
) -> torch.Tensor:
    sigma = _scheduler_sigma_for_step(
        scheduler=scheduler,
        timesteps=timesteps,
        step_index=step_index,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    if sigma is not None:
        return clean_latents + float(noise_scale) * sigma * noise

    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        return clean_latents + float(noise_scale) * noise
    timestep = timesteps[step_index].to(device=alphas_cumprod.device).long()
    timestep = timestep.clamp(0, alphas_cumprod.shape[0] - 1)
    alpha_prod = alphas_cumprod[timestep].to(device=clean_latents.device, dtype=clean_latents.dtype)
    while alpha_prod.ndim < clean_latents.ndim:
        alpha_prod = alpha_prod.unsqueeze(-1)
    return (
        alpha_prod.clamp_min(1.0e-12).sqrt() * clean_latents
        + float(noise_scale) * (1.0 - alpha_prod).clamp_min(0.0).sqrt() * noise
    )


def _cfg_unet_step(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    negative_add_time_ids: torch.Tensor | None,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    if do_cfg:
        if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None or negative_add_time_ids is None:
            raise ValueError("CFG requires negative prompt/text/time embeddings.")
        encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    else:
        encoder_hidden_states = prompt_embeds
        text_embeds = pooled_prompt_embeds
        time_ids = add_time_ids

    noise_pred = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        },
    ).sample
    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)
    return noise_pred


@torch.inference_mode()
def generate_image_first_video_frames(
    pipe,
    temporal_mlp,
    frame_position_encoder,
    latent_calibrator,
    latent_calibrator_config: dict | None,
    prompt: str,
    num_frames: int,
    output_dir: str | Path,
    resolution: int,
    temporal_alpha: float,
    t1_ratio: float,
    guidance_scale: float = 8.0,
    injection_mode: str = "add_to_pooled_prompt_embeds",
    frame_token_embedding_mode: str = "temporal_cross_attention_only",
    frame_token_alpha: float = 1.0,
    num_inference_steps: int = 30,
    seed: int | None = None,
    save_grid: bool = True,
    save_video: bool = True,
    fps: int = 8,
    switch_noise_scale: float = 0.1,
    switch_mode: str = IMAGE_FIRST_SWITCH_REPEAT_ADD_NOISE,
    renoise_noise_mode: str = IMAGE_FIRST_RENOISE_INDEPENDENT,
    renoise_noise_scale: float = 1.0,
    anchor_image_path: str | Path | None = None,
) -> list[Path]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not 0.0 <= float(t1_ratio) <= 1.0:
        raise ValueError("t1_ratio must be in [0, 1].")
    if float(switch_noise_scale) < 0.0:
        raise ValueError("switch_noise_scale must be non-negative.")
    if float(renoise_noise_scale) < 0.0:
        raise ValueError("renoise_noise_scale must be non-negative.")
    switch_mode = normalize_image_first_switch_mode(switch_mode)
    renoise_noise_mode = normalize_image_first_renoise_mode(renoise_noise_mode)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_caption_file(output_dir, prompt)

    device = pipe._execution_device
    dtype = getattr(pipe.unet, "dtype", None)
    if dtype is None:
        dtype = next(pipe.unet.parameters()).dtype
    do_cfg = guidance_scale > 1.0
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
    timesteps = pipe.scheduler.timesteps
    first_stage_steps = min(len(timesteps), max(0, int(round(float(t1_ratio) * len(timesteps)))))

    unet_resnet_state = _snapshot_active_states(iter_video_resnet_blocks(pipe.unet))
    unet_attention_state = _snapshot_active_states(iter_video_attention_blocks(pipe.unet))
    vae_decoder = getattr(pipe.vae, "decoder", pipe.vae)
    vae_resnet_state = _snapshot_active_states(iter_video_resnet_blocks(vae_decoder))

    try:
        (
            prompt_embeds_img,
            negative_prompt_embeds_img,
            pooled_prompt_embeds_img,
            negative_pooled_prompt_embeds_img,
        ) = pipe.encode_prompt(
            prompt=[prompt],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        add_time_ids_img = compute_sdxl_time_ids(
            original_size=(resolution, resolution),
            crop_coords=(0, 0),
            target_size=(resolution, resolution),
            batch_size=1,
            device=device,
            dtype=pooled_prompt_embeds_img.dtype,
        )

        latents = _prepare_latents(
            batch_size=1,
            channels=int(pipe.unet.config.in_channels),
            height=int(resolution),
            width=int(resolution),
            dtype=dtype,
            device=device,
            generator=generator,
            scheduler=pipe.scheduler,
            vae_scale_factor=int(pipe.vae_scale_factor),
        )

        _set_active_states(unet_resnet_state, active=False)
        _set_active_states(unet_attention_state, active=False)
        for timestep in timesteps[:first_stage_steps]:
            noise_pred = _cfg_unet_step(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds_img,
                pooled_prompt_embeds=pooled_prompt_embeds_img,
                add_time_ids=add_time_ids_img,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds_img,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_img,
                negative_add_time_ids=add_time_ids_img if do_cfg else None,
            )
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        # Single-image clean anchor [1, C, H, W] for persistent anchor conditioning
        # (Agenda A1). Prefer the pred_x0 estimate; fall back to the stage-1 latent.
        anchor_feature_latents = None
        if switch_mode == IMAGE_FIRST_SWITCH_PRED_X0_RENOISE and first_stage_steps < len(timesteps):
            switch_timestep = timesteps[first_stage_steps]
            noise_pred = _cfg_unet_step(
                pipe=pipe,
                latents=latents,
                timestep=switch_timestep,
                prompt_embeds=prompt_embeds_img,
                pooled_prompt_embeds=pooled_prompt_embeds_img,
                add_time_ids=add_time_ids_img,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds_img,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_img,
                negative_add_time_ids=add_time_ids_img if do_cfg else None,
            )
            anchor_clean_latents = _predict_original_from_epsilon(
                scheduler=pipe.scheduler,
                sample=latents,
                noise_pred=noise_pred,
                timesteps=timesteps,
                step_index=first_stage_steps,
            )
            anchor_feature_latents = anchor_clean_latents
            anchor_latents = anchor_clean_latents.expand(int(num_frames), -1, -1, -1).contiguous()
            switch_noise = _sample_frame_noise(
                reference=anchor_latents,
                num_frames=int(num_frames),
                generator=generator,
                mode=renoise_noise_mode,
            )
            latents = _renoise_clean_latents_for_step(
                scheduler=pipe.scheduler,
                clean_latents=anchor_latents,
                noise=switch_noise,
                timesteps=timesteps,
                step_index=first_stage_steps,
                noise_scale=float(renoise_noise_scale),
            )
        else:
            anchor_feature_latents = latents
            anchor_latents = latents.expand(int(num_frames), -1, -1, -1).contiguous()
            latents = anchor_latents
            if float(switch_noise_scale) > 0.0 and first_stage_steps < len(timesteps):
                switch_noise_level = _scheduler_switch_noise_level(
                    scheduler=pipe.scheduler,
                    timesteps=timesteps,
                    step_index=first_stage_steps,
                    device=device,
                    dtype=latents.dtype,
                )
                switch_noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=device,
                    dtype=latents.dtype,
                )
                latents = latents + float(switch_noise_scale) * switch_noise_level * switch_noise
        _restore_active_states(unet_resnet_state)
        _restore_active_states(unet_attention_state)

        frame_positions = make_frame_positions(num_frames, normalize=True).to(device)
        frame_adapter_pooled = None
        frame_adapter_tokens = None
        frame_attention_tokens = None
        if frame_position_encoder is not None:
            frame_adapter_pooled, frame_adapter_tokens = frame_position_encoder(frame_positions)

        prompts = [prompt for _ in range(num_frames)]
        (
            prompt_embeds_vid,
            negative_prompt_embeds_vid,
            pooled_prompt_embeds_vid,
            negative_pooled_prompt_embeds_vid,
        ) = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        pooled_prompt_embeds_vid, _ = add_temporal_embedding_to_pooled_prompt_embeds(
            pooled_prompt_embeds=pooled_prompt_embeds_vid,
            frame_positions=frame_positions,
            frame_position_mlp=temporal_mlp,
            alpha=temporal_alpha,
            injection_mode=injection_mode,
        )
        if negative_pooled_prompt_embeds_vid is not None:
            negative_pooled_prompt_embeds_vid, _ = add_temporal_embedding_to_pooled_prompt_embeds(
                pooled_prompt_embeds=negative_pooled_prompt_embeds_vid,
                frame_positions=frame_positions,
                frame_position_mlp=temporal_mlp,
                alpha=temporal_alpha,
                injection_mode=injection_mode,
            )
        if frame_position_encoder is not None:
            prompt_embeds_vid, frame_attention_tokens = apply_frame_token_conditioning(
                prompt_embeds=prompt_embeds_vid,
                frame_embeddings=frame_adapter_pooled,
                frame_tokens=frame_adapter_tokens,
                num_frames=num_frames,
                mode=frame_token_embedding_mode,
                alpha=frame_token_alpha,
            )
            if negative_prompt_embeds_vid is not None:
                negative_prompt_embeds_vid, _ = apply_frame_token_conditioning(
                    prompt_embeds=negative_prompt_embeds_vid,
                    frame_embeddings=frame_adapter_pooled,
                    frame_tokens=frame_adapter_tokens,
                    num_frames=num_frames,
                    mode=frame_token_embedding_mode,
                    alpha=frame_token_alpha,
                )

        add_time_ids_vid = compute_sdxl_time_ids(
            original_size=(resolution, resolution),
            crop_coords=(0, 0),
            target_size=(resolution, resolution),
            batch_size=num_frames,
            device=device,
            dtype=pooled_prompt_embeds_vid.dtype,
        )

        if latent_calibrator is not None and first_stage_steps < len(timesteps):
            calibrator_config = dict(latent_calibrator_config or {})
            apply_mode = str(calibrator_config.get("apply_mode", "switch_only"))
            if apply_mode != "switch_only":
                raise ValueError("Only latent_calibrator.apply_mode='switch_only' is supported for inference.")
            switch_timestep = timesteps[first_stage_steps].repeat(int(num_frames))
            latents, _, _ = latent_calibrator(
                noisy_latents=latents,
                anchor_latents=anchor_latents,
                timesteps=switch_timestep,
                frame_positions=frame_positions,
                pooled_prompt_embeds=pooled_prompt_embeds_vid,
                num_frames=int(num_frames),
                noise_scheduler=pipe.scheduler,
            )

        set_video_resnet_context(
            pipe.unet,
            num_frames=num_frames,
            frame_positions=frame_positions,
            frame_embeddings=frame_adapter_pooled,
        )
        set_video_attention_context(
            pipe.unet,
            num_frames=num_frames,
            frame_tokens=frame_attention_tokens,
            anchor_latents=anchor_feature_latents,
        )
        try:
            for timestep in timesteps[first_stage_steps:]:
                noise_pred = _cfg_unet_step(
                    pipe=pipe,
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds_vid,
                    pooled_prompt_embeds=pooled_prompt_embeds_vid,
                    add_time_ids=add_time_ids_vid,
                    guidance_scale=guidance_scale,
                    negative_prompt_embeds=negative_prompt_embeds_vid,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_vid,
                    negative_add_time_ids=add_time_ids_vid if do_cfg else None,
                )
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        finally:
            clear_video_resnet_context(pipe.unet)
            clear_video_attention_context(pipe.unet)

        set_video_resnet_context(
            vae_decoder,
            num_frames=num_frames,
            frame_positions=frame_positions,
            frame_embeddings=frame_adapter_pooled,
        )
        try:
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        finally:
            clear_video_resnet_context(vae_decoder)
        images = pipe.image_processor.postprocess(image, output_type="pil")
    finally:
        _restore_active_states(unet_resnet_state)
        _restore_active_states(unet_attention_state)
        _restore_active_states(vae_resnet_state)

    # Optionally decode and save the single-image stage-1 anchor (pred_x0 or the
    # stage-1 latent) for anchor-fidelity evaluation. Decoded with video adapters
    # off so it reflects the image the video stage was conditioned on.
    if anchor_image_path is not None and anchor_feature_latents is not None:
        _set_active_states(unet_resnet_state, active=False)
        _set_active_states(unet_attention_state, active=False)
        _set_active_states(vae_resnet_state, active=False)
        try:
            anchor_decoded = pipe.vae.decode(
                anchor_feature_latents.to(latents.dtype) / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            anchor_image = pipe.image_processor.postprocess(anchor_decoded, output_type="pil")[0]
        finally:
            _restore_active_states(unet_resnet_state)
            _restore_active_states(unet_attention_state)
            _restore_active_states(vae_resnet_state)
        Path(anchor_image_path).parent.mkdir(parents=True, exist_ok=True)
        anchor_image.save(anchor_image_path)

    output_paths: list[Path] = []
    for index, image in enumerate(images):
        path = output_dir / f"frame_{index:03d}.png"
        image.save(path)
        output_paths.append(path)

    if save_grid:
        save_image_grid(images, output_dir / "grid.png")
    if save_video:
        ok = save_mp4(images, output_dir / "video.mp4", fps=fps)
        if not ok:
            print("MP4 export failed; skipped video.mp4.")
    metadata = (
        f"prompt: {prompt}\n"
        f"t1_ratio: {float(t1_ratio):g}\n"
        f"guidance_scale: {float(guidance_scale):g}\n"
        f"num_inference_steps: {int(num_inference_steps)}\n"
        f"switch_noise_scale: {float(switch_noise_scale):g}\n"
        f"switch_mode: {switch_mode}\n"
        f"renoise_noise_mode: {renoise_noise_mode}\n"
        f"renoise_noise_scale: {float(renoise_noise_scale):g}\n"
    )
    (output_dir / "image_first_metadata.txt").write_text(metadata, encoding="utf-8")
    return output_paths


def image_first_output_dir(root: str | Path, t1_ratio: float, guidance_scale: float) -> Path:
    return Path(root) / t1_ratio_label(t1_ratio) / guidance_scale_label(guidance_scale)
