from __future__ import annotations

from pathlib import Path

import torch

from .data import make_frame_positions
from .sdxl import add_temporal_embedding_to_pooled_prompt_embeds
from .temporal import apply_frame_token_conditioning
from .utils import save_image_grid, save_mp4
from .video_attention import clear_video_attention_context, set_video_attention_context
from .video_resnet import clear_video_resnet_context, set_video_resnet_context


def _slice_frame_token_context(
    frame_tokens: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if frame_tokens is None:
        return None
    if frame_tokens.ndim == 4:
        return frame_tokens[:, start:end]
    if frame_tokens.ndim == 3:
        return frame_tokens[start:end]
    raise ValueError(f"Unsupported frame token context shape: {tuple(frame_tokens.shape)}.")


@torch.inference_mode()
def generate_video_frames(
    pipe,
    temporal_mlp,
    frame_position_encoder,
    prompt: str,
    num_frames: int,
    output_dir: str | Path,
    resolution: int,
    temporal_alpha: float,
    injection_mode: str = "add_to_pooled_prompt_embeds",
    frame_token_embedding_mode: str = "temporal_cross_attention_only",
    frame_token_alpha: float = 1.0,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    batch_size: int | None = None,
    save_grid: bool = True,
    save_video: bool = False,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pipe._execution_device
    prompts = [prompt for _ in range(num_frames)]
    frame_positions = make_frame_positions(num_frames, normalize=True).to(device)
    frame_adapter_pooled = None
    frame_adapter_tokens = None
    frame_attention_tokens = None
    if frame_position_encoder is not None:
        frame_adapter_pooled, frame_adapter_tokens = frame_position_encoder(frame_positions)

    do_classifier_free_guidance = guidance_scale > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    modified_pooled_prompt_embeds, _ = add_temporal_embedding_to_pooled_prompt_embeds(
        pooled_prompt_embeds=pooled_prompt_embeds,
        frame_positions=frame_positions,
        frame_position_mlp=temporal_mlp,
        alpha=temporal_alpha,
        injection_mode=injection_mode,
    )
    if negative_pooled_prompt_embeds is not None:
        negative_pooled_prompt_embeds, _ = add_temporal_embedding_to_pooled_prompt_embeds(
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            frame_positions=frame_positions,
            frame_position_mlp=temporal_mlp,
            alpha=temporal_alpha,
            injection_mode=injection_mode,
        )

    if frame_position_encoder is not None:
        prompt_embeds, frame_attention_tokens = apply_frame_token_conditioning(
            prompt_embeds=prompt_embeds,
            frame_embeddings=frame_adapter_pooled,
            frame_tokens=frame_adapter_tokens,
            num_frames=num_frames,
            mode=frame_token_embedding_mode,
            alpha=frame_token_alpha,
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds, _ = apply_frame_token_conditioning(
                prompt_embeds=negative_prompt_embeds,
                frame_embeddings=frame_adapter_pooled,
                frame_tokens=frame_adapter_tokens,
                num_frames=num_frames,
                mode=frame_token_embedding_mode,
                alpha=frame_token_alpha,
            )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    batch_size = batch_size or num_frames
    images = []
    vae_decoder = getattr(pipe.vae, "decoder", pipe.vae)
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        set_video_resnet_context(
            pipe.unet,
            num_frames=end - start,
            frame_positions=frame_positions[start:end],
            frame_embeddings=(
                frame_adapter_pooled[start:end] if frame_adapter_pooled is not None else None
            ),
        )
        set_video_attention_context(
            pipe.unet,
            num_frames=end - start,
            frame_tokens=_slice_frame_token_context(frame_attention_tokens, start, end),
        )
        set_video_resnet_context(
            vae_decoder,
            num_frames=end - start,
            frame_positions=frame_positions[start:end],
            frame_embeddings=(
                frame_adapter_pooled[start:end] if frame_adapter_pooled is not None else None
            ),
        )
        try:
            generated = pipe(
                prompt_embeds=prompt_embeds[start:end],
                pooled_prompt_embeds=modified_pooled_prompt_embeds[start:end],
                negative_prompt_embeds=(
                    negative_prompt_embeds[start:end] if negative_prompt_embeds is not None else None
                ),
                negative_pooled_prompt_embeds=(
                    negative_pooled_prompt_embeds[start:end]
                    if negative_pooled_prompt_embeds is not None
                    else None
                ),
                height=resolution,
                width=resolution,
                original_size=(resolution, resolution),
                target_size=(resolution, resolution),
                crops_coords_top_left=(0, 0),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
        finally:
            clear_video_resnet_context(pipe.unet)
            clear_video_attention_context(pipe.unet)
            clear_video_resnet_context(vae_decoder)
        images.extend(generated)

    output_paths: list[Path] = []
    for index, image in enumerate(images):
        path = output_dir / f"frame_{index:03d}.png"
        image.save(path)
        output_paths.append(path)

    if save_grid:
        save_image_grid(images, output_dir / "grid.png")
    if save_video:
        ok = save_mp4(images, output_dir / "video.mp4")
        if not ok:
            print("MP4 export failed or imageio is unavailable; skipped video.mp4.")

    return output_paths
