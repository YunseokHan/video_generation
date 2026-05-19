from __future__ import annotations

import logging
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


def encode_prompts_with_sdxl_logic(
    prompts: str | Sequence[str],
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    clip_skip: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with the same hidden-state selection used by SDXL.

    This mirrors diffusers' SDXL encode_prompt path for positive prompts:
    tokenize for both SDXL tokenizers, take the penultimate hidden state from
    each encoder, concatenate them, and use the pooled output from the final
    text encoder as the added-conditioning text embedding.
    """

    if isinstance(prompts, str):
        prompt_batch = [prompts]
    else:
        prompt_batch = list(prompts)
    if not prompt_batch:
        raise ValueError("prompts must not be empty.")

    prompt_2_batch = prompt_batch
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
    prompt_batches = [prompt_batch, prompt_2_batch] if tokenizer is not None else [prompt_2_batch]

    prompt_embeds_list: list[torch.Tensor] = []
    pooled_prompt_embeds: torch.Tensor | None = None

    for current_prompts, current_tokenizer, current_text_encoder in zip(
        prompt_batches, tokenizers, text_encoders
    ):
        text_inputs = current_tokenizer(
            current_prompts,
            padding="max_length",
            max_length=current_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = current_tokenizer(
            current_prompts,
            padding="longest",
            return_tensors="pt",
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = current_tokenizer.batch_decode(
                untruncated_ids[:, current_tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "Prompt was truncated because CLIP can only handle %s tokens: %s",
                current_tokenizer.model_max_length,
                removed_text,
            )

        outputs = current_text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = outputs[0]
        if clip_skip is None:
            prompt_embeds = outputs.hidden_states[-2]
        else:
            prompt_embeds = outputs.hidden_states[-(clip_skip + 2)]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    if pooled_prompt_embeds is None:
        raise RuntimeError("SDXL prompt encoding did not produce pooled prompt embeddings.")

    if dtype is not None:
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    else:
        prompt_embeds = prompt_embeds.to(device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)

    return prompt_embeds, pooled_prompt_embeds


def compute_sdxl_time_ids(
    original_size: tuple[int, int],
    crop_coords: tuple[int, int],
    target_size: tuple[int, int],
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    add_time_ids = list(original_size + crop_coords + target_size)
    time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    return time_ids.repeat(batch_size, 1)


def validate_sdxl_added_conditioning_dimensions(
    unet,
    text_encoder_2,
    num_time_ids: int = 6,
) -> None:
    text_encoder_projection_dim = text_encoder_2.config.projection_dim
    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * num_time_ids + text_encoder_projection_dim
    )
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            "SDXL added-conditioning dimension mismatch: "
            f"UNet expects {expected_add_embed_dim}, but time_ids/text_embeds produce "
            f"{passed_add_embed_dim}."
        )


def add_temporal_embedding_to_pooled_prompt_embeds(
    pooled_prompt_embeds: torch.Tensor,
    frame_positions: torch.Tensor,
    frame_position_mlp,
    alpha: float,
    injection_mode: str = "add_to_pooled_prompt_embeds",
) -> tuple[torch.Tensor, torch.Tensor]:
    if injection_mode != "add_to_pooled_prompt_embeds":
        raise NotImplementedError(
            "Only temporal_conditioning.injection_mode='add_to_pooled_prompt_embeds' "
            "is implemented in this milestone."
        )

    frame_embeds = frame_position_mlp(frame_positions)
    frame_embeds = frame_embeds.to(device=pooled_prompt_embeds.device, dtype=pooled_prompt_embeds.dtype)
    assert frame_embeds.shape == pooled_prompt_embeds.shape

    modified_pooled_prompt_embeds = pooled_prompt_embeds + float(alpha) * frame_embeds
    assert modified_pooled_prompt_embeds.shape == pooled_prompt_embeds.shape
    return modified_pooled_prompt_embeds, frame_embeds
