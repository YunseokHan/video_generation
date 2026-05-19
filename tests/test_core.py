from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from framegen.data import flatten_video_batch, make_frame_positions, video_collate_fn
from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file
from framegen.sdxl import (
    add_temporal_embedding_to_pooled_prompt_embeds,
    compute_sdxl_time_ids,
)
from framegen.temporal import (
    FramePositionMLP,
    FramePositionTokenEncoder,
    SinusoidalFramePositionEncoder,
)
from framegen.video_attention import VideoBasicTransformerBlock, set_video_attention_context
from framegen.video_resnet import VideoResnetBlock2D, set_video_resnet_context


def test_frame_positions_single_and_uniform() -> None:
    assert torch.equal(make_frame_positions(1), torch.zeros(1))
    assert torch.allclose(make_frame_positions(4), torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))


def test_flatten_video_batch_repeats_captions_and_positions() -> None:
    items = [
        {
            "frames": torch.zeros(3, 3, 8, 8),
            "caption": "first",
            "frame_positions": torch.tensor([0.0, 0.5, 1.0]),
        },
        {
            "frames": torch.ones(3, 3, 8, 8),
            "caption": "second",
            "frame_positions": torch.tensor([0.0, 0.5, 1.0]),
        },
    ]
    batch = video_collate_fn(items)
    frames_flat, captions_flat, positions_flat = flatten_video_batch(batch)
    assert frames_flat.shape == (6, 3, 8, 8)
    assert captions_flat == ["first", "first", "first", "second", "second", "second"]
    assert torch.allclose(positions_flat, torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]))


def test_temporal_mlp_matches_pooled_shape() -> None:
    pooled = torch.zeros(5, 7)
    positions = torch.linspace(0.0, 1.0, steps=5)
    mlp = FramePositionMLP(output_dim=7, hidden_dim=11, num_layers=2)
    modified, frame_embeds = add_temporal_embedding_to_pooled_prompt_embeds(
        pooled_prompt_embeds=pooled,
        frame_positions=positions,
        frame_position_mlp=mlp,
        alpha=0.5,
    )
    assert frame_embeds.shape == pooled.shape
    assert modified.shape == pooled.shape


def test_frame_position_token_encoder_shapes() -> None:
    encoder = FramePositionTokenEncoder(
        embedding_dim=13,
        hidden_dim=17,
        num_layers=2,
        num_tokens=3,
    )
    positions = torch.linspace(0.0, 1.0, steps=8).reshape(2, 4)
    pooled, tokens = encoder(positions)
    assert pooled.shape == (2, 4, 13)
    assert tokens.shape == (2, 4, 3, 13)


def test_sinusoidal_frame_position_encoder_shapes() -> None:
    encoder = SinusoidalFramePositionEncoder(embedding_dim=12)
    positions = torch.linspace(0.0, 1.0, steps=8).reshape(2, 4)
    pooled, tokens = encoder(positions)
    assert pooled.shape == (2, 4, 12)
    assert tokens.shape == (2, 4, 1, 12)


def test_sdxl_time_ids_shape_and_values() -> None:
    time_ids = compute_sdxl_time_ids(
        original_size=(1024, 1024),
        crop_coords=(0, 0),
        target_size=(1024, 1024),
        batch_size=3,
        device="cpu",
        dtype=torch.float32,
    )
    assert time_ids.shape == (3, 6)
    assert torch.equal(time_ids[0], torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.float32))


def test_video_resnet_adapter_initially_matches_base() -> None:
    ResnetBlock2D = pytest.importorskip("diffusers.models.resnet").ResnetBlock2D

    torch.manual_seed(0)
    base = ResnetBlock2D(
        in_channels=4,
        out_channels=8,
        temb_channels=16,
        groups=4,
        groups_out=4,
    )
    base.eval()
    sample = torch.randn(6, 4, 8, 8)
    temb = torch.randn(6, 16)
    with torch.no_grad():
        expected = base(sample, temb)

    adapter = VideoResnetBlock2D(base, frame_embedding_dim=1, active=True)
    adapter.eval()
    set_video_resnet_context(adapter, num_frames=3, frame_positions=torch.linspace(0.0, 1.0, 6))
    with torch.no_grad():
        actual = adapter(sample, temb)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_video_attention_adapter_initially_matches_base() -> None:
    BasicTransformerBlock = pytest.importorskip("diffusers.models.attention").BasicTransformerBlock

    torch.manual_seed(0)
    base = BasicTransformerBlock(
        dim=16,
        num_attention_heads=4,
        attention_head_dim=4,
        cross_attention_dim=32,
    )
    base.eval()
    hidden_states = torch.randn(6, 5, 16)
    encoder_hidden_states = torch.randn(6, 7, 32)
    with torch.no_grad():
        expected = base(hidden_states, encoder_hidden_states=encoder_hidden_states)

    adapter = VideoBasicTransformerBlock(base, active=True)
    adapter.eval()
    set_video_attention_context(adapter, num_frames=3)
    with torch.no_grad():
        actual = adapter(hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_env_file_loads_hf_aliases(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_TOKEN=token-value\nHF_CACHE=/tmp/framegen-hf-cache\n",
        encoding="utf-8",
    )
    previous = {key: os.environ.get(key) for key in ["HF_TOKEN", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"]}
    try:
        for key in previous:
            os.environ.pop(key, None)
        assert load_env_file(env_path, override=True)
        assert get_hf_token() == "token-value"
        assert get_hf_cache_dir() == "/tmp/framegen-hf-cache/hub"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
