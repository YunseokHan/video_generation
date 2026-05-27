from __future__ import annotations

import importlib
import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

try:
    import pytest
except ImportError:  # pragma: no cover - lets the video env run smoke tests without pytest.
    pytest = None

from framegen.data import (
    DEFAULT_OPENVID_ROOT,
    PlaceholderVideoDataset,
    OpenVidVideoDataset,
    flatten_video_batch,
    make_frame_positions,
    video_collate_fn,
)
from framegen.config import load_config
from framegen.env import get_hf_cache_dir, get_hf_token, load_env_file
from framegen.generation import guidance_scale_label, write_caption_file
from framegen.image_first_generation import (
    _scheduler_switch_noise_level,
    image_first_output_dir,
    t1_ratio_label,
)
from framegen.latent_calibrator import (
    TemporalConvLatentCalibrator,
    latent_calibrator_alignment_loss,
    latent_calibrator_norm_loss,
)
from framegen.utils import save_mp4
from framegen.sdxl import (
    add_temporal_embedding_to_pooled_prompt_embeds,
    compute_sdxl_time_ids,
)
from framegen.temporal import (
    FramePositionMLP,
    FramePositionTokenEncoder,
    SinusoidalFramePositionEncoder,
    apply_frame_token_conditioning,
)
from framegen.video_attention import VideoBasicTransformerBlock, set_video_attention_context
from framegen.video_resnet import VideoResnetBlock2D, set_video_resnet_context
from infer import resolve_checkpoint_dir, resolve_config_path, resolve_guidance_scales, resolve_output_dir
from train import (
    LATENT_INIT_FIRST_FRAME_REPEAT,
    _expand_rollout_switch_level,
    compute_clean_latent_epsilon_target,
    compute_image_first_bridge_mask,
    generate_timestep_weights,
    normalize_image_first_bridge_mode,
    normalize_image_first_noise_mode,
    normalize_latent_init_mode,
    repeat_first_frame_latents,
    rollout_image_first_anchor_latents,
    sample_image_first_noise,
    sample_video_timesteps,
    move_video_frames_to_device,
)


def _importorskip(module_name: str):
    if pytest is not None:
        return pytest.importorskip(module_name)
    return importlib.import_module(module_name)


def main() -> int:
    if pytest is not None:
        return pytest.main([str(Path(__file__).resolve()), *sys.argv[1:]])

    smoke_tests = [
        test_frame_positions_single_and_uniform,
        test_flatten_video_batch_repeats_captions_and_positions,
        test_uint8_frame_storage_moves_to_normalized_float,
        test_temporal_mlp_matches_pooled_shape,
        test_frame_position_token_encoder_shapes,
        test_sinusoidal_frame_position_encoder_shapes,
        test_add_to_text_frame_token_conditioning_shapes,
        test_concat_frame_token_conditioning_shapes,
        test_sdxl_time_ids_shape_and_values,
        test_train_configs_share_same_schema,
        test_sample_video_timesteps_shared_per_video,
        test_timestep_bias_weights_preserve_shared_video_timesteps,
        test_image_first_latent_repeat_and_target,
        test_image_first_snr_bridge_mask,
        test_image_first_rollout_source_shape,
        test_image_first_shared_noise_sampler,
        test_image_first_switch_noise_level_from_sigmas,
        test_latent_calibrator_zero_init_and_aux_losses,
        test_infer_name_step_path_resolution,
        test_guidance_scale_label_and_caption_file,
        test_infer_guidance_scale_resolution,
        test_save_mp4_tiny_clip_when_backend_available,
        test_openvid_dataset_reads_video_sample_when_available,
        test_video_resnet_adapter_initially_matches_base,
        test_video_attention_adapter_initially_matches_base,
    ]
    for test_fn in smoke_tests:
        test_fn()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_env_file_loads_hf_aliases(Path(temp_dir))
    print("all core tests ok")
    return 0


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
    assert batch["original_sizes"].shape == (2, 2)
    assert batch["crop_top_lefts"].shape == (2, 2)
    assert batch["target_sizes"].shape == (2, 2)


def test_uint8_frame_storage_moves_to_normalized_float() -> None:
    dataset = PlaceholderVideoDataset(
        num_videos=1,
        num_frames_per_video=2,
        resolution=8,
        frame_storage_dtype="uint8",
    )
    item = dataset[0]
    assert item["frames"].dtype == torch.uint8
    batch = video_collate_fn([item])
    moved = move_video_frames_to_device(
        batch["frames"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert moved.dtype == torch.float32
    assert moved.shape == (1, 2, 3, 8, 8)
    assert -1.01 <= float(moved.min()) <= 1.01
    assert -1.01 <= float(moved.max()) <= 1.01


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


def test_add_to_text_frame_token_conditioning_shapes() -> None:
    prompt_embeds = torch.zeros(6, 7, 5)
    frame_embeddings = torch.ones(2, 3, 5)
    frame_tokens = torch.randn(2, 3, 4, 5)
    conditioned, temporal_tokens = apply_frame_token_conditioning(
        prompt_embeds=prompt_embeds,
        frame_embeddings=frame_embeddings,
        frame_tokens=frame_tokens,
        num_frames=3,
        mode="add_to_text",
        alpha=0.5,
    )
    assert conditioned.shape == (6, 7, 5)
    assert temporal_tokens is not None
    assert temporal_tokens.shape == (2, 3, 7, 5)
    assert torch.allclose(conditioned, torch.full_like(conditioned, 0.5))


def test_concat_frame_token_conditioning_shapes() -> None:
    prompt_embeds = torch.zeros(6, 7, 5)
    frame_embeddings = torch.ones(2, 3, 5)
    frame_tokens = torch.ones(2, 3, 4, 5)
    conditioned, temporal_tokens = apply_frame_token_conditioning(
        prompt_embeds=prompt_embeds,
        frame_embeddings=frame_embeddings,
        frame_tokens=frame_tokens,
        num_frames=3,
        mode="concat_tokens",
        alpha=2.0,
    )
    assert conditioned.shape == (6, 11, 5)
    assert temporal_tokens is not None
    assert temporal_tokens.shape == (2, 3, 4, 5)
    assert torch.allclose(conditioned[:, :7], torch.zeros(6, 7, 5))
    assert torch.allclose(conditioned[:, 7:], torch.full((6, 4, 5), 2.0))


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


def _schema_keys(value, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if not isinstance(value, dict):
        return {prefix}
    keys: set[tuple[str, ...]] = {prefix} if prefix else set()
    for key, child in value.items():
        keys.update(_schema_keys(child, (*prefix, str(key))))
    return keys


def test_train_configs_share_same_schema() -> None:
    paths = sorted(Path("configs/train").glob("*.yaml"))
    assert paths
    schemas = {path: _schema_keys(load_config(path)) for path in paths}
    reference_path = paths[0]
    reference_schema = schemas[reference_path]
    for path, schema in schemas.items():
        assert schema == reference_schema, (
            f"{path} schema mismatch against {reference_path}: "
            f"missing={sorted(reference_schema - schema)}, extra={sorted(schema - reference_schema)}"
        )


def test_sample_video_timesteps_shared_per_video() -> None:
    timesteps = sample_video_timesteps(
        batch_size=4,
        num_frames=3,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert timesteps.shape == (12,)
    grouped = timesteps.reshape(4, 3)
    assert torch.equal(grouped[:, 0], grouped[:, 1])
    assert torch.equal(grouped[:, 0], grouped[:, 2])
    assert int(timesteps.min()) >= 0
    assert int(timesteps.max()) < 1000


def test_timestep_bias_weights_preserve_shared_video_timesteps() -> None:
    weights = generate_timestep_weights(
        {
            "timestep_bias_strategy": "range",
            "timestep_bias_begin": 2,
            "timestep_bias_end": 5,
            "timestep_bias_multiplier": 3.0,
            "timestep_bias_portion": 0.25,
        },
        num_timesteps=10,
    )
    assert weights.shape == (10,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    timesteps = sample_video_timesteps(
        batch_size=5,
        num_frames=4,
        num_train_timesteps=10,
        device=torch.device("cpu"),
        timestep_weights=weights,
    )
    grouped = timesteps.reshape(5, 4)
    assert torch.equal(grouped[:, 0], grouped[:, 1])
    assert torch.equal(grouped[:, 0], grouped[:, 2])
    assert torch.equal(grouped[:, 0], grouped[:, 3])


def test_image_first_latent_repeat_and_target() -> None:
    latents = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(6, 1, 2, 2)
    repeated = repeat_first_frame_latents(latents, batch_size=2, num_frames=3)
    grouped = repeated.reshape(2, 3, 1, 2, 2)
    original = latents.reshape(2, 3, 1, 2, 2)
    assert torch.equal(grouped[:, 0], original[:, 0])
    assert torch.equal(grouped[:, 1], original[:, 0])
    assert torch.equal(grouped[:, 2], original[:, 0])
    assert normalize_latent_init_mode("image_first") == LATENT_INIT_FIRST_FRAME_REPEAT

    class DummyScheduler:
        alphas_cumprod = torch.tensor([0.25, 0.64], dtype=torch.float32)

    clean = torch.full((2, 1, 2, 2), 2.0)
    target = torch.full_like(clean, 0.5)
    timesteps = torch.tensor([0, 1])
    alpha = DummyScheduler.alphas_cumprod[timesteps].reshape(2, 1, 1, 1)
    noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * target
    recovered = compute_clean_latent_epsilon_target(
        noisy_latents=noisy,
        clean_latents=clean,
        noise_scheduler=DummyScheduler(),
        timesteps=timesteps,
    )
    assert torch.allclose(recovered, target)


def test_image_first_snr_bridge_mask() -> None:
    class DummyScheduler:
        alphas_cumprod = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float32)

    timesteps = torch.tensor([0, 1, 2], dtype=torch.long)
    mask = compute_image_first_bridge_mask(
        noise_scheduler=DummyScheduler(),
        timesteps=timesteps,
        bridge_mode="snr",
        snr_min=None,
        snr_max=1.0,
    )
    assert torch.equal(mask, torch.tensor([False, True, True]))
    assert normalize_image_first_bridge_mode("snr_hybrid") == "snr"
    assert normalize_image_first_bridge_mode("anchor_rollout") == "rollout"
    assert normalize_image_first_bridge_mode("snr_rollout") == "rollout_snr"

    rollout_mask = compute_image_first_bridge_mask(
        noise_scheduler=DummyScheduler(),
        timesteps=timesteps,
        bridge_mode="rollout_snr",
        snr_min=None,
        snr_max=1.0,
    )
    assert torch.equal(rollout_mask, mask)


def test_image_first_rollout_source_shape() -> None:
    DDPMScheduler = _importorskip("diffusers").DDPMScheduler

    class DummyOutput:
        def __init__(self, sample: torch.Tensor) -> None:
            self.sample = sample

    class DummyUnet(torch.nn.Module):
        def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            added_cond_kwargs: dict | None = None,
        ) -> DummyOutput:
            return DummyOutput(torch.zeros_like(sample))

    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    anchors = torch.randn(2, 4, 4, 4)
    timesteps = torch.tensor([3, 8], dtype=torch.long)
    prompt_embeds = torch.randn(2, 5, 8)
    pooled_prompt_embeds = torch.randn(2, 8)
    time_ids = torch.randn(2, 6)
    rollout = rollout_image_first_anchor_latents(
        unet=DummyUnet(),
        noise_scheduler=scheduler,
        anchor_latents=anchors,
        target_timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        add_time_ids=time_ids,
        rollout_steps=2,
    )
    assert rollout.shape == anchors.shape
    assert rollout.dtype == anchors.dtype

    reference = torch.zeros(6, 4, 8, 8)
    switch_level = _expand_rollout_switch_level(
        torch.tensor([0.25, 0.5]),
        batch_size=2,
        num_frames=3,
        reference=reference,
    )
    assert switch_level.shape == (6, 1, 1, 1)
    assert torch.equal(switch_level[:3], torch.full((3, 1, 1, 1), 0.25))
    assert torch.equal(switch_level[3:], torch.full((3, 1, 1, 1), 0.5))


def test_image_first_shared_noise_sampler() -> None:
    reference = torch.zeros(2 * 3, 1, 2, 2)
    noise, shared_fraction = sample_image_first_noise(
        reference=reference,
        batch_size=2,
        num_frames=3,
        noise_mode="shared",
        shared_noise_prob=1.0,
        noise_offset=0.0,
    )
    grouped = noise.reshape(2, 3, 1, 2, 2)
    assert shared_fraction == 1.0
    assert torch.equal(grouped[:, 0], grouped[:, 1])
    assert torch.equal(grouped[:, 0], grouped[:, 2])

    mixed_noise, mixed_fraction = sample_image_first_noise(
        reference=reference,
        batch_size=2,
        num_frames=3,
        noise_mode="mixed",
        shared_noise_prob=1.0,
        noise_offset=0.0,
    )
    mixed_grouped = mixed_noise.reshape(2, 3, 1, 2, 2)
    assert mixed_fraction == 1.0
    assert torch.equal(mixed_grouped[:, 0], mixed_grouped[:, 1])
    assert normalize_image_first_noise_mode("same_noise") == "shared"


def test_image_first_switch_noise_level_from_sigmas() -> None:
    class DummyScheduler:
        sigmas = torch.tensor([3.0, 2.0, 1.0, 0.0])

    timesteps = torch.tensor([999, 500, 0])
    assert torch.equal(
        _scheduler_switch_noise_level(
            DummyScheduler(),
            timesteps=timesteps,
            step_index=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        torch.tensor(2.0),
    )
    assert torch.equal(
        _scheduler_switch_noise_level(
            DummyScheduler(),
            timesteps=timesteps,
            step_index=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        torch.tensor(0.0),
    )


def test_latent_calibrator_zero_init_and_aux_losses() -> None:
    class DummyScheduler:
        alphas_cumprod = torch.linspace(0.95, 0.05, steps=10)

    torch.manual_seed(0)
    calibrator = TemporalConvLatentCalibrator(
        {
            "enabled": True,
            "hidden_channels": 8,
            "num_blocks": 1,
            "timestep_embedding_dim": 8,
            "prompt_embedding_dim": 16,
            "residual": {
                "scale_mode": "clipped_snr",
                "max_scale": 0.5,
                "norm_cap": 1.0,
            },
        }
    )
    noisy = torch.randn(6, 4, 8, 8)
    anchor = torch.randn_like(noisy)
    timesteps = torch.tensor([1, 1, 1, 5, 5, 5], dtype=torch.long)
    frame_positions = make_frame_positions(3).repeat(2)
    pooled_prompt_embeds = torch.randn(6, 16)

    out, delta, scale = calibrator(
        noisy_latents=noisy,
        anchor_latents=anchor,
        timesteps=timesteps,
        frame_positions=frame_positions,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_frames=3,
        noise_scheduler=DummyScheduler(),
    )

    assert out.shape == noisy.shape
    assert delta.shape == noisy.shape
    assert scale.shape == (6, 1, 1, 1)
    assert torch.allclose(out, noisy, atol=1e-6)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)

    mask = torch.tensor([True, False, True, True, False, True])
    assert torch.equal(
        latent_calibrator_alignment_loss(out, out, mask),
        torch.tensor(0.0),
    )
    norm_delta = torch.zeros_like(delta, requires_grad=True)
    norm_loss = latent_calibrator_norm_loss(norm_delta, mask, norm_cap=1.0)
    assert torch.equal(norm_loss, torch.tensor(0.0))
    norm_loss.backward()
    assert norm_delta.grad is not None
    assert torch.isfinite(norm_delta.grad).all()


def test_infer_name_step_path_resolution(tmp_path: Path | None = None) -> None:
    if tmp_path is None:
        tmp_context = tempfile.TemporaryDirectory()
        root = Path(tmp_context.name)
    else:
        tmp_context = None
        root = tmp_path
    try:
        args = argparse.Namespace(
            checkpoint=None,
            name="run-a",
            step="25",
            output_dir=None,
            config=None,
        )
        checkpoint = root / "outputs" / "run-a" / "checkpoint-25"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        assert resolve_checkpoint_dir(args, root) == checkpoint
        assert resolve_output_dir(args, root) == root / "outputs" / "infer" / "run-a-25"
        assert resolve_config_path(args, checkpoint, root) == checkpoint / "config.yaml"

        args.step = "last"
        assert resolve_checkpoint_dir(args, root) == root / "outputs" / "run-a" / "checkpoint-last"
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


def test_guidance_scale_label_and_caption_file(tmp_path: Path | None = None) -> None:
    if tmp_path is None:
        tmp_context = tempfile.TemporaryDirectory()
        root = Path(tmp_context.name)
    else:
        tmp_context = None
        root = tmp_path
    try:
        assert guidance_scale_label(1.0) == "cfg_1"
        assert guidance_scale_label(8.0) == "cfg_8"
        assert guidance_scale_label(7.5) == "cfg_7p5"
        assert t1_ratio_label(0.25) == "t1_0p25"
        assert image_first_output_dir(root, 0.25, 8.0) == root / "t1_0p25" / "cfg_8"
        caption_path = write_caption_file(root / "sample", "A test caption.")
        assert caption_path.name == "caption.txt"
        assert caption_path.read_text(encoding="utf-8") == "A test caption.\n"
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


def test_infer_guidance_scale_resolution() -> None:
    base_args = argparse.Namespace(guidance_scale=None, guidance_scales=None)
    assert resolve_guidance_scales(base_args, {"guidance_scales": [1.0, 8.0]}) == [1.0, 8.0]

    base_args.guidance_scales = ["1", "8"]
    assert resolve_guidance_scales(base_args, {}) == [1.0, 8.0]

    base_args.guidance_scales = None
    base_args.guidance_scale = 8.0
    assert resolve_guidance_scales(base_args, {"guidance_scales": [1.0, 8.0]}) == [8.0]


def test_save_mp4_tiny_clip_when_backend_available(tmp_path: Path | None = None) -> None:
    if tmp_path is None:
        tmp_context = tempfile.TemporaryDirectory()
        root = Path(tmp_context.name)
    else:
        tmp_context = None
        root = tmp_path
    try:
        from PIL import Image

        frames = [
            Image.new("RGB", (32, 32), (index * 50, 0, 0))
            for index in range(3)
        ]
        output = root / "tiny.mp4"
        ok = save_mp4(frames, output, fps=2)
        if ok:
            assert output.exists()
            assert output.stat().st_size > 0
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


def test_openvid_dataset_reads_video_sample_when_available() -> None:
    root = Path(DEFAULT_OPENVID_ROOT)
    if not (root / "OpenVid.csv").exists():
        return
    dataset = OpenVidVideoDataset(
        root=root,
        num_frames_per_video=3,
        resolution=32,
        max_videos=1,
        min_frames=3,
        min_seconds=0.1,
    )
    item = dataset[0]
    assert item["frames"].shape == (3, 3, 32, 32)
    assert item["frames"].dtype == torch.float32
    assert -1.01 <= float(item["frames"].min()) <= 1.01
    assert -1.01 <= float(item["frames"].max()) <= 1.01
    assert item["frame_positions"].shape == (3,)
    assert item["caption"]
    assert len(item["original_size"]) == 2
    assert len(item["crop_top_left"]) == 2
    assert item["target_size"] == (32, 32)


def test_video_resnet_adapter_initially_matches_base() -> None:
    ResnetBlock2D = _importorskip("diffusers.models.resnet").ResnetBlock2D

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
    BasicTransformerBlock = _importorskip("diffusers.models.attention").BasicTransformerBlock

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


if __name__ == "__main__":
    raise SystemExit(main())
