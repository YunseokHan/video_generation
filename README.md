# SDXL Frame Generator

Experimental SDXL frame-level video generator baseline:

```text
p_image(x) -> p_frame(x | c, t_f)
```

The repository keeps the pretrained SDXL backbone intact and adds frame-aware
adapters for UNet Resnet blocks, UNet attention blocks, VAE decoder Resnet
blocks, and frame-wise token conditioning.

## Maintenance Rule

Every implementation change must update the matching markdown section under
`information/` in the same change. Treat `information/` as the living design
spec for future sessions and coding agents.

Use these sections as the first stop before editing code:

- `information/0_overview.md`: status, module map, implemented vs not fully implemented
- `information/1_vae.md`: VAE and VAE decoder adapter behavior
- `information/2_unet_resnet.md`: UNet Resnet adapter behavior
- `information/3_attention.md`: temporal attention and temporal FFN behavior
- `information/4_frame_conditioning.md`: pooled and token frame conditioning
- `information/5_training_and_inference.md`: top-level entrypoints and checkpoint flow
- `information/6_backbone_compatibility.md`: verified SDXL backbone compatibility notes
- `information/7_openvid_dataloader.md`: OpenVid CSV/mp4 dataloader details
- `information/8_resource_estimates.md`: parameter counts and VRAM estimates
- `information/9_loss_and_timestep_schedule.md`: diffusion loss and timestep sampling
- `information/10_image_first_training.md`: first-frame-repeat latent objective and image-first inference
- `information/11_latent_calibrator.md`: zero-init anchor-to-video latent calibrator

When adding a new module or ablation, either update the relevant file above or
add a new numbered markdown file and link it here.

## Environment

The default scripts use the local `video` conda env:

```text
/NHNHOME/WORKSPACE/26moe001_D/miniconda3/envs/video/bin/python
```

The verified environment is:

```text
torch 2.12.0+cu130
diffusers 0.38.0
```

Diffusers `0.38.0` ignores `dtype=` in `DiffusionPipeline.from_pretrained(...)`.
Use `torch_dtype=` for actual bf16 loading:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

## Entrypoints

Training, inference, and tests live at the repository root:

```text
train.py
infer.py
test.py
```

Config is managed under `configs/`, split by responsibility:

```text
configs/train/       # model, data, optimizer, logging, and ablation settings
configs/accelerate/  # Accelerate launcher settings
```

## Configs

Default sinusoidal frame-token setup:

```bash
TRAIN_CONFIG=configs/train/default.yaml bash scripts/train.sh
```

Learnable frame-token ablation:

```bash
TRAIN_CONFIG=configs/train/learnable_frame_tokens.yaml bash scripts/train.sh
```

Image-first mixed-noise sinusoidal training:

```bash
bash scripts/train_image_first.sh
```

Image-first sinusoidal training without mixed training noise:

```bash
bash scripts/train_image_first_sinusoidal.sh
```

Image-first learnable frame-token ablation:

```bash
bash scripts/train_image_first_learnable.sh
```

Image-first SNR-gated hybrid training:

```bash
bash scripts/train_image_first_snr.sh
```

Image-first SNR-gated hybrid training with pred-x0 re-noise validation:

```bash
bash scripts/train_image_first_snr_renoise.sh
```

Image-first SNR-gated training with anchor-expansion calibrator:

```bash
bash scripts/train_image_first_snr_ea.sh
```

Image-first rollout-source training:

```bash
bash scripts/train_image_first_rollout.sh
```

Image-first rollout-source training with SNR-gated fallback:

```bash
bash scripts/train_image_first_rollout_snr.sh
```

Image-first smooth-SNR bridge:

```bash
bash scripts/train_image_first_smooth_snr.sh
```

Image-first smooth-SNR bridge with boundary loss:

```bash
bash scripts/train_image_first_smooth_snr_boundary.sh
```

Image-first smooth-SNR bridge with pred-x0 re-noise validation and boundary loss:

```bash
bash scripts/train_image_first_smooth_snr_renoise_boundary.sh
```

Experiment E1 — same bridge, temporal cross-attn removed
(see `information/claude-codex-discussion.md` §5):

```bash
bash scripts/train_image_first_smooth_snr_renoise_boundary_xnocross.sh
```

Experiment E2 — persistent anchor conditioning v1 + mid+up placement, as a
continuation fine-tune from E1. **Requires the code changes listed in the
E2 config header before the run is meaningful**:

```bash
bash scripts/train_image_first_smooth_snr_renoise_boundary_anchor.sh
```

VAE temporal coherence diagnostic (A2). Read-only measurement; runs in
~25 min for 100 clips and writes `metrics.json`, `per_clip.json`, and a
contact-sheet folder under `outputs/diagnostics/vae_temporal/`. See
`information/12_vae_temporal_diagnostic.md`:

```bash
bash scripts/diagnostic_vae_temporal.sh --smoke         # 32 clips, ~5 min
bash scripts/diagnostic_vae_temporal.sh --num_clips 100 # recommended
```

`configs/train/default.yaml` enables:

- UNet convolutional/Resnet video adapters
- UNet attention video adapters
- VAE decoder Resnet video adapters, active but not trained by default
- sinusoidal frame-wise token embedding with `token_embedding_mode: add_to_text`
- OpenVid video loading from `/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted`
- adapter-only training with the base SDXL UNet, VAE, and text encoders frozen
- official SDXL trainer options for fp32 VAE encode, LR scheduling/warmup,
  gradient clipping, optional Min-SNR weighting, timestep bias, noise offset,
  prompt dropout, TF32, xFormers, 8-bit Adam, checkpoint resume, and SDXL
  resize/crop `time_ids`

`configs/train/learnable_frame_tokens.yaml` keeps the same adapter setup but
changes frame-wise token embedding to learned tokens with
`token_embedding_mode: concat_tokens`.

`configs/train/image_first_mixed.yaml`,
`configs/train/image_first_sinusoidal.yaml`,
`configs/train/image_first_learnable_frame_tokens.yaml`,
`configs/train/image_first_snr.yaml`,
`configs/train/image_first_snr_renoise.yaml`,
`configs/train/image_first_snr_ea.yaml`,
`configs/train/image_first_rollout.yaml`,
`configs/train/image_first_rollout_snr.yaml`,
`configs/train/image_first_smooth_snr.yaml`,
`configs/train/image_first_smooth_snr_boundary.yaml`, and
`configs/train/image_first_smooth_snr_renoise_boundary.yaml` set
`training.latent_init_mode: "first_frame_repeat"`. In this mode training
corrupts repeated first-frame latents while the objective denoises to the full
ground-truth video latent. Validation runs CFG 8 at `t1` ratios
`0, 0.25, 0.5, 0.75`.

`configs/train/image_first_mixed.yaml` is the default image-first launcher
config. It uses `training.image_first_noise_mode: "mixed"` with
`image_first_shared_noise_prob: 0.5`, so half of the videos use frame-shared
noise and half use frame-independent noise.

`configs/train/image_first_snr.yaml` uses
`training.image_first_bridge_mode: "snr"` and applies the anchor bridge only
when SNR is at or below `image_first_bridge_snr_max`; high-SNR/near-clean
timesteps fall back to standard video DDPM noising.

`configs/train/image_first_snr_renoise.yaml` keeps the same hard SNR bridge as
`image_first_snr.yaml` but validates with
`validation.image_first_switch_mode: "pred_x0_renoise"`.

`configs/train/image_first_snr_ea.yaml` keeps the SNR-gated bridge and enables
the zero-init temporal-conv latent calibrator. The calibrator maps
anchor-expanded noisy latents toward the true-video noised latent distribution
before the UNet/video adapters, with a weak low-frequency alignment loss.

`configs/train/image_first_rollout.yaml` uses
`training.image_first_bridge_mode: "rollout"` and creates the source image
latent by noising the anchor, denoising it for a small number of base-SDXL
steps with video adapters inactive, then duplicating the resulting latent
across frames.

`configs/train/image_first_rollout_snr.yaml` uses
`training.image_first_bridge_mode: "rollout_snr"`. Timesteps inside the SNR
range use the rollout source above; timesteps outside the range fall back to
standard video DDPM noising and sampled epsilon targets.

`configs/train/image_first_smooth_snr.yaml` uses
`training.image_first_bridge_mode: "smooth_snr"`. It blends the anchor source
into the true video source with a cosine gate from SNR 1 to SNR 5, without
boundary loss or pred-x0 re-noise validation.

`configs/train/image_first_smooth_snr_boundary.yaml` adds the weak boundary
re-noising auxiliary loss at SNR 5 to the smooth-SNR bridge, while keeping the
legacy repeat-add-noise validation switch.

`configs/train/image_first_smooth_snr_renoise_boundary.yaml` uses
`training.image_first_bridge_mode: "smooth_snr"`. It blends the anchor source
into the true video source with a cosine gate from SNR 1 to SNR 5, adds a weak
boundary re-noising auxiliary loss at SNR 5, and validates with
`validation.image_first_switch_mode: "pred_x0_renoise"` so the image switch
latent is converted to `pred_x0` and re-noised at the scheduler's switch level.

`configs/accelerate/default.yaml` is the default multi-GPU launcher config and
uses 4 GPU processes (`gpu_ids: "0,1,2,3"`).

All files under `configs/train/` share the same key schema. Some keys are
ablation-specific and ignored outside their mode, but they remain present so
future edits can diff configs by value instead of by structure.

## Train

Default training uses `scripts/train.sh`, which owns both config choices:

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
bash scripts/train.sh
```

Override either config from the shell:

```bash
TRAIN_CONFIG=configs/train/learnable_frame_tokens.yaml \
ACCELERATE_CONFIG=configs/accelerate/default.yaml \
bash scripts/train.sh
```

Compatibility wrappers are still available:

```bash
bash scripts/train_multi_gpu.sh    # delegates to scripts/train.sh
bash scripts/train_single_gpu.sh   # delegates to scripts/train.sh with TRAIN_LAUNCHER=python
```

The shell wrappers read `.env`; use `ENV_FILE=.env.other` to select another
environment file. `CONFIG=...` still works as a legacy alias for
`TRAIN_CONFIG=...`, but new work should use `TRAIN_CONFIG`.

Fill these slots in `.env` before online training:

```text
HF_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_MODE=online
```

The OpenVid dataset location is also read from `.env`, so the same configs run
on any server by editing only this file:

```text
OPENVID_ROOT=/abs/path/to/OpenVid_extracted
OPENVID_CSV=          # optional; defaults to <OPENVID_ROOT>/OpenVid.csv
```

Resolution precedence is `OPENVID_ROOT` env var > a config's `data.openvid_root`
> the hardcoded default. All `configs/train/*.yaml` ship with
`openvid_root: null` so the env var is the single source of truth. Pass
`--openvid_root` to `diagnostics/vae_temporal_diagnostic.py` to override per
run (CLI > env).

When `logging.report_to: "wandb"` is enabled, training metrics such as loss,
learning rate, epoch progress, seen frames, parameter counts, checkpoint
events, and GPU memory are logged to W&B. Default validation runs every 1000
optimization steps, generates CFG 1 and CFG 8 validation videos, writes
`video.mp4` in separate folders, and logs them under `validation/cfg_1/video`
and `validation/cfg_8/video` when W&B is enabled. The training loop can
also periodically run inference on one caption from the current training batch
and log the generated grid/video under `train_caption/*`; configure this with
`logging.log_training_caption_inference` and
`logging.training_caption_inference_steps`.

Resume long runs with:

```yaml
training:
  resume_from_checkpoint: "latest"
```

`checkpoint-N/accelerator_state` stores optimizer, scheduler, model wrapper,
and RNG state; `checkpoint-last` is kept as the latest copy for inference and
resume convenience.

## Inference

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
PROMPT="Astronaut walking through a jungle, cold color palette" \
NUM_FRAMES=16 \
NAME=sdxl-resnet-attention-sinusoidal \
STEP=1000 \
bash scripts/generate_video_frames.sh
```

This loads `outputs/sdxl-resnet-attention-sinusoidal/checkpoint-1000` and writes
separate CFG folders under
`outputs/infer/sdxl-resnet-attention-sinusoidal-1000/`:

```text
cfg_1/
  caption.txt
  frame_000.png
  ...
  grid.png
  video.mp4
cfg_8/
  caption.txt
  frame_000.png
  ...
  grid.png
  video.mp4
```

The direct Python form
is:

```bash
python infer.py \
  --name sdxl-resnet-attention-sinusoidal \
  --step 1000 \
  --prompt "Astronaut walking through a jungle, cold color palette" \
  --num_frames 16
```

When `--config` is omitted, `infer.py` first tries
`outputs/{name}/checkpoint-{step}/config.yaml`, which makes copied checkpoint
folders portable to a validation server.

Image-first inference uses a separate entrypoint:

```bash
python infer_image_first.py \
  --name image-first-sinusoidal-res512-bs5 \
  --step 1000 \
  --prompt "Astronaut walking through a jungle, cold color palette" \
  --num_frames 8 \
  --t1 0.25 \
  --guidance_scale 8 \
  --switch_noise_scale 0.1
```

It writes to `outputs/infer_image_first/{name}-{step}/t1_*/cfg_*/` and keeps
`caption.txt` beside the generated frames, grid, and MP4.

## Tests

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
bash scripts/run_core_tests.sh -q
```

`test.py` can run directly even when `pytest` is not installed; if `pytest` is
available, it delegates to pytest.
