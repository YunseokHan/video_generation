# Image-First Training And Inference

This file documents the `first_frame_repeat` latent initialization mode added
for the image-first video ablation.

## Motivation

The standard video objective corrupts the ground-truth latent for every frame:

```text
z*_video = [z*1, z*2, ..., z*N]
z_t      = q(z_t | z*_video)
```

The image-first mode instead asks the model to start from a latent trajectory
anchored on the first frame, then denoise toward the full video:

```text
z_anchor = [z*1, z*1, ..., z*1]
z_t      = q(z_t | z_anchor)
target   = z*_video
```

This matches the intended inference path where SDXL first denoises one image
latent, then repeats that latent and lets the video adapters produce motion and
frame-specific content.

## Training Objective

Enable the mode with:

```yaml
training:
  latent_init_mode: "first_frame_repeat"
```

The dataloader and VAE still encode all video frames:

```text
frames_flat:   [B*F, 3, H, W]
clean_latents: [B*F, 4, H/8, W/8] = z*
```

Before forward noising, `train.py` reshapes to `[B, F, C, H/8, W/8]`, takes the
first frame latent per video, and repeats it across `F`:

```text
noising_latents = [z*1, z*1, ..., z*1]
```

Timesteps are still sampled once per video and repeated across frames. Noise is
independent per frame:

```text
t:       [B] -> [B*F]
epsilon: [B*F, 4, H/8, W/8]
z_t = sqrt(alpha_bar_t) * noising_latents
    + sqrt(1 - alpha_bar_t) * epsilon
```

Because the model is trained to denoise `z_t` to the full ground-truth video
latent `z*`, the epsilon target is not the sampled `epsilon`. It is the epsilon
that would make the scheduler's epsilon-to-x0 conversion reconstruct `z*`:

```text
target_epsilon =
    (z_t - sqrt(alpha_bar_t) * z*) / sqrt(1 - alpha_bar_t)
```

This is implemented by `compute_clean_latent_epsilon_target(...)`. The mode
currently requires scheduler `prediction_type: epsilon`; `train.py` raises if a
config tries to combine `first_frame_repeat` with `v_prediction` or `sample`.

## Validation Logic

When `latent_init_mode: first_frame_repeat`, `train.py` switches validation from
standard `generate_video_frames(...)` to
`generate_image_first_video_frames(...)`.

Validation uses:

```yaml
validation:
  guidance_scale: 8.0
  t1_ratios: [0.0, 0.25, 0.5, 0.75]
```

For each `t1_ratio`, the denoising schedule is split as:

```text
first_stage_steps = round(t1_ratio * num_inference_steps)
```

Stage 1:

- one latent is sampled,
- UNet video Resnet and attention adapters are set inactive,
- SDXL denoises the single image latent using only the base image model.

Stage 2:

- the current image latent is repeated to `[F, 4, H/8, W/8]`,
- adapter active states are restored,
- frame positions, pooled frame conditioning, and frame-token conditioning are
  applied,
- the remaining scheduler steps denoise as a video.

`t1_ratio: 0.0` skips stage 1. It starts all frames from the same initial noise
latent and uses the video model for the full denoising chain.

Validation outputs are written as:

```text
outputs/{run}/validation/step_1000/
  t1_0/cfg_8/
  t1_0p25/cfg_8/
  t1_0p5/cfg_8/
  t1_0p75/cfg_8/
```

Each folder contains `caption.txt`, `frame_*.png`, optional `grid.png`,
optional `video.mp4`, and `image_first_metadata.txt`.

## Inference

Use the separate entrypoint:

```bash
python infer_image_first.py \
  --name image-first-sinusoidal-res512-bs5 \
  --step 1000 \
  --prompt "A dog running through a grassy field, cinematic lighting" \
  --num_frames 8 \
  --t1 0.25 \
  --guidance_scale 8
```

The default output root is:

```text
outputs/infer_image_first/{name}-{step}/
```

The final generated media is placed under:

```text
t1_0p25/cfg_8/
```

`--step last` maps to `checkpoint-last`. `--checkpoint` and `--output_dir` are
available for copied checkpoint folders on another server. If `--config` is not
given, the script first reads `checkpoint/config.yaml`, which keeps copied
checkpoints portable.

## Configs And Scripts

Sinusoidal frame embedding:

```bash
bash scripts/train_image_first.sh
```

uses:

```text
configs/train/image_first_sinusoidal.yaml
```

Learnable frame-token ablation:

```bash
bash scripts/train_image_first_learnable.sh
```

uses:

```text
configs/train/image_first_learnable_frame_tokens.yaml
```

Both configs keep the base SDXL image model frozen by default and train only
the added temporal/adaptor parameters configured in `video_adapters`.
