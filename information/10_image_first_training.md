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

Timesteps are still sampled once per video and repeated across frames. The
default non-mixed mode uses independent noise per frame:

```text
t:       [B] -> [B*F]
epsilon: [B*F, 4, H/8, W/8]
z_t = sqrt(alpha_bar_t) * noising_latents
    + sqrt(1 - alpha_bar_t) * epsilon
```

The `image-first-mixed` config instead uses mixed noise sampling:

```yaml
training:
  image_first_noise_mode: "mixed"   # independent | shared | mixed
  image_first_shared_noise_prob: 0.5
```

For each video, `mixed` samples whether the video uses frame-independent noise
or frame-shared noise. Shared mode samples one noise tensor per video and
repeats it over frames:

```text
epsilon: [B, 1, 4, H/8, W/8] -> [B, F, 4, H/8, W/8]
```

This makes part of training match the image-first inference switch point, where
all frames start from the same duplicated latent. The target definition stays
the same.

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

## Bridge Source Modes

The image-first source distribution is controlled by:

```yaml
training:
  image_first_bridge_mode: "always"  # always | snr | rollout | rollout_snr
```

`always` is the original objective: every sampled timestep corrupts the
repeated first-frame latent and uses the clean-video epsilon target above.

`snr` is the SNR-gated hybrid objective used by
`configs/train/image_first_snr.yaml` and
`configs/train/image_first_snr_ea.yaml`:

```yaml
training:
  image_first_bridge_mode: "snr"
  image_first_bridge_snr_min: null
  image_first_bridge_snr_max: 5.0
```

For each sampled video timestep, `train.py` computes SNR. Timesteps inside the
configured range use the anchor bridge. Timesteps outside the range use
standard video DDPM noising:

```text
if SNR(t) <= image_first_bridge_snr_max:
  z_t    = q(z_t | [z*1, ..., z*1])
  target = epsilon_to_reconstruct(z*)
else:
  z_t    = q(z_t | z*)
  target = sampled epsilon
```

This avoids forcing the model to learn the large
`sqrt(SNR(t)) * ([z*1, ..., z*1] - z*)` correction at near-clean timesteps.

`image_first_snr_ea.yaml` additionally enables `latent_calibrator`, a
zero-init temporal-conv residual mapper that is applied to bridged
anchor-expanded noisy latents before the UNet/video adapters. It uses the same
bridge noise to form a weak low-frequency target
`q(z_t | z*)`, so the auxiliary loss aligns deterministic anchor-to-video
latent shift rather than independent noise.

`rollout` is the rollout-source objective used by
`configs/train/image_first_rollout.yaml`. `rollout_snr` is the same rollout
source with SNR-gated fallback and is used by
`configs/train/image_first_rollout_snr.yaml`:

```yaml
training:
  image_first_bridge_mode: "rollout"      # or "rollout_snr"
  image_first_bridge_snr_max: 5.0         # used by rollout_snr
  image_first_rollout_steps: 4
  image_first_rollout_switch_noise_scale: 0.1
```

For each video, the first-frame anchor is noised at
`target_t + image_first_rollout_steps` and then denoised down to `target_t`
with the base SDXL UNet while UNet video Resnet and attention adapters are
inactive. The resulting image latent is duplicated across frames. Optional
frame-wise switch noise is added after duplication to mirror image-first
inference:

```text
z_img,target = base_sdxl_rollout(q(z*1, target_t + K), target_t)
z_t          = [z_img,target, ..., z_img,target] + switch_noise
target       = epsilon_to_reconstruct(z*)
```

The rollout switch noise level is one scalar per video timestep, repeated over
frames as `[B*F, 1, 1, 1]` and broadcast over latent channels/spatial
dimensions. It must not be reshaped to the full `[B*F, C, H, W]` latent shape.

This makes training inputs closer to the intermediate image latents that
`generate_image_first_video_frames(...)` produces at the image-to-video switch.
For `rollout_snr`, timesteps outside the configured SNR range do not use the
rollout source in the loss path; they use standard video DDPM noising plus the
sampled epsilon target.

## Validation Logic

When `latent_init_mode: first_frame_repeat`, `train.py` switches validation from
standard `generate_video_frames(...)` to
`generate_image_first_video_frames(...)`.

Validation uses:

```yaml
validation:
  guidance_scale: 8.0
  t1_ratios: [0.0, 0.25, 0.5, 0.75]
  switch_noise_scale: 0.1
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
- small frame-wise switch noise is optionally added:

```text
z_switch^f = z_img + switch_noise_scale * sigma_switch * eta^f
```

- `sigma_switch` is read from the inference scheduler if available,
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

`--switch_noise_scale` overrides `validation.switch_noise_scale`; the default
fallback is `0.1`. Set it to `0.0` to recover exact latent duplication at the
switch point.

`--step last` maps to `checkpoint-last`. `--checkpoint` and `--output_dir` are
available for copied checkpoint folders on another server. If `--config` is not
given, the script first reads `checkpoint/config.yaml`, which keeps copied
checkpoints portable.

## Configs And Scripts

Mixed-noise sinusoidal frame embedding:

```bash
bash scripts/train_image_first.sh
```

uses:

```text
configs/train/image_first_mixed.yaml
```

This run is named `image-first-mixed` and writes to:

```text
outputs/image-first-mixed
```

The non-mixed sinusoidal config remains available at:

```text
configs/train/image_first_sinusoidal.yaml
```

Run it directly with:

```bash
bash scripts/train_image_first_sinusoidal.sh
```

Learnable frame-token ablation:

```bash
bash scripts/train_image_first_learnable.sh
```

uses:

```text
configs/train/image_first_learnable_frame_tokens.yaml
```

SNR-gated hybrid bridge:

```bash
bash scripts/train_image_first_snr.sh
```

uses:

```text
configs/train/image_first_snr.yaml
```

SNR-gated hybrid bridge with latent calibrator:

```bash
bash scripts/train_image_first_snr_ea.sh
```

uses:

```text
configs/train/image_first_snr_ea.yaml
```

Rollout-source bridge:

```bash
bash scripts/train_image_first_rollout.sh
```

uses:

```text
configs/train/image_first_rollout.yaml
```

Rollout-source bridge with SNR-gated fallback:

```bash
bash scripts/train_image_first_rollout_snr.sh
```

uses:

```text
configs/train/image_first_rollout_snr.yaml
```

All image-first configs keep the base SDXL image model frozen by default and
train only the added temporal/adaptor parameters configured in
`video_adapters`.
