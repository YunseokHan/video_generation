# Latent Calibrator

The latent calibrator is an optional lightweight mapper between
first-frame-repeat image-first noising and the UNet/video adapters. It is meant
to align the input distribution, not to replace the video denoiser.

## Motivation

Image-first training with first-frame repeat builds:

```text
z_t,anchor = sqrt(alpha_bar_t) * E(a) + sqrt(1 - alpha_bar_t) * epsilon
```

where `E(a)` is the repeated first-frame latent. The target clean latent is the
full video `z*`, so the epsilon target contains:

```text
epsilon + sqrt(SNR(t)) * (E(a) - z*)
```

`image_first_snr.yaml` reduces the worst near-clean correction by disabling the
anchor bridge above the configured SNR threshold. `image_first_snr_ea.yaml`
keeps that gate and adds a residual calibrator for the remaining bridged
timesteps:

```text
z_t,calib = z_t,anchor + gamma(t) * Delta_phi(z_t,anchor, E(a), t, f, c)
```

## Architecture

The implemented architecture is `latent_calibrator.architecture=temporal_conv`
in `framegen/latent_calibrator.py`.

Input per frame is:

```text
concat[z_t^f, E(a)^f, z_t^f - E(a)^f]
```

The mapper applies:

```text
1x1 projection
temporal depthwise Conv3d kernel=(3,1,1)
2D FiLM ResBlocks conditioned on timestep, frame position, and pooled prompt
temporal depthwise Conv3d kernel=(3,1,1)
zero-init 3x3 projection to 4 latent channels
```

The final projection is always zero-initialized. Therefore an enabled
calibrator starts as an exact identity:

```text
Delta_phi = 0
z_t,calib = z_t,anchor
```

This prevents the module from changing the existing image-first input
distribution before it has learned a useful residual.

## Residual Scaling

`configs/train/image_first_snr_ea.yaml` uses:

```yaml
latent_calibrator:
  residual:
    scale_mode: "clipped_snr"
    max_scale: 0.5
    norm_cap: 1.0
```

`clipped_snr` computes `sqrt(SNR(t))` and clips it to `max_scale`. The raw
delta is also passed through a `tanh` cap when `norm_cap > 0`.

## Training Path

In training, the calibrator is applied only when:

```text
latent_init_mode == first_frame_repeat
image_first_bridge_mode != rollout
bridge_mask == true
```

The normal denoising target remains the image-first epsilon target. The
calibrator adds optional auxiliary losses but does not change the UNet output
contract.

For the map-alignment target, training reuses the same bridge noise:

```text
z_t,video = sqrt(alpha_bar_t) * z* + sqrt(1 - alpha_bar_t) * epsilon_bridge
```

Then:

```text
L_map = ||Down(z_t,calib) - Down(sg(z_t,video))||^2
```

Using the same noise means the auxiliary loss sees only the deterministic
anchor-to-video mismatch:

```text
z_t,video - z_t,anchor = sqrt(alpha_bar_t) * (z* - E(a))
```

`image_first_snr_ea.yaml` sets `map_weight: 0.05`,
`map_lowfreq_only: true`, `map_downsample_factor: 4`, and
`norm_weight: 0.001`.

The norm auxiliary computes the residual RMS with a small epsilon before the
square root. This is required because the calibrator output is zero-initialized;
without the epsilon, `sqrt(0)` can produce NaN gradients even when the forward
norm loss is exactly zero.

## Inference Path

Image-first inference still runs:

```text
Stage 1: image denoising with video adapters inactive
Stage 2: repeat current image latent to F frames, add optional switch noise
Stage 3: video denoising with adapters active
```

When a checkpoint contains `latent_calibrator.pt`,
`infer_image_first.py` loads it and applies it once at the image-to-video
switch before Stage 3. Current inference supports only
`latent_calibrator.apply_mode: "switch_only"`.

## Config

The first enabled experiment is:

```text
configs/train/image_first_snr_ea.yaml
scripts/train_image_first_snr_ea.sh
```

All other train configs keep the same `latent_calibrator` schema with
`enabled: false` so config diffs remain value-only across ablations.
