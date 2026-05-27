# Loss And Timestep Schedule

This file documents the current training loss and diffusion timestep behavior
implemented in `train.py` and `framegen/generation.py`.

## Training Scheduler

Training loads the SDXL scheduler subfolder as a `DDPMScheduler`:

```python
noise_scheduler = DDPMScheduler.from_pretrained(
    model_config["pretrained_model_name_or_path"],
    subfolder="scheduler",
)
```

The locally verified scheduler config for
`stabilityai/stable-diffusion-xl-base-1.0` is:

```text
class:                  DDPMScheduler
num_train_timesteps:    1000
beta_start:             0.00085
beta_end:               0.012
beta_schedule:          scaled_linear
prediction_type:        epsilon
variance_type:          fixed_small
clip_sample:            False
thresholding:           False
timestep_spacing:       leading
steps_offset:           1
rescale_betas_zero_snr: False
```

## Timestep Sampling

Each training batch starts as video tensors:

```text
frames: [B, F, 3, H, W]
```

The batch is flattened before VAE encoding:

```text
frames_flat: [B*F, 3, H, W]
latents:     [B*F, 4, H/8, W/8]
```

Timesteps are sampled once per video and then repeated over the frame dimension:

```python
video_timesteps = torch.randint(
    0,
    noise_scheduler.config.num_train_timesteps,
    (batch_size,),
    device=latents.device,
).long()
timesteps = video_timesteps.repeat_interleave(num_frames)
```

With the verified SDXL config, this means uniform integer sampling over:

```text
t ~ Uniform({0, 1, ..., 999})
```

Important implication: all `F` frames from the same video receive the same
diffusion timestep in a training step. This matches inference more closely,
where all frame latents move through the same reverse denoising schedule
together. Different videos in the same batch still sample independent timesteps.

Optional timestep bias follows the official SDXL trainer arguments:

```yaml
training:
  timestep_bias_strategy: "none"   # none, earlier, later, range
  timestep_bias_multiplier: 1.0
  timestep_bias_begin: 0
  timestep_bias_end: 1000
  timestep_bias_portion: 0.25
```

When enabled, the bias distribution is sampled at video level and then repeated
over frames. It never samples independent timesteps for frames within the same
video.

## Forward Noising

The VAE latent uses the SDXL VAE scaling factor before noising:

```python
latents = vae.encode(frames_flat).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

Noise is sampled with the same shape as the latent:

```python
noise = torch.randn_like(latents)
```

`training.noise_offset` implements the official offset-noise option. The default
is `0.0`, which keeps the standard Gaussian noise:

```python
noise += noise_offset * torch.randn([B*F, C, 1, 1])
```

Noisy latents are created by the scheduler:

```python
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(torch_dtype)
```

Conceptually, this applies the DDPM forward process:

```text
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where `epsilon` is the sampled Gaussian noise.

For `training.latent_init_mode: "first_frame_repeat"`, the VAE still encodes
the full video to `clean_latents = z*`, but `add_noise(...)` receives the first
frame latent repeated over all frames:

```text
noising_latents = [z*1, z*1, ..., z*1]
z_t = q(z_t | noising_latents)
```

Noise is still independent per flattened frame, while timesteps are shared per
video as described above.

`training.image_first_noise_mode` controls this noise only for
`first_frame_repeat`:

```yaml
training:
  image_first_noise_mode: "independent"  # independent | shared | mixed
  image_first_shared_noise_prob: 0.5
```

`shared` uses one Gaussian noise tensor per video and repeats it over all
frames. `mixed` chooses shared noise for each video with
`image_first_shared_noise_prob` and otherwise uses frame-independent noise.
`configs/train/image_first_mixed.yaml` sets `mixed` with probability `0.5`.

`training.image_first_bridge_mode` controls which source distribution is used
for `first_frame_repeat`:

```yaml
training:
  image_first_bridge_mode: "always"  # always | snr | rollout | rollout_snr
  image_first_bridge_snr_min: null
  image_first_bridge_snr_max: null
  image_first_rollout_steps: 0
  image_first_rollout_noise_offset: 0.0
  image_first_rollout_switch_noise_scale: 0.0
```

`always` is the original behavior above. `snr` computes SNR from the sampled
timestep and applies the anchor bridge only inside the configured SNR range.
Frames outside that range use standard video noising from `clean_latents`.
`configs/train/image_first_snr.yaml` sets `image_first_bridge_snr_max: 5.0`,
so high-SNR/near-clean timesteps avoid the large anchor-to-video correction.
`configs/train/image_first_snr_ea.yaml` keeps this SNR gate and enables
`latent_calibrator`, which is applied only to the bridged frames before the
UNet/video adapters.

`rollout` does not call `add_noise(...)` on the duplicated frame sequence.
Instead, the first-frame anchor is noised at a slightly earlier timestep,
denoised for `image_first_rollout_steps` base-SDXL steps with video adapters
inactive, duplicated across frames, and optionally perturbed with frame-wise
`image_first_rollout_switch_noise_scale` noise.

`rollout_snr` uses the same rollout source for frames inside the configured SNR
range and uses standard video noising/epsilon targets for frames outside that
range. `configs/train/image_first_rollout_snr.yaml` sets
`image_first_bridge_snr_max: 5.0`.

## Prediction Target

The UNet receives:

```text
noisy_latents:          [B*F, 4, H/8, W/8]
timesteps:              [B*F]
prompt_embeds:          [B*F, 77 or 77+N_F, 2048]
modified pooled embeds: [B*F, 1280]
SDXL time_ids:          [B*F, 6]
```

The SDXL scheduler has `prediction_type: epsilon` by default, so the target is
the sampled noise:

```python
target = noise
```

For `training.latent_init_mode: "first_frame_repeat"`, the model is trained to
denoise the anchor-corrupted latent `z_t` to the full clean video latent `z*`.
The epsilon target is therefore:

```text
target_epsilon =
    (z_t - sqrt(alpha_bar_t) * z*) / sqrt(1 - alpha_bar_t)
```

This keeps the UNet interface in epsilon-prediction form while changing the
implied clean-latent target from the repeated first frame to the full video.
This mode currently requires `prediction_type: epsilon`; `train.py` raises for
other prediction types.

When `image_first_bridge_mode: "snr"` or `"rollout_snr"`, the formula above is
used only for frames whose sampled timestep falls inside the configured SNR
bridge range. Other frames use the standard DDPM epsilon target. When
`image_first_bridge_mode: "rollout"`, the same epsilon-to-clean conversion is
applied to rollout-produced source latents rather than analytic
anchor-forward-noised latents.

If `latent_calibrator.enabled=true`, `train.py` computes:

```text
z_t,calib = M_phi(z_t,anchor, E(a), t, f, c)
```

for frames selected by the image-first bridge mask. The calibrator is a
zero-init residual mapper, so the initial model input is identical to the
uncalibrated image-first input. The same bridge noise is used to construct the
auxiliary map target:

```text
z_t,video = sqrt(alpha_bar_t) * z* + sqrt(1 - alpha_bar_t) * epsilon_bridge
```

This keeps the auxiliary target focused on the deterministic anchor-to-video
latent mismatch rather than an unrelated stochastic noise mismatch.

The code also supports `prediction_type: v_prediction` if a future scheduler
config uses it:

```python
target = noise_scheduler.get_velocity(latents, noise, timesteps)
```

`training.prediction_type` can override the scheduler config with `epsilon`,
`v_prediction`, or `sample`. Unsupported prediction types raise an error.

## Loss Function

The default loss is plain mean-squared error over all latent elements:

```python
loss = torch_f.mse_loss(
    model_pred.float(),
    target.float(),
    reduction="mean",
)
```

For a 1024x1024, 8-frame, batch-per-GPU-1 reference setting:

```text
model_pred: [8, 4, 128, 128]
target:     [8, 4, 128, 128]
loss:       scalar mean over 524,288 latent values
```

The tensors are cast to fp32 for loss computation even when training runs in
bf16 mixed precision.

`training.snr_gamma` enables Min-SNR weighting from the official SDXL script.
When set, MSE is reduced per `[B*F]` latent first and weighted using
`diffusers.training_utils.compute_snr`. For `epsilon` prediction the weight is:

```text
min(snr, gamma) / snr
```

For `v_prediction` it is:

```text
min(snr, gamma) / (snr + 1)
```

The default is `null`, so the loss is unweighted MSE.

When `latent_calibrator.enabled=true`, the total objective adds optional
calibrator-only auxiliary terms:

```text
loss =
  diffusion_mse
  + map_weight  * ||Down(z_t,calib) - Down(sg(z_t,video))||^2
  + norm_weight * max(0, rms(delta_phi) - norm_cap)^2
```

`configs/train/image_first_snr_ea.yaml` sets `map_weight: 0.05`,
`map_lowfreq_only: true`, `map_downsample_factor: 4`, and
`norm_weight: 0.001`.

## Optimization Step Accounting

`global_step` is incremented only when `accelerator.sync_gradients` is true and
`optimizer.step()` has run. It is therefore an optimizer-step counter.

For a 4-GPU config with one video per GPU microbatch:

```text
train_batch_size:              1 video per GPU microbatch
num_frames_per_video:          8
gradient_accumulation_steps:   4
num_processes:                 4

videos per optimizer step:     1 * 4 * 4 = 16
frames per optimizer step:     16 * 8 = 128
```

The logged `train/loss` is averaged over the local accumulation window and then
averaged across processes. This makes the terminal/W&B loss correspond to the
same optimization step represented by `global_step`.

## Optimizer And LR Schedule

The optimizer defaults to AdamW, with optional bitsandbytes 8-bit AdamW:

```yaml
optimizer:
  type: "adamw"      # or "adamw8bit"

training:
  use_8bit_adam: false
```

Learning-rate scheduling uses `diffusers.optimization.get_scheduler`:

```yaml
training:
  lr_scheduler: "constant"
  lr_warmup_steps: 500
  scale_lr: false
  max_grad_norm: 1.0
```

The scheduler steps once per optimizer step. `max_grad_norm` clips trainable
adapter parameters before each optimizer step when it is greater than zero.

## What Is Not Implemented

The current objective is intentionally minimal:

- no independent per-frame timestep ablation,
- no temporal consistency loss,
- no optical-flow or feature matching loss,
- no VAE decode/reconstruction loss,
- no explicit frame-smoothness regularizer.

Gradients come only from denoising MSE. With default adapter-only training, the
base SDXL UNet participates in the forward/backward graph but its parameters are
frozen; only the added adapter modules and temporal conditioning modules are
optimized.

## Inference Scheduler

Inference uses `StableDiffusionXLPipeline` through `generate_video_frames`.
The locally verified default pipeline scheduler is:

```text
class:                EulerDiscreteScheduler
num_train_timesteps:  1000
prediction_type:      epsilon
timestep_spacing:     leading
steps_offset:         1
use_karras_sigmas:    False
```

`num_inference_steps` is passed to the pipeline at generation time. The default
validation/inference config uses 30 denoising steps unless overridden.

Training therefore uses a DDPM forward-noising scheduler for supervised noise
prediction, while inference uses the SDXL pipeline's Euler reverse-denoising
scheduler.
