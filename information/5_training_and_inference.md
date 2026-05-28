# Training And Inference Wiring

## Training Flow

The training script is `train.py` at the repository root.

1. Load SDXL pipeline, tokenizer, text encoders, UNet, and a separately loaded
   SDXL VAE.
2. Inject configured adapters:
   - `video_adapters.resnet` into UNet Resnet blocks
   - `video_adapters.attention` into UNet BasicTransformerBlock modules
   - `video_adapters.vae_decoder_resnet` into VAE decoder Resnet blocks
3. Build temporal conditioning modules:
   - `FramePositionMLP` for pooled `[B*F, 1280]`
   - optional frame position encoder for token/frame embeddings
4. Flatten video batch:

```text
frames:          [B, F, 3, H, W] -> [B*F, 3, H, W]
captions:        B captions -> B*F repeated captions
frame_positions: [B, F] -> [B*F]
```

5. Encode frames with the VAE encoder in `model.vae_dtype` and prompts with
   SDXL two-encoder logic.
6. Apply pooled frame conditioning.
7. Apply token embedding mode if `frame_position_encoder.enabled=true`.
8. Set UNet video adapter context and run denoising loss.

If `training.latent_init_mode: "first_frame_repeat"`, step 5 still VAE-encodes
all frames to ground-truth latents, but the forward noising source is changed to
the first frame latent repeated over `F`. The loss target remains the full
ground-truth video latent through an effective epsilon target. See
`information/10_image_first_training.md`.

When text encoders are frozen, `train.py` encodes each video caption once at
`[B, 77, 2048]` and repeats the resulting embeddings across frames on GPU.
This is equivalent to encoding the duplicated `[B*F]` caption list, but avoids
`F` redundant CLIP/OpenCLIP forward passes per video.

`configs/train/default.yaml` and
`configs/train/learnable_frame_tokens.yaml` now use `data.dataset_type:
openvid`, which reads the local OpenVid CSV/mp4 extraction. See
`information/7_openvid_dataloader.md` for dataset-specific filters and decode
behavior.

## Config Layout And Launchers

Config files are split by responsibility:

```text
configs/train/default.yaml
configs/train/learnable_frame_tokens.yaml
configs/accelerate/default.yaml
```

`configs/train/` contains model, data, optimizer, logging, and ablation
settings. `configs/accelerate/` contains Accelerate launch settings.

All YAML files under `configs/train/` intentionally share the same recursive
key schema. Ablation-specific fields are still present in configs that do not
use them; for example, `training.image_first_noise_mode` is ignored when
`training.latent_init_mode: "video_gt"`, and `validation.guidance_scales` is
ignored by image-first validation, which uses `validation.guidance_scale` and
`validation.t1_ratios`.

`configs/accelerate/default.yaml` is the default launcher config for
`scripts/train.sh` and uses 4 GPU processes:

```yaml
distributed_type: MULTI_GPU
gpu_ids: "0,1,2,3"
num_processes: 4
mixed_precision: bf16
```

`scripts/train.sh` owns config path selection:

```bash
TRAIN_CONFIG=configs/train/default.yaml \
ACCELERATE_CONFIG=configs/accelerate/default.yaml \
bash scripts/train.sh
```

`TRAIN_CONFIG` selects training settings. `ACCELERATE_CONFIG` selects launch
settings. `CONFIG` remains a legacy alias for `TRAIN_CONFIG`, but new scripts
should use `TRAIN_CONFIG`. `_common.sh` only manages shared paths, the Python
and Accelerate binaries, `.env` loading, and project-root `cd`.

Compatibility wrappers:

```bash
bash scripts/train_multi_gpu.sh
bash scripts/train_single_gpu.sh
```

`train_multi_gpu.sh` delegates to `train.sh` with the default Accelerate path.
`train_single_gpu.sh` delegates to `train.sh` with `TRAIN_LAUNCHER=python`.

The training dataloader uses pinned host memory and non-blocking host-to-device
copies when `data.pin_memory=true`. With `num_workers>0`, it also enables
persistent workers and prefetching from config.
Train configs keep decoded frame batches in `uint8` while they pass through
DataLoader multiprocessing queues, then `train.py` converts them to the VAE
dtype and normalizes to `[-1, 1]` on the training device. This keeps
multiprocessing enabled while reducing CPU shared-memory payload.

Large video batches are transferred from worker processes through PyTorch CPU
shared memory. If a worker cannot allocate a shared-memory object, one rank can
stop receiving batches while other ranks continue to NCCL collectives; the
visible tail error is then often an allreduce timeout. Reduce
`data.num_workers` and `data.prefetch_factor`, disable persistent workers, or
set `data.torch_multiprocessing_sharing_strategy: "file_system"` for more
conservative runs. The train configs now set the sharing strategy to
`"file_system"` and use `data.dataloader_timeout: 120` so stalled workers fail
with a DataLoader error instead of waiting for the NCCL watchdog.

The VAE is loaded through `AutoencoderKL.from_pretrained(...)` instead of
reusing the bf16 pipeline copy. By default `model.vae_dtype: "fp32"`, matching
the official SDXL trainer's stability choice. Set
`model.pretrained_vae_model_name_or_path` to use an external fixed VAE such as
`madebyollin/sdxl-vae-fp16-fix`.

The following official SDXL trainer options are config-backed in `train.py`:

```yaml
model:
  pretrained_vae_model_name_or_path: null
  revision: null
  variant: null
  vae_dtype: "fp32"

data:
  center_crop: false
  random_flip: false
  image_interpolation_mode: "lanczos"

training:
  scale_lr: false
  lr_scheduler: "constant"
  lr_warmup_steps: 500
  num_train_epochs: 100
  gradient_checkpointing: false
  allow_tf32: false
  enable_xformers_memory_efficient_attention: false
  use_8bit_adam: false
  max_grad_norm: 1.0
  prediction_type: null
  noise_offset: 0.0
  snr_gamma: null
  timestep_bias_strategy: "none"
  timestep_bias_multiplier: 1.0
  timestep_bias_begin: 0
  timestep_bias_end: 1000
  timestep_bias_portion: 0.25
  proportion_empty_prompts: 0.0
  checkpoints_total_limit: null
  resume_from_checkpoint: null
  logging_dir: "logs"
```

The option names and defaults mirror the official SDXL script where they apply
cleanly. The experiment configs may still choose smaller video-safe values for
fields like `train_batch_size`, `resolution`, or `learning_rate`.

The main process shows one `tqdm` progress bar at the bottom of the terminal.
Its postfix reports recent loss, learning rate, epoch progress, seen frame
count, and peak GPU memory. Status messages such as checkpoint saves are written
through the progress bar so they do not break the live display.

The current default configs enable checkpoint validation every 1000 optimizer
steps and keep training-caption inference disabled for throughput.
Training-caption media remains available through
`logging.log_training_caption_inference=true`.

Default configs also set `training.train_unet=false`. The base SDXL UNet is
frozen; only the added UNet Resnet adapters, UNet attention adapters, temporal
pooled MLP, enabled frame-token encoder parameters, and enabled latent
calibrator parameters are optimized. The VAE decoder adapter remains active for
decode/inference compatibility but is not trained by the current denoising-only
objective.

Image-first configs are available at:

```text
configs/train/image_first_mixed.yaml
configs/train/image_first_sinusoidal.yaml
configs/train/image_first_learnable_frame_tokens.yaml
configs/train/image_first_snr.yaml
configs/train/image_first_snr_renoise.yaml
configs/train/image_first_snr_ea.yaml
configs/train/image_first_rollout.yaml
configs/train/image_first_rollout_snr.yaml
configs/train/image_first_smooth_snr.yaml
configs/train/image_first_smooth_snr_boundary.yaml
configs/train/image_first_smooth_snr_renoise_boundary.yaml
```

Launch them with:

```bash
bash scripts/train_image_first.sh
bash scripts/train_image_first_sinusoidal.sh
bash scripts/train_image_first_learnable.sh
bash scripts/train_image_first_snr.sh
bash scripts/train_image_first_snr_renoise.sh
bash scripts/train_image_first_snr_ea.sh
bash scripts/train_image_first_rollout.sh
bash scripts/train_image_first_rollout_snr.sh
bash scripts/train_image_first_smooth_snr.sh
bash scripts/train_image_first_smooth_snr_boundary.sh
bash scripts/train_image_first_smooth_snr_renoise_boundary.sh
```

These configs validate with CFG 8 only and run `t1` ratios
`0, 0.25, 0.5, 0.75`.

`scripts/train_image_first.sh` launches `configs/train/image_first_mixed.yaml`.
That config is named `image-first-mixed` and uses mixed image-first noise:
per video, 50% frame-shared noise and 50% frame-independent noise.

`scripts/train_image_first_sinusoidal.sh` launches
`configs/train/image_first_sinusoidal.yaml`. It keeps the original
frame-independent training noise while using the same image-first validation
path and default `validation.switch_noise_scale: 0.1`.

`scripts/train_image_first_snr.sh` launches
`configs/train/image_first_snr.yaml`. It sets
`training.image_first_bridge_mode: "snr"` and
`training.image_first_bridge_snr_max: 5.0`, so timesteps above that SNR use
standard video DDPM noising/epsilon targets instead of the anchor bridge.

`scripts/train_image_first_snr_renoise.sh` launches
`configs/train/image_first_snr_renoise.yaml`. It keeps the same hard SNR
training bridge as `image_first_snr.yaml`, but uses
`validation.image_first_switch_mode: "pred_x0_renoise"` so validation converts
the image switch latent to `pred_x0` and re-noises it at the scheduler switch
level.

`scripts/train_image_first_snr_ea.sh` launches
`configs/train/image_first_snr_ea.yaml`. It keeps the same SNR-gated bridge and
enables `latent_calibrator`, a zero-init temporal-conv residual mapper applied
to bridged anchor latents before the UNet/video adapters. It also uses a weak
low-frequency latent alignment auxiliary loss.

`scripts/train_image_first_rollout.sh` launches
`configs/train/image_first_rollout.yaml`. It sets
`training.image_first_bridge_mode: "rollout"`, creates a source image latent
by running a small base-SDXL denoising rollout with video adapters inactive,
then duplicates that source latent across frames.

`scripts/train_image_first_rollout_snr.sh` launches
`configs/train/image_first_rollout_snr.yaml`. It sets
`training.image_first_bridge_mode: "rollout_snr"` and
`training.image_first_bridge_snr_max: 5.0`, so bridged timesteps use the
rollout source while higher-SNR timesteps fall back to standard video DDPM
noising/epsilon targets.

`scripts/train_image_first_smooth_snr.sh` launches
`configs/train/image_first_smooth_snr.yaml`. It sets
`training.image_first_bridge_mode: "smooth_snr"` and uses a cosine gate from
SNR 1 to SNR 5, without boundary loss or pred-x0 re-noise validation.

`scripts/train_image_first_smooth_snr_boundary.sh` launches
`configs/train/image_first_smooth_snr_boundary.yaml`. It adds the weak boundary
re-noising loss at SNR 5 to the smooth-SNR bridge, while keeping the legacy
repeat-add-noise validation switch.

`scripts/train_image_first_smooth_snr_renoise_boundary.sh` launches
`configs/train/image_first_smooth_snr_renoise_boundary.yaml`. It sets
`training.image_first_bridge_mode: "smooth_snr"`, uses a cosine gate from
SNR 1 to SNR 5 to transition from anchor-source noising to standard video
noising, adds a weak boundary re-noising loss at SNR 5, and uses
`validation.image_first_switch_mode: "pred_x0_renoise"` for matched
image-to-video switch inputs.

`global_step` is an optimizer-step counter, not a dataloader microbatch counter.
With `train_batch_size: 1`, `gradient_accumulation_steps: 4`,
`num_frames_per_video: 8`, and 4 Accelerate processes, one logged optimization
step aggregates 16 videos and 128 frames. With the current
`configs/train/default.yaml` value `train_batch_size: 5`, that becomes 80
videos and 640 frames per optimizer step. `train/loss` is averaged over the
local accumulation window and then averaged across processes before logging.

Prompt tokenization still truncates captions to the SDXL tokenizer max length,
but repeated truncation warnings are disabled by default to keep large-dataset
training logs readable.

## Inference Flow

The inference script is `infer.py` at the repository root.

Checkpoint loading covers:

- trained UNet directory
- temporal pooled MLP
- optional frame position encoder
- optional UNet Resnet adapter state
- optional UNet attention adapter state
- optional VAE decoder Resnet adapter state

Generation uses `framegen/generation.py`:

1. Encode the prompt once for all frames.
2. Build frame positions `[F]`.
3. Apply pooled frame conditioning to positive and negative pooled embeddings.
4. Apply selected token embedding mode to positive and negative prompt tokens.
5. Generate in frame chunks if `batch_size` is set.
6. Set context on UNet adapters and VAE decoder adapters for each chunk.
7. Save `frame_000.png`, `grid.png`, and optionally `video.mp4`.

MP4 export first tries `imageio.v3`. If the environment has `imageio` but not
an MP4 writer backend such as `imageio_ffmpeg`, `framegen.utils.save_mp4`
falls back to the system `ffmpeg` binary with `libx264` and `yuv420p` output.
If both paths fail, it prints the backend error instead of silently swallowing
the cause.

For checkpoint-only validation on another server, `infer.py` can resolve the
checkpoint and output path from an experiment name and step:

```bash
python infer.py \
  --name sdxl-resnet-attention-sinusoidal \
  --step 1000 \
  --prompt "Astronaut walking through a jungle, cold color palette" \
  --num_frames 16
```

This loads:

```text
outputs/sdxl-resnet-attention-sinusoidal/checkpoint-1000
```

and writes:

```text
outputs/infer/sdxl-resnet-attention-sinusoidal-1000/
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

`--step last` maps to `checkpoint-last`. `--checkpoint` and `--output_dir`
remain available for arbitrary paths. When `--config` is omitted, inference
first reads `checkpoint/config.yaml`, falling back to
`configs/train/default.yaml` only if the checkpoint does not carry a config.
Use `--guidance_scale 8` for one CFG value, or `--guidance_scales 1 8` for
explicit multi-CFG output. The default multi-CFG inference path uses CFG 1 and
CFG 8. Each generated folder writes the actual prompt to `caption.txt`.

Image-first inference is intentionally separate from `infer.py`:

```bash
python infer_image_first.py \
  --name image-first-sinusoidal-res512-bs5 \
  --step 1000 \
  --prompt "A dog running through a grassy field, cinematic lighting" \
  --num_frames 8 \
  --t1 0.25 \
  --guidance_scale 8 \
  --switch_noise_scale 0.1
```

It performs base-image denoising with adapters off for
`round(t1 * num_inference_steps)` scheduler steps, repeats the current image
latent to all frames, optionally adds small frame-wise switch noise, restores
adapter active states, and completes denoising as a video. Outputs are saved
under:

```text
outputs/infer_image_first/{name}-{step}/t1_*/cfg_*/
```

`validation.switch_noise_scale` controls the training-time image-first
validation default. Image-first configs set it to `0.1`; omit or set `0.0` to
use exact latent duplication.

## Checkpoint Files

```text
checkpoint-N/
  unet/
  accelerator_state/
  temporal_mlp.pt
  frame_position_encoder.pt
  resnet_video_adapter.pt
  attention_video_adapter.pt
  vae_decoder_video_adapter.pt
  config.yaml
  trainer_state.json
```

Only files for enabled/present modules are written. `accelerator_state/` stores
optimizer, scheduler, model wrapper, and RNG state for exact resume. Use:

```yaml
training:
  resume_from_checkpoint: "latest"
```

or point it to `checkpoint-N` / `checkpoint-last`. `checkpoints_total_limit`
rotates numbered checkpoints while preserving the latest checkpoint-last copy.

## Tests

Core tests live in `test.py` at the repository root.

```bash
bash scripts/run_core_tests.sh -q
```

The script can run without `pytest`; if `pytest` is installed it delegates to
pytest, otherwise it runs the core test functions directly.

## W&B Logging

`.env` exposes explicit slots for online W&B and Hugging Face authentication:

```text
HF_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_MODE=online
```

When `logging.report_to: "wandb"` is enabled, `train.py` logs scalar training
metrics through Accelerate:

```text
train/loss
train/lr
train/epoch
train/global_step
train/samples_seen_frames
train/effective_batch_videos
train/effective_batch_frames
system/gpu_max_memory_allocated_mb
params/trainable
params/frozen
checkpoint/saved_step
```

Default validation config:

```yaml
training:
  validation_steps: 1000

logging:
  log_validation_media: true

validation:
  enabled: true
  num_frames: 8
  save_grid: true
  save_mp4: true
  guidance_scales: [1.0, 8.0]
```

Every 1000 optimizer steps, the main process generates one validation video per
configured CFG scale and logs separate `validation/cfg_1/video` and
`validation/cfg_8/video` artifacts to W&B when `logging.report_to` includes
`wandb`. The same prompt is saved as `caption.txt` in each CFG folder.

Training-caption inference is controlled by:

```yaml
logging:
  log_training_caption_inference: false
  training_caption_inference_steps: 100
  training_caption_output_dir: "training_caption_samples"
  training_caption_num_frames: 8
  training_caption_num_inference_steps: 20
  training_caption_guidance_scale: 7.5
  training_caption_batch_size: null
  training_caption_seed: null
  training_caption_resolution: null
  training_caption_save_grid: true
  training_caption_save_mp4: false
  training_caption_fps: 8
```

At each configured interval, the main process takes `batch["captions"][0]`,
temporarily switches the unwrapped model modules to eval mode, runs actual
pipeline inference, logs `train_caption/grid` and optionally
`train_caption/video` to W&B, then restores the original training/eval states.

## Ablation Examples

Add sinusoidal frame embeddings directly to text tokens:

```yaml
video_adapters:
  frame_position_encoder:
    enabled: true
    type: "sinusoidal"
    token_embedding_mode: "add_to_text"
    embedding_dim: 2048
    num_tokens: 1
```

This is the default config:

```bash
TRAIN_CONFIG=configs/train/default.yaml bash scripts/train.sh
```

Concatenate learnable frame tokens:

```yaml
video_adapters:
  frame_position_encoder:
    enabled: true
    type: "learned_tokens"
    token_embedding_mode: "concat_tokens"
    embedding_dim: 2048
    hidden_dim: 1024
    num_layers: 2
    num_tokens: 4
```

This config is available as:

```bash
TRAIN_CONFIG=configs/train/learnable_frame_tokens.yaml bash scripts/train.sh
```

Keep the old behavior where frame tokens only feed temporal cross-attention:

```yaml
video_adapters:
  frame_position_encoder:
    enabled: true
    token_embedding_mode: "temporal_cross_attention_only"
```
