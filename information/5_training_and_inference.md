# Training And Inference Wiring

## Training Flow

The training script is `train.py` at the repository root.

1. Load SDXL pipeline, tokenizer, text encoders, VAE, and UNet.
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

5. Encode frames with VAE encoder and prompts with SDXL two-encoder logic.
6. Apply pooled frame conditioning.
7. Apply token embedding mode if `frame_position_encoder.enabled=true`.
8. Set UNet video adapter context and run denoising loss.

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

Default configs keep validation generation and training-caption inference
disabled for throughput. They remain available through
`validation.enabled=true` and `logging.log_training_caption_inference=true`
when media logging is needed.

Default configs also set `training.train_unet=false`. The base SDXL UNet is
frozen; only the added UNet Resnet adapters, UNet attention adapters, temporal
pooled MLP, and enabled frame-token encoder parameters are optimized. The VAE
decoder adapter remains active for decode/inference compatibility but is not
trained by the current denoising-only objective.

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

## Checkpoint Files

```text
checkpoint-N/
  unet/
  temporal_mlp.pt
  frame_position_encoder.pt
  resnet_video_adapter.pt
  attention_video_adapter.pt
  vae_decoder_video_adapter.pt
  config.yaml
```

Only files for enabled/present modules are written.

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
system/gpu_max_memory_allocated_mb
params/trainable
params/frozen
checkpoint/saved_step
```

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
