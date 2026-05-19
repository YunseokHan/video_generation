# SDXL Frame Generator

Experimental SDXL frame-level video generator baseline:

```text
p_image(x) -> p_frame(x | c, t_f)
```

The implementation keeps SDXL text conditioning intact and injects normalized frame position through an MLP added to SDXL pooled prompt embeddings before the UNet added-conditioning path.

## Train

Use the local environment that has diffusers installed. For one process:

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
scripts/train_single_gpu.sh
```

For multi-GPU, launch with Accelerate so each process is assigned one GPU:

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
NUM_PROCESSES=4 scripts/train_multi_gpu.sh
```

The training script uses `Accelerator.prepare(...)` for the trainable modules, optimizer, and dataloader. Startup logs include the distributed type, process count, process index, local process index, and device.

The shell wrappers read `.env` and accept environment-variable overrides:

```bash
CONFIG=configs/default.yaml ENV_FILE=.env WANDB_MODE=online scripts/train_single_gpu.sh
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 MIXED_PRECISION=bf16 scripts/train_multi_gpu.sh
```

The placeholder dataloader returns:

```text
frames: [B, F, 3, H, W]
captions: List[str]
frame_positions: [B, F]
```

Training flattens this to image-like samples, repeats each caption across that video's frames, embeds `t_f`, and trains the SDXL UNet plus temporal MLP by default.

## Environment And W&B

Experiment environment variables are managed through a local `.env` file, which is intentionally gitignored. Start from [.env.example](/NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation/.env.example):

```bash
cp .env.example .env
```

Fill in `HF_TOKEN` for gated Hugging Face access and `WANDB_API_KEY` for online W&B logging. The default template sets `WANDB_MODE=offline` so a dry run can log locally without a key; switch it to `online` after adding the key.

W&B logging is controlled by `logging:` in [configs/default.yaml](/NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation/configs/default.yaml). Logged metrics include loss, learning rate, global step, estimated frames seen, GPU max memory, parameter counts, checkpoint events, validation frame counts, and validation grid/video media.

## Inference

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
PROMPT="Astronaut walking through a jungle, cold color palette" \
NUM_FRAMES=16 \
OUTPUT_DIR=outputs/example_video \
scripts/generate_video_frames.sh
```

This writes `frame_000.png`, `frame_001.png`, and so on. Add `--save_mp4` to request `video.mp4` when `imageio` is available.

Useful scripts:

```bash
scripts/train_single_gpu.sh
scripts/train_multi_gpu.sh
scripts/generate_video_frames.sh
scripts/run_core_tests.sh
```
