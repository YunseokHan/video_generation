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

`configs/train/default.yaml` enables:

- UNet convolutional/Resnet video adapters
- UNet attention video adapters
- VAE decoder Resnet video adapters, active but not trained by default
- sinusoidal frame-wise token embedding with `token_embedding_mode: add_to_text`
- OpenVid video loading from `/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted`
- adapter-only training with the base SDXL UNet, VAE, and text encoders frozen

`configs/train/learnable_frame_tokens.yaml` keeps the same adapter setup but
changes frame-wise token embedding to learned tokens with
`token_embedding_mode: concat_tokens`.

`configs/accelerate/default.yaml` is the default multi-GPU launcher config and
uses 4 GPU processes (`gpu_ids: "0,1,2,3"`).

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

When `logging.report_to: "wandb"` is enabled, training metrics such as loss,
learning rate, epoch progress, seen frames, parameter counts, checkpoint
events, and GPU memory are logged to W&B. The training loop can also
periodically run inference on one caption from the current training batch and
log the generated grid/video under `train_caption/*`; configure this with
`logging.log_training_caption_inference` and
`logging.training_caption_inference_steps`.

## Inference

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
PROMPT="Astronaut walking through a jungle, cold color palette" \
NUM_FRAMES=16 \
OUTPUT_DIR=outputs/example_video \
bash scripts/generate_video_frames.sh
```

This writes `frame_000.png`, `frame_001.png`, and so on. Add `--save_mp4` to
request `video.mp4` when `imageio` is available.

## Tests

```bash
cd /NHNHOME/WORKSPACE/26moe001_D/yunseok/video_generation
bash scripts/run_core_tests.sh -q
```

`test.py` can run directly even when `pytest` is not installed; if `pytest` is
available, it delegates to pytest.
