# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintenance Rule

Every implementation change **must** update the matching section under `information/` in the same change. Treat `information/` as the living design spec — read it before editing code. Key files:

- `information/0_overview.md` — status, module map, implemented vs not-yet-implemented
- `information/5_training_and_inference.md` — entrypoints and checkpoint flow
- `information/10_image_first_training.md` — first-frame-repeat objective and inference
- `information/11_latent_calibrator.md` — latent calibrator architecture

When adding a new module or ablation, update the relevant file or add a new numbered markdown file and link it in `information/0_overview.md`.

## Instruction

- If you start new session to work on a new git repository, always read README.md first.
- If my instruction is unclear or ambigue, you should first try to interpret my intent and request me to clarify the instruction.
- If my instruction is logically flawed, you should rebut my instruction and provide the alternative approach. Do not follow my instruction passively even if it is not a valid approach.

## Environment

Python binary: `/home/work/data/miniconda3/envs/video/bin/python` (the `video` conda env).

Verified stack: `torch 2.12.0+cu130`, `diffusers 0.38.0`.

> diffusers `0.38.0` ignores `dtype=` in `from_pretrained`. Use `torch_dtype=torch.bfloat16` instead.

Training requires a `.env` file with:
```
HF_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_MODE=online
```

## Commands

**Run tests** (does not require pytest):
```bash
bash scripts/run_core_tests.sh -q
# or directly:
/home/work/data/miniconda3/envs/video/bin/python test.py -q
```

**Train** (canonical launcher — selects both train and accelerate configs):
```bash
bash scripts/train.sh
# Override configs:
TRAIN_CONFIG=configs/train/learnable_frame_tokens.yaml bash scripts/train.sh
# Image-first variants:
bash scripts/train_image_first.sh
bash scripts/train_image_first_snr.sh
bash scripts/train_image_first_snr_ea.sh  # + latent calibrator
bash scripts/train_image_first_rollout.sh
bash scripts/train_image_first_rollout_snr.sh
```

**Inference** (standard):
```bash
PROMPT="..." NUM_FRAMES=16 NAME=<run-name> STEP=<n> bash scripts/generate_video_frames.sh
# or directly:
python infer.py --name <run-name> --step <n> --prompt "..." --num_frames 16
```
Writes to `outputs/infer/<name>-<step>/cfg_1/` and `cfg_8/`.

**Image-first inference**:
```bash
python infer_image_first.py --name <run-name> --step <n> --prompt "..." \
  --num_frames 8 --t1 0.25 --guidance_scale 8 --switch_noise_scale 0.1
```
Writes to `outputs/infer_image_first/<name>-<step>/t1_*/cfg_*/`.

## Architecture

The repo keeps the pretrained SDXL backbone frozen and wraps it with trainable video adapters. Training flattens `[B, F, H, W]` video to `[B*F, H, W]` before feeding the UNet; adapters re-fold frames for temporal operations.

**Adapter injection sites** (exact SDXL counts):
- 17 UNet `ResnetBlock2D` → `VideoResnetBlock2D` (`framegen/video_resnet.py`)
- 70 UNet `BasicTransformerBlock` → `VideoBasicTransformerBlock` (`framegen/video_attention.py`)
- 14 VAE decoder `ResnetBlock2D` → same wrapper (active but not trained by default)

**`framegen/` module map**:
| File | Role |
|------|------|
| `sdxl.py` | SDXL dual-encoder prompt encoding, time IDs, pooled temporal MLP |
| `temporal.py` | Frame position encoders; `token_embedding_mode` ablations (`add_to_text`, `concat_tokens`, `temporal_cross_attention_only`, `none`) |
| `video_resnet.py` | `VideoResnetBlock2D`: identity-init temporal conv3D + frame conditioning |
| `video_attention.py` | `VideoBasicTransformerBlock`: temporal self-attn, temporal cross-attn, temporal FFN |
| `latent_calibrator.py` | Zero-init temporal-conv calibrator that maps anchor-expanded latents toward video latent distribution |
| `data.py` | `PlaceholderVideoDataset` and `OpenVidVideoDataset` (CSV/mp4, SDXL crop metadata) |
| `generation.py` | CFG video generation loop used by validation and `infer.py` |
| `image_first_generation.py` | Two-stage denoising split for image-first inference (`infer_image_first.py`) |
| `checkpointing.py` | `save_checkpoint` / `load_checkpoint`; filenames: `resnet_video_adapter.pt`, `attention_video_adapter.pt`, `frame_position_encoder.pt`, `latent_calibrator.pt` |
| `config.py` | `load_config` / `save_config` (YAML), `get_torch_dtype` |
| `env.py` | HF token, cache dir, `.env` loading |
| `utils.py` | `save_mp4`, `save_image_grid` |

**Image-first training** (`latent_init_mode: "first_frame_repeat"`): the noisy input is constructed from the repeated first-frame latent; the denoising target is the full ground-truth video latent. Bridge modes: `mixed`, `snr` (SNR-gated fallback to standard noising), `rollout` (denoised anchor as source), `rollout_snr`.

**Checkpoints** are saved under `outputs/<name>/checkpoint-<step>/` and mirrored to `checkpoint-last`. `infer.py` auto-discovers `config.yaml` from the checkpoint directory, making checkpoint folders portable.

**Dataset**: OpenVid-1M extracted at `/NHNHOME/WORKSPACE/26moe001_D/dataset/OpenVid-1M/OpenVid_extracted`. Default multi-GPU config uses 4 GPUs (`configs/accelerate/default.yaml`).
