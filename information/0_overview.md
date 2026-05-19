# SDXL Video Surgery Overview

This repository starts from SDXL image generation and adds frame-aware video
adapters without replacing the pretrained SDXL backbone.

## Implemented

- SDXL prompt encoding follows the two-tokenizer SDXL path:
  - token embeddings: `[B, 77, 2048]`
  - pooled embeddings: `[B, 1280]`
- Training flattens videos from `[B, F, 3, H, W]` to `[B*F, 3, H, W]`.
- OpenVid CSV/mp4 data loading is implemented for the local extracted dataset.
- UNet Resnet blocks can be wrapped with temporal convolution and frame
  conditioning adapters.
- UNet BasicTransformerBlock modules can be wrapped with:
  - temporal self-attention over `[B*N, F, C]`
  - temporal/frame cross-attention over `[B*N, F, C]`
  - temporal FFN over `[B*N, F, C]`
- Frame token conditioning has ablation modes controlled by config:
  - `add_to_text`
  - `concat_tokens`
  - `temporal_cross_attention_only`
  - `none`
- VAE decoder Resnet adapter injection is available through
  `video_adapters.vae_decoder_resnet`.

## Not Fully Implemented

- The training objective still optimizes denoising loss through the UNet path.
  VAE decoder adapters can be injected and used during inference decode, but
  the current training loop does not add a reconstruction/video decode loss for
  learning VAE decoder temporal behavior.
- VAE decoder adapters are enabled and active in the default config but
  `train=false` because the current loss does not call the decoder.

## Main Code Locations

- `framegen/sdxl.py`: SDXL prompt encoding, time IDs, pooled temporal MLP.
- `framegen/data.py`: placeholder and OpenVid video datasets.
- `framegen/temporal.py`: frame position encoders and token conditioning modes.
- `framegen/video_resnet.py`: ResnetBlock2D video adapter.
- `framegen/video_attention.py`: BasicTransformerBlock video adapter.
- `train.py`: training wiring.
- `framegen/generation.py`: validation and inference generation wiring.
- `infer.py`: checkpoint loading and CLI inference.
- `test.py`: core shape and adapter smoke tests.
- `configs/train/`: training, data, logging, and ablation configs.
- `configs/accelerate/`: Accelerate launcher configs; the default uses 4 GPUs.
- `scripts/train.sh`: canonical training launcher that selects both config paths.

## Verified SDXL Backbone

The `video` conda env was checked against
`stabilityai/stable-diffusion-xl-base-1.0` with diffusers `0.38.0`.

```text
UNet ResnetBlock2D:          17
UNet BasicTransformerBlock:  70
VAE decoder ResnetBlock2D:   14
tokenizer max length:        77 / 77
UNet cross_attention_dim:    2048
pooled text dim:             1280
VAE scaling_factor:          0.13025
```

Adapter injection currently replaces exactly those 17 UNet Resnet blocks, 70
UNet transformer blocks, and 14 VAE decoder Resnet blocks.
