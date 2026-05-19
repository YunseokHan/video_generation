# Backbone Compatibility

This file records the latest local compatibility check against the actual SDXL
backbone used for this project.

## Checked Environment

```text
python:    /NHNHOME/WORKSPACE/26moe001_D/miniconda3/envs/video/bin/python
torch:     2.12.0+cu130
diffusers: 0.38.0
model:     stabilityai/stable-diffusion-xl-base-1.0
```

## Loading Note

The user-provided load snippet used:

```python
DiffusionPipeline.from_pretrained(..., dtype=torch.bfloat16, device_map="cuda")
```

In diffusers `0.38.0`, `dtype=` is ignored by `StableDiffusionXLPipeline`.
The repository loaders use `torch_dtype=...`, which correctly loads bf16:

```python
DiffusionPipeline.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="cuda")
```

This is the main mismatch found during the backbone check.

## Verified Architecture

```text
pipeline:                       StableDiffusionXLPipeline
vae:                            AutoencoderKL
unet:                           UNet2DConditionModel
tokenizer max length:           77 / 77
text_encoder hidden/proj:       768 / 768
text_encoder_2 hidden/proj:     1280 / 1280
vae scaling_factor:             0.13025
vae latent_channels:            4
vae block_out_channels:         [128, 256, 512, 512]
unet sample_size:               128
unet in/out channels:           4 / 4
unet block_out_channels:        [320, 640, 1280]
unet cross_attention_dim:       2048
unet addition_time_embed_dim:   256
unet add_embedding input dim:   2816
transformer_layers_per_block:   [1, 2, 10]
```

## Verified Module Counts

```text
UNet ResnetBlock2D:             17
UNet BasicTransformerBlock:     70
UNet Transformer2DModel:        11
UNet Attention:                 140
VAE decoder ResnetBlock2D:      14
VAE decoder UpDecoderBlock2D:   4
```

Adapter injection with `configs/train/default.yaml` replaces:

```text
UNet VideoResnetBlock2D:        17
UNet VideoBasicTransformerBlock:70
VAE decoder VideoResnetBlock2D: 14
```

No architecture-count mismatch was found for the implemented adapters.

## Known Non-Covered Modules

- The VAE decoder has one internal attention module in the mid block. The
  current VAE decoder adapter only wraps `ResnetBlock2D`, matching the current
  surgery plan.
- The SDXL pipeline includes `feature_extractor` and `image_encoder`
  components, but text-to-image training and inference here do not use them.
