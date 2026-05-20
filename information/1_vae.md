# VAE

## SDXL VAE Shape Contract

The SDXL VAE uses spatial factor `8`.

```text
image:  [B, 3, H, W]
latent: [B, 4, H/8, W/8]
```

For `1024 x 1024` images:

```text
image:  [B, 3, 1024, 1024]
latent: [B, 4, 128, 128]
```

Training uses the SDXL scaling factor through diffusers:

```text
latents = vae.encode(frames).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

The repository relies on the model config rather than hard-coding the SDXL
constant.

For SDXL stability, `train.py` reloads the VAE separately with
`AutoencoderKL.from_pretrained(...)` and keeps it in fp32 by default:

```yaml
model:
  pretrained_vae_model_name_or_path: null
  vae_dtype: "fp32"
```

When `pretrained_vae_model_name_or_path` is `null`, the VAE is loaded from the
base SDXL model's `vae` subfolder. Setting it to an external model path enables
the same fixed-VAE workflow supported by the official SDXL trainer.

## Implemented In This Repo

- The training loop uses the VAE encoder to convert flattened frames
  `[B*F, 3, H, W]` into diffusion latents `[B*F, 4, H/8, W/8]`.
- The default training objective does not train the VAE.
- Input frames are moved to `model.vae_dtype` for VAE encode; noisy latents are
  cast back to `model.dtype` before the UNet forward pass.
- VAE decoder Resnet surgery is available through:

```yaml
video_adapters:
  vae_decoder_resnet:
    enabled: true
    active: true
    train: false
    frame_embedding_dim: 2048
    use_temporal_conv1: true
    use_temporal_conv2: true
    use_frame_conditioning: false
```

When enabled, `inject_video_resnet_adapters(vae.decoder, ...)` wraps the 14
decoder `ResnetBlock2D` modules with the same temporal adapter used for UNet
Resnet blocks.

## Runtime Context

During generation, `framegen/generation.py` sets VAE decoder context before the
pipeline call:

```text
num_frames:      current generated frame batch size
frame_positions: [F]
frame_embeddings:[F, D] when a frame position encoder exists
```

This lets decoder temporal convolutions see the flattened frame batch as video:

```text
[B*F, C, H, W] -> [B, C, F, H, W] -> Conv3d(k=3,1,1) -> [B*F, C, H, W]
```

## Current Limitation

The VAE decoder adapter is enabled and active by default, but `train=false`. It
is wired for injection, checkpointing, loading, and generation-time context. It
is not meaningfully trained by the current denoising loss because the training
loop does not decode latents back to pixels. To train it, add a reconstruction
or perceptual/video decode loss that calls `vae.decode(...)` during training.
