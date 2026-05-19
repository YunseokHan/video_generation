# UNet Resnet Blocks

## Base SDXL Flow

The UNet receives flattened frame latents:

```text
z_t:      [B*F, 4, H/8, W/8]
timestep: [B*F]
context:  [B*F, 77, 2048]
pooled:   [B*F, 1280]
```

The output keeps the same latent shape:

```text
noise_pred: [B*F, 4, H/8, W/8]
```

## Implemented Adapter

`framegen/video_resnet.py` replaces diffusers `ResnetBlock2D` modules with
`VideoResnetBlock2D` when `video_adapters.resnet.enabled=true`.

Each wrapped block keeps the original image path:

```text
GroupNorm -> SiLU -> Conv2d
time embedding add
GroupNorm -> SiLU -> Dropout -> Conv2d
residual add
```

Then the adapter adds temporal-only 3D convolutions after the original spatial
convolutions:

```text
[B*F, C, H, W]
-> reshape [B, C, F, H, W]
-> Conv3d(kernel=(3,1,1), padding=(1,0,0))
-> reshape [B*F, C, H, W]
```

The temporal convolutions are initialized as identity, so enabling the adapter
does not immediately change the base Resnet output.

## Frame Conditioning Branch

If `use_frame_conditioning=true`, the block projects frame embeddings and adds
them to the timestep embedding branch:

```text
frame_embedding: [B*F, D_frame]
Linear(D_frame -> C_out)
add to projected timestep embedding
```

The frame projection is zero-initialized to preserve the pretrained image prior
at adapter initialization.

When `use_frame_conditioning=false`, the implementation does not allocate the
frame projection module and skips the zero-add branch entirely. This keeps the
VAE decoder adapter lighter when only temporal convolutions are requested.

## Config

```yaml
video_adapters:
  resnet:
    enabled: true
    active: true
    train: true
    frame_embedding_dim: 2048
    use_temporal_conv1: true
    use_temporal_conv2: true
    use_frame_conditioning: true
```

## Implemented Coverage

- UNet down, mid, and up Resnet blocks are covered because the injector walks
  the full UNet module tree.
- The verified SDXL base UNet contains 17 `ResnetBlock2D` modules, and all 17
  are replaced when the default config is loaded.
- VAE decoder Resnet blocks are covered separately by
  `video_adapters.vae_decoder_resnet`.
