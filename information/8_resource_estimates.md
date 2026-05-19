# Resource Estimates

This file records the current SDXL default-config parameter and VRAM estimates.
The counts were measured by loading
`stabilityai/stable-diffusion-xl-base-1.0`, injecting the configured adapters,
and counting module parameters.

## Default Config

```text
config:        configs/train/default.yaml
resolution:    1024
frames/video:  8
batch/GPU:     1 video = 8 flattened frames
precision:     bf16
GPUs checked:  4 x NVIDIA B200, 183359 MiB each
media eval:    disabled by default for throughput
```

## Parameter Counts

```text
Base UNet, no adapters:          2,567,463,684
UNet Resnet adapters:              112,380,800
UNet attention adapters:         1,026,377,670
Temporal pooled MLP:                 1,314,048
Default sinusoidal frame encoder:            0

Default trainable total:         1,140,072,518

Base VAE:                           83,653,863
VAE decoder Resnet adapters:        14,067,968
Text encoder 1, CLIP ViT-L:        123,060,480
Text encoder 2, OpenCLIP bigG:     694,659,840

Learnable frame-token encoder:       6,306,816
Learnable-token trainable total:  1,146,379,334
```

Default training sets `training.train_unet: false`, so the base SDXL UNet is
frozen. The optimized default parameters are the added UNet Resnet adapters,
UNet attention adapters, and temporal pooled MLP. Text encoders, VAE, VAE
decoder adapters, base UNet weights, and the sinusoidal frame encoder are
resident/frozen by default.

## Persistent VRAM Estimate

For the current code path, bf16 parameters use 2 bytes per parameter. PyTorch
AdamW creates bf16 `exp_avg` and `exp_avg_sq` states when the optimized
parameters are bf16.

```text
Trainable params:       2.12 GiB
Trainable grads:        2.12 GiB
AdamW m/v states:       4.25 GiB
Frozen params:          6.49 GiB
Subtotal:              14.98 GiB per GPU
```

DDP may add gradient bucket storage. Budget roughly another 0-3 GiB depending
on bucket/view settings and iteration state.

If using fp32 optimizer states or fp32 master weights later, the optimizer
budget changes substantially:

```text
fp32 AdamW m/v only:     8.49 GiB
fp32 m/v + master:      12.74 GiB
```

## Measured UNet Peak

A synthetic single-GPU measurement was run for the trainable UNet path only:

```text
latent:       [8, 4, 128, 128]
prompt:       [8, 77, 2048]
pooled:       [8, 1280]
frame mode:   sinusoidal add_to_text
operation:    UNet forward + MSE backward, base UNet frozen
trainable:    1,140,072,518 params
peak alloc:   70.51 GiB
post-bwd:      9.28 GiB
```

The measurement excludes VAE/text encoder resident weights and optimizer state.
In full training, add about 1.7 GiB for frozen VAE/text weights, about 4.25 GiB
for bf16 AdamW m/v states, and allow DDP bucket overhead. On the local 183 GiB
B200 GPUs, the default 4-GPU config should fit. On 80 GiB GPUs, this adapter-only
default is plausible but tight because the synthetic UNet path alone peaks at
70.51 GiB before full training overhead. Reducing frames/resolution or adding
activation checkpointing is recommended for 80 GiB-class devices.

## Main Activation Risks

The attention adapter is the dominant added parameter and activation source.
At 64x64 latent feature resolution with 8 frames:

```text
N:              4096 spatial tokens
context length: 77 + 8*77 = 693 tokens with include_prompt_tokens=true
attention map:  ~0.42 GiB bf16 for one 10-head temporal cross-attn score tensor
```

The implementation keeps temporal cross-attention context as `[B, S_ctx, D]`
and queries as `[B, N*F, C]`. This avoids materializing the repeated context:

```text
old repeated context: [B*N, 693, 2048] ~= 10.83 GiB bf16 at 64x64
new shared context:   [B,   693, 2048] ~=  0.003 GiB bf16
```

The score tensor size is unchanged, but K/V projection and context storage no
longer scale with spatial token count.

## Dataloader Memory

Each decoded sample at the default shape is:

```text
[8, 3, 1024, 1024] float32 ~= 96 MiB host memory
```

With `num_workers: 4`, `prefetch_factor: 2`, and one sample per worker queue,
each training process can hold roughly 768 MiB of prefetched decoded frame
tensors, plus ffmpeg buffers. Across 4 GPU processes, budget several GiB of
pinned host memory. This is intentional to reduce GPU starvation from video
decode latency.
