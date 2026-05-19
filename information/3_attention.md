# Attention Blocks

## Base SDXL Transformer2DModel

Diffusers flattens spatial feature maps before transformer blocks:

```text
h: [B*F, C, H, W]
x: [B*F, H*W, C]
N = H*W
```

The original `BasicTransformerBlock` path is preserved:

```text
x = x + SelfAttention(LN(x))
x = x + CrossAttention(LN(x), text_context)
x = x + FFN(LN(x))
```

## Implemented Video Adapter

`framegen/video_attention.py` replaces diffusers `BasicTransformerBlock` with
`VideoBasicTransformerBlock` when `video_adapters.attention.enabled=true`.
The verified SDXL base UNet contains 70 `BasicTransformerBlock` modules, and all
70 are replaced by the default config.

The wrapped block adds three video paths.

### Temporal Self-Attention

After spatial self-attention, tokens are regrouped by spatial location:

```text
[B*F, N, C] -> [B*N, F, C]
```

Self-attention then runs over the frame dimension only. The result is reshaped
back to `[B*F, N, C]` and added residually.

### Temporal / Frame Cross-Attention

After the original text cross-attention, the adapter runs another cross-attn
over temporal tokens:

```text
query:   [B*N, F, C]
context: static prompt tokens and/or frame token bank
output:  [B*N, F, C] -> [B*F, N, C]
```

Context sources:

- static prompt tokens from the original SDXL prompt context when
  `include_prompt_tokens=true`
- frame-conditioned text tokens in `add_to_text` mode
- frame tokens in `concat_tokens` mode

The temporal attention output projection is zero-initialized.

Implementation note: the context is kept at `[B, S_ctx, D]` and the query is
temporarily viewed as `[B, N*F, C]` for the cross-attention call. This is
mathematically equivalent to repeating the same context for each spatial token,
but avoids materializing `[B*N, S_ctx, D]`. At SDXL 1024 resolution, this avoids
multi-GB context tensors in the 64x64 transformer blocks.

### Frame-Dimension FFN

The adapter now includes the requested temporal FFN:

```text
LN -> Linear(C -> r) -> Conv1d over F -> SiLU -> Linear(r -> C)
```

It is applied over `[B*N, F, C]` after the original image FFN residual. The final
projection and scalar `gamma` are zero-initialized, so the block initially
matches the base image transformer.

## Config

```yaml
video_adapters:
  attention:
    enabled: true
    active: true
    train: true
    use_temporal_self_attention: true
    use_temporal_cross_attention: true
    use_temporal_ffn: true
    temporal_ffn_rank: 0       # 0 means C // 4
    temporal_ffn_kernel_size: 3
    temporal_ffn_gamma_init: 0.0
    include_prompt_tokens: true
```

## Image Prior Preservation

The new temporal attention outputs and temporal FFN output start at zero. This
keeps the adapter path initially residual-neutral. In `concat_tokens` mode, the
base spatial text cross-attention sequence length changes from `77` to
`77 + N_F`; that ablation can affect the base cross-attention distribution as
soon as it is enabled.
