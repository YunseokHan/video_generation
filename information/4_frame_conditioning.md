# Frame Conditioning

Frame conditioning is split into pooled conditioning for SDXL added-conditioning
and token conditioning for cross-attention.

## Pooled SDXL Conditioning

`framegen/sdxl.py` keeps the original milestone path:

```text
frame_position: [B*F]
FramePositionMLP -> [B*F, 1280]
pooled_prompt_embeds + alpha * frame_embedding
```

The modified pooled embedding is passed through SDXL `added_cond_kwargs`:

```python
added_cond_kwargs={
    "text_embeds": modified_pooled_prompt_embeds,
    "time_ids": add_time_ids,
}
```

## Frame Position Encoder

`framegen/temporal.py` contains two encoder families.

### Sinusoidal

```text
frame_position -> sinusoidal embedding [D]
tokens: repeat to [N_F, D]
pooled: [D]
```

This has no trainable parameters.

The sinusoidal frequency vector is cached as a non-persistent buffer, so each
step only moves/casts the cached vector instead of rebuilding the full frequency
range on the GPU.

### Learnable Token Encoder

The learned token path follows the requested concat-token structure:

```text
frame_position -> MLP -> frame_pos_emb [D]
token_basis: [N_F, D]
frame_token = frame_pos_emb + token_basis
token_refine_mlp(frame_token)
```

It returns:

```text
pooled: [B, F, D]
tokens: [B, F, N_F, D]
```

## Token Embedding Ablation Modes

Configured at:

```yaml
video_adapters:
  frame_position_encoder:
    enabled: true
    type: "sinusoidal"       # or "learned_tokens"
    token_embedding_mode: "add_to_text"
    alpha: 1.0
    embedding_dim: 2048
    num_tokens: 1
```

`configs/train/default.yaml` uses the sinusoidal `add_to_text` setup. The
learnable ablation config is `configs/train/learnable_frame_tokens.yaml`,
which uses `type: learned_tokens`, `token_embedding_mode: concat_tokens`, and
`num_tokens: 4`.

### `add_to_text`

Adds frame embeddings to the SDXL text tokens:

```text
prompt_embeds:  [B*F, 77, 2048]
frame_embeds:   [B*F, 2048]
output:         [B*F, 77, 2048]
```

The temporal cross-attention adapter receives the frame-conditioned text tokens
reshaped as:

```text
[B, F, 77, 2048]
```

### `concat_tokens`

Concatenates frame tokens to the SDXL text sequence:

```text
prompt_embeds:  [B*F, 77, 2048]
frame_tokens:   [B*F, N_F, 2048]
output:         [B*F, 77 + N_F, 2048]
```

The temporal cross-attention adapter receives:

```text
[B, F, N_F, 2048]
```

This is the primary ablation mode for learnable frame token concatenation.

### `temporal_cross_attention_only`

Keeps the SDXL text sequence unchanged and sends frame tokens only to the
temporal cross-attention adapter. This preserves the previous repository
behavior.

### `none`

Disables token-level frame conditioning from the frame position encoder.

## Classifier-Free Guidance

During generation, the same frame conditioning is applied to positive and
negative prompt embeddings so CFG keeps frame position as a shared condition
rather than making it part of the text guidance difference.
