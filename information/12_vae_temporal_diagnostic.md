# VAE Temporal Coherence Diagnostic (A2)

This is the read-only measurement that decides whether the frozen, frame-independent
SDXL VAE is a binding bottleneck for video flicker, or whether perceived flicker
is upstream in the UNet's latent trajectory.

Companion: `information/claude-codex-discussion.md` §2 A2 and §5.

## Why

The full pipeline encodes/decodes every frame independently through the frozen
SDXL VAE. The VAE decoder is deterministic, so it cannot manufacture stochastic
flicker by itself — but small latent differences across frames are amplified
into uncorrelated high-frequency texture changes (skin, foliage, fabric).

Before we spend any training compute on decoder adaptation or a temporal
decoder, we should know whether the VAE is responsible at all.

## What it measures

For each clip, sample F frames, decode them independently, then compute
**flow-warped consistency excess**:

```
excess(t→t+1) = E(decode_{t+1}, warp(decode_t, flow_GT))
              − E(gt_{t+1},     warp(gt_t,     flow_GT))
```

with two functionals `E`:

- `charbonnier` on **high-pass RGB residuals** (pixel-level shimmer signal)
- `vgg_feat` = LPIPS-style multi-layer VGG16 feature L2 (perceptual signal)

The flow is computed with **RAFT-Large** (`torchvision.models.optical_flow`).
A **forward-backward consistency mask** drops occluded pixels so motion that
the flow cannot follow does not pollute the metric:

```
valid = || flow_fwd(a→b) + warp(flow_bwd(b→a), flow_fwd) || < occlusion_threshold
```

Reporting:

- `gt_*` is the GT video's own warp-residual baseline (real motion + occlusion +
  imperfect flow), measured the same way for fair comparison.
- `decoded_*` is the same residual after independent VAE round-trip.
- `excess_ratio = (decoded − gt) / gt` — what fraction of additional temporal
  inconsistency the VAE contributes on top of the underlying signal.

## Verdict thresholds

| `excess_ratio_charbonnier` mean | Decision |
|--------------------------------|----------|
| `< 5%` | `VAE_OK` — not the binding bottleneck. Move on; revisit only if a future system already has very smooth latents and visible texture boil. |
| `5–10%` | `VAE_MARGINAL` — borderline. Defer training-side intervention until after [[A1]]. Re-run after E2. |
| `≥ 10%` | `VAE_BINDING` — real flicker source. Consider an R4 (pred_x0 consistency at mid/low noise) or R2 (flow-warp consistency) auxiliary loss, or evaluate a temporal VAE decoder. |

These thresholds are the ones agreed in `claude-codex-discussion.md` §2 A2 turn 2.

## Files

- `diagnostics/vae_temporal_diagnostic.py` — the diagnostic itself.
- `scripts/diagnostic_vae_temporal.sh` — launcher that wires up `.env` and the
  project python binary.

## Usage

**Smoke test** (32 clips, 4 contact sheets — ~5 min on one GPU):

```bash
bash scripts/diagnostic_vae_temporal.sh --smoke
```

**Recommended run** (100 clips, 20 contact sheets — ~25–30 min):

```bash
bash scripts/diagnostic_vae_temporal.sh \
  --num_clips 100 \
  --contact_sheet_count 20
```

**Robust run** (300 clips):

```bash
bash scripts/diagnostic_vae_temporal.sh \
  --num_clips 300 \
  --output_dir outputs/diagnostics/vae_temporal_robust
```

Outputs land under `--output_dir` (default
`outputs/diagnostics/vae_temporal`):

- `metrics.json` — aggregate statistics (mean / median / p25 / p75 / p90 / std)
  for every metric, plus `verdict` (one of `VAE_OK`, `VAE_MARGINAL`, `VAE_BINDING`).
- `per_clip.json` — per-clip metrics for re-analysis or stratification.
- `contact_sheets/clip_NNN_idxIIIIII.png` — 3-row sheet (GT / decoded /
  abs-diff heatmap clipped at 0.2) for the first `--contact_sheet_count`
  clips. Use these to look for **texture boiling** that the metric can miss.

## Configuration knobs

| Flag | Default | Note |
|------|---------|------|
| `--num_clips` | `100` | Statistical power; 100 is the recommended floor, 32 for a smoke test. |
| `--num_frames` | `8` | Same as the training config so the diagnostic reflects training-time clip structure. |
| `--resolution` | `512` | Same as the training config. |
| `--occlusion_threshold` | `1.0` | Pixels with FB-discrepancy ≥ this many pixels are masked. Lower = stricter occlusion handling. |
| `--highpass_sigma` | `1.5` | Gaussian std for the high-pass; lower = catches finer flicker. |
| `--seed` | `42` | Selects which clip indices are evaluated. Keep fixed for reproducibility. |
| `--contact_sheet_count` | `20` | Number of clips to save as side-by-side PNGs. |
| `--smoke` | — | Sets `num_clips=32`, `contact_sheet_count=4`. |

## Implementation notes

- VAE is loaded at **fp32** per the project's convention
  (`information/0_overview.md`).
- The "LPIPS-style" perceptual distance is a **multi-layer VGG16 channel-
  normalized L2**. It is *not* the human-calibrated LPIPS; install the `lpips`
  pip package and patch `VGGFeatureDistance` if you need the calibrated
  version. Codex's recommendation to use LPIPS was about catching perceptual
  flicker, which VGG features approximate well in *relative* terms across
  clips on the same data distribution.
- Flow is computed at the diagnostic's input resolution; for 512² it is
  ~5–7s/clip on a single GPU with `Raft_Large_Weights.C_T_SKHT_V2`.
- The script mirrors the training data sampling
  (`OpenVidVideoDataset(num_frames_per_video=F, frame_sampling="uniform")`)
  so the metric reflects what training actually sees. Clip indices are drawn
  by `torch.randperm` with the same `--seed`, so repeated runs evaluate the
  same clips.

## Reading the contact sheets

The third (`|diff|`) row is clipped at 0.2 with a `hot` colormap. Bright
regions are where the VAE's per-frame independent decode departs from the
GT pixel. Look for:

- **Persistent bright structure** — VAE is consistently re-rendering a region
  differently than the ground truth (encoding-bottleneck distortion). This is
  not flicker; this is single-frame fidelity.
- **Flickering brightness** that *changes between consecutive frames* — true
  temporal flicker. This is the failure mode the metric is designed to detect.

If the metric says `VAE_OK` but the contact sheets show texture boiling,
re-run with `--occlusion_threshold 0.5` and `--highpass_sigma 1.0` to make
the metric more sensitive.
