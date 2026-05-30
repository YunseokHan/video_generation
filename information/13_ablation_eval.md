# Ablation Evaluation Harness

Quantitative, multi-prompt, paired evaluation of trained image-first
checkpoints, with a single-wandb-run comparison view. This is the
"Minimum Viable Eval" (MVE) specified in
`information/claude-codex-discussion.md` §8 — built because train/loss curves
and single-prompt eyeballing cannot rank inference-time ablation axes
(switch mode, bridge, boundary) defensibly.

## Files
- `diagnostics/ablation_eval.py` — the harness.
- `scripts/ablation_eval.sh` — launcher (loads `.env`, project python).
- `diagnostics/prompts_ablation.txt` — curated 24-prompt synthetic set (4 motion
  buckets × 6).
- `diagnostics/build_eval_prompts.py` — mixes the synthetic set with **held-out
  OpenVid captions** into a 96-prompt benchmark (see below).
- `diagnostics/prompts_eval.txt` — the generated 96-prompt benchmark
  (24 synthetic + 72 held-out OpenVid), the recommended `--prompts_file`.

## Prompt set (what gets evaluated)
Two sources, selected by `--prompts_file`:
- omitted → 8 built-in `DEFAULT_PROMPTS` (smoke only).
- `diagnostics/prompts_ablation.txt` → 24 curated synthetic prompts.
- `diagnostics/prompts_eval.txt` → **recommended**: 24 synthetic + 72 held-out
  OpenVid captions = 96, matching §8's "controlled synthetic + held-out
  in-distribution" guidance and prompt count.

Build / rebuild the 96-prompt benchmark:
```bash
python diagnostics/build_eval_prompts.py --num_openvid 72 --output diagnostics/prompts_eval.txt
```
`build_eval_prompts.py` reads the OpenVid CSV (path from `.env`), keeps a
**deterministic held-out partition** (`hash(video) % 100 < --holdout_pct`,
default 10%), filters captions by motion score and length, dedups, and
round-robins across coarse camera-motion buckets (static/pan/tilt/zoom) for
diversity, then appends the curated synthetic prompts.

> Held-out caveat: the currently-running models were trained on the full
> OpenVid index, so for THOSE checkpoints these captions are "OpenVid-style
> in-distribution", not strictly unseen. To make the partition genuinely
> held-out for FUTURE training, exclude the same hash partition in the
> dataloader (recipe printed by the script).

## What it does
1. For each model name under `outputs/<name>/checkpoint-<step>` (default
   step = `last`), loads the trained pipeline (reusing
   `infer_image_first.load_image_first_pipeline`).
2. Generates a clip for every (prompt, seed) at a **fixed operating point**
   (`--cfg`, `--t1`) for comparability, but with each model's **own** inference
   switch settings (`image_first_switch_mode` / renoise) read from its
   `config.yaml` — because the switch mode is itself part of the ablation.
3. Computes per-clip metrics, aggregates, runs a paired bootstrap decision
   rule vs a baseline model, and writes artifacts + (optionally) one wandb run.

The stage-1 anchor image is saved per clip via the new `anchor_image_path`
argument of `generate_image_first_video_frames` (decoded with video adapters
off) so anchor-fidelity metrics have a reference.

## Metrics (per clip)
| metric | meaning | better |
|--------|---------|--------|
| `clip_t` | mean OpenCLIP frame↔text cosine over frames {0,2,4,6} | higher |
| `clip_anchor` | OpenCLIP anchor↔frame0 image cosine | higher |
| `vgg_anchor` | VGG16 feature L1, anchor↔frame0 | lower |
| `pix_anchor_low` | 64²-downsampled L1, anchor↔frame0 | lower |
| `warp_vgg` | RAFT flow-warped, occlusion-masked VGG error over adjacent frames | lower |
| `motion_mag` | median RAFT flow magnitude | — (report) |
| `motion_cov` | fraction of pixels with flow > 2px | guardrail |

`clip_t` / `clip_anchor` require `open_clip_torch`. If it is not installed the
harness still runs and sets those to `NaN` (RAFT + VGG + pixel + motion metrics
always work, reusing the A2 diagnostic's backbones). Install with:
`pip install open_clip_torch`.

### Primary metric by ablation axis (from §8)
- cross-attn on/off → `clip_t` (guardrails `warp_vgg`, `motion_cov`)
- bridge smooth vs hard → `warp_vgg` (guardrails `clip_t`, `vgg_anchor`)
- boundary loss on/off → `vgg_anchor` / `pix_anchor_low`
- switch pred_x0_renoise vs repeat → `vgg_anchor` / `clip_anchor`
- anchor on/off (E2) → `clip_anchor` then `vgg_anchor`

## Decision rule (paired, vs baseline)
1. Per-prompt delta at identical seeds (average over seeds within each prompt).
2. 20% trimmed mean across prompts.
3. 10k-bootstrap 95% CI over prompts.
4. `sigma_null` from baseline-only extra seeds: split the baseline seeds in half
   and take the std of baseline-vs-baseline per-prompt deltas (needs ≥4 baseline
   seeds; otherwise `NaN`).
5. An ablation is **DIFFERENT** iff the 95% CI excludes 0 **and**
   `|delta| > 2·sigma_null` (the latter skipped only if `sigma_null` is NaN).

This guards against calling a difference real when it is below run-to-run
(seed/data-order) noise — verified on synthetic data: a true −0.08 effect on
`vgg_anchor` is flagged DIFFERENT, while a CI-significant but sub-2σ `clip_t`
blip is correctly left "ns".

## Outputs
Under `--output_dir` (default `outputs/eval/<timestamp>/`):
- `per_clip.json` — every (model, prompt, seed) metric row.
- `summary.json` — per-model trimmed means, `sigma_null`, and the paired
  `decisions` list.
- `<model>/p###_s#/` — generated frames, `video.mp4`, `anchor.png` per clip.

With `--wandb`, **one run** holds it all:
- `eval/per_clip` (long table), `eval/per_model_summary`, `eval/decisions`,
- `eval/samples` (rows = prompts, columns = models, cells = `wandb.Video`) for
  side-by-side comparison,
- `summary/<metric>/<model>` scalars for grouped bar charts.

## Usage
```bash
# Smoke (no wandb): 2 models, 1 prompt, 1 seed
bash scripts/ablation_eval.sh \
  --models image-first-smooth-snr-renoise-boundary,image-first-smooth-snr-renoise-boundary-xnocross \
  --num_prompts 1 --seeds 0 --baseline_extra_seeds "" --num_log_videos 1

# Full MVE over the 6 runs -> one wandb run
bash scripts/ablation_eval.sh \
  --models image-first-smooth-snr-renoise-boundary,image-first-smooth-snr-renoise-boundary-xnocross,image-first-smooth-snr-boundary,image-first-smooth-snr,image-first-snr-renoise,image-first-smooth-snr-renoise \
  --baseline image-first-smooth-snr-renoise-boundary \
  --prompts_file diagnostics/prompts_ablation.txt \
  --seeds 0,1,2 --baseline_extra_seeds 3,4,5 --cfg 8 --t1 0.5 --wandb

# Re-aggregate / re-log without regenerating (no GPU):
bash scripts/ablation_eval.sh --models a,b --from_per_clip outputs/eval/<ts>/per_clip.json --wandb
```

`--step <N>` selects `checkpoint-<N>`; default `last` → `checkpoint-last`.
`--from_per_clip` reuses a prior `per_clip.json` to recompute aggregation /
decisions / wandb tables with no model loading (useful for tuning the decision
rule or re-logging).

## Notes / limitations
- The full sweep loads each model sequentially (frees the pipeline between
  models), so it needs one free GPU; run it after the training jobs release
  GPUs. Generation for 8-frame 512² clips is a few seconds each.
- `vgg_frechet` / FVD are intentionally omitted; VGG-Frechet is acceptable only
  as an internal realism proxy. For publication-grade claims install a real
  video feature extractor (see §8).
- Each model is evaluated at ONE operating point (CFG, t1) for first decisions.
  For a winner vs baseline vs nearest competitor, re-run a small stress sweep
  over the full t1×CFG grid; ranking reversal across operating points ⇒ axis
  "unresolved".
