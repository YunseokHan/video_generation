# Claude–Codex Discussion Log

> Living document. Tracks the critical methodology review of the image-first video diffusion
> design between Claude (the planning/analysis agent) and Codex (the second-opinion agent).
> Every new round of discussion — including corrections to earlier conclusions — is appended
> chronologically inside its agenda section. **Always update this file** when a new discussion
> happens or when a prior conclusion is revised.
>
> Maintained per the repo's `CLAUDE.md` maintenance rule. Companion of
> `information/10_image_first_training.md` and `information/11_latent_calibrator.md`.

---

## 0. Context Snapshot

- Active ablation under review: `configs/train/image_first_smooth_snr_renoise_boundary.yaml`.
- Pipeline: frozen SDXL backbone + frame-aware adapters
  (`framegen/video_resnet.py`, `framegen/video_attention.py`,
  `framegen/temporal.py`, optional `framegen/latent_calibrator.py`).
- Two-stage inference (`framegen/image_first_generation.py`):
  Stage 1 base SDXL denoise of one image latent with video adapters OFF →
  switch (`pred_x0_renoise`) → Stage 2 video denoise with adapters ON.
- Training objective is the epsilon that reconstructs `z*` from a smooth-SNR-blended
  noising source (`g(t)·anchor + (1−g(t))·z*`), plus a low-frequency boundary
  re-noise auxiliary loss at SNR=5 weight 0.05.

The researcher already listed three concerns at the start of the review:

- **U1** — Adapter is too heavy (~1.1B trainable params, ~1B in attention).
- **U2** — Training/inference manifold mismatch (the `latent_calibrator` in
  `image_first_snr_ea.yaml` is the current workaround).
- **U3** — Forward/reverse process consistency
  (the `smooth_snr_renoise_boundary` design addresses it).

Claude additionally raised three new agendas (A1–A3) on top of those concerns.

---

## 1. Agenda Index

| ID | Topic | Status | Owner of next step |
|----|-------|--------|--------------------|
| A1 | Why first-frame anchoring? → persistent anchor conditioning | Closed v1 | Implementation |
| A2 | Frozen frame-independent VAE as quality ceiling | **Closed** (2026-05-29): `VAE_OK` at 300 clips | No training-side action; revisit only if E2 exposes content-specific flicker |
| A3 | Attention adapter capacity allocation | Closed v1 | Implementation |
| U1 | Adapter weight — structural alternatives | Closed | Tracks A3 |
| U2 | Manifold mismatch — `latent_calibrator` redesign | Closed v1 | Implementation |
| U3 | Forward/reverse consistency — SNR endpoints, rollout, boundary | Closed v1 | Implementation |

---

## 2. Claude-raised Agendas

### A1 — First-frame anchoring justification and persistent anchor conditioning

**Concern.** The pipeline is structured around the first frame being THE anchor.
This is convenient (stage 1 just runs SDXL on text), but the anchor only enters
training as a **noising source** — there is no mechanism inside the denoising
chain that keeps the model honest to the stage-1 image.

**Discussion path (3 turns).**

1. Codex turn 1 — convenience or principled?
   - First-frame anchoring is the cleanest *causal contract*: SDXL defines scene
     identity, video adapters learn motion residual on top.
   - "Anchor as noising source only" is too weak. Better: **persistent anchor
     conditioning** through cross-attention K/V, ControlNet residual, or extra
     UNet input channels.
   - Random-anchor-position is implementable but inference semantics are weak
     (SDXL does not natively generate "frame k of an action").

2. Codex turn 2 — pick the mechanism.
   - Among (i) K/V token, (ii) ControlNet residual, (iii) extended UNet input
     channels: **(i) wins** for image-first because it preserves SDXL load
     compatibility, reuses the existing `VideoBasicTransformerBlock` temporal
     cross-attn path, and stays small.
   - Anchor feature = VAE/`pred_x0` latent (cheap, spatially meaningful, aligned
     with `smooth_snr`). Multi-resolution spatial tokens beat one global token.
   - Identity at checkpoint load requires a **separate gated anchor-attn branch**
     (not naive zero K/V append, which still changes softmax normalisation).
   - Inference anchor = `pred_x0` at the switch, *not* the re-noised latent.

3. Codex turn 3 — sharp proposal.
   - **Token design**: spatial-aligned anchor K/V tokens at each UNet resolution
     (per spatial site, projected from anchor latent).
   - **Training anchor source**: clean `z*_1` (matches inference's `pred_x0`).
   - **Gating**: separate anchor-attn contribution `output = temp_cross + gate·anchor`,
     `gate=0` at init (scalar or per-channel per block), projector normally inited.
   - **Files touched** — `framegen/video_attention.py`, `train.py`,
     `framegen/image_first_generation.py`, the active config.
   - **Latent-calibrator interaction** — orthogonal (calibrator fixes input
     distribution, anchor branch gives a clean reference). 2×2 ablation: baseline
     / calibrator-only / anchor-only / both. When combined, anchor tokens must
     be derived from clean `z*_1` (train) or `pred_x0` (infer), *never from
     calibrated latents*.

**Decision (v1).** Ship spatial-aligned, gated anchor K/V conditioning in
temporal cross-attn at mid+up blocks only. Continuation fine-tune on top of
the current `image_first_smooth_snr_renoise_boundary` checkpoint.

---

### A2 — Frozen, frame-independent VAE as a quality ceiling

**Concern.** Every frame is encoded/decoded independently with the frozen SDXL
VAE. `vae_decoder_resnet` adapters exist but `train: false`. A perfectly
temporally coherent latent does not guarantee a temporally coherent pixel video.

**Discussion path (2 turns).**

1. Codex turn 1 — strong pushback.
   - VAE is **probably not a top-3 bottleneck yet**: the decoder is deterministic;
     it amplifies upstream latent inconsistency rather than producing stochastic
     flicker on its own. UNet latent inconsistency, sampling schedule, and weak
     temporal conditioning are usually larger.
   - Run a diagnostic *first* before any training-side intervention.

2. Codex turn 2 — concrete plan.
   - **Diagnostic** — flow-warped excess error
     `E(decode_{t+1}, warp(decode_t, flow_GT)) − E(gt_{t+1}, warp(gt_t, flow_GT))`,
     LPIPS + Charbonnier on high-pass RGB, with occlusion masks.
     Budget: 100 clips × 8 frames, stratified by content (faces, foliage,
     fabric, low/medium/high motion). Smoke test 32 clips. Decision:
     `excess <5%` → VAE not the bottleneck; `10–20%` → real issue.
   - **If training-side action is justified**, R4 (pred_x0 consistency between
     adjacent frames at mid/low noise) is the best pragmatic loss. R2
     (flow-warp consistency) is most principled. R1 (temporal Laplacian) is too
     blunt (penalizes acceleration → suppresses real motion). R3 (FFT
     high-pass) is only useful as flow-compensated residual energy.
   - **Public reference** — no canonical SDXL VAE video-flicker benchmark
     exists; this is a research opportunity if the diagnostic turns out positive.
   - **Priority** — ship A1 first. Spend 1–2 days on the diagnostic; not the
     2–3 weeks of training compute.

**Decision.** Diagnostic-only this cycle. Postpone any decoder-adapter
training until the diagnostic justifies it.

#### A2 update — 2026-05-29 — diagnostic implemented

Implemented the read-only diagnostic. Files:

- `diagnostics/vae_temporal_diagnostic.py` — main script.
  - SDXL VAE (fp32) round-trip per frame on F=8 at 512².
  - RAFT-Large (`torchvision.models.optical_flow`, weights
    `C_T_SKHT_V2`) for `flow_GT`.
  - Forward-backward consistency mask for occlusion.
  - Two error functionals: high-pass-RGB Charbonnier and
    LPIPS-style VGG16 channel-normalized feature L2 (the `lpips` pip
    package is missing in the `video` env, so the diagnostic ships a
    VGG substitute; install `lpips` and swap `VGGFeatureDistance` if
    calibrated LPIPS is needed).
  - Reads OpenVid via existing `OpenVidVideoDataset` so the sample
    distribution matches training.
- `scripts/diagnostic_vae_temporal.sh` — launcher (loads `.env`, project
  `PYTHON_BIN`).
- `information/12_vae_temporal_diagnostic.md` — design doc with verdict
  thresholds, knobs, and how to read contact sheets.

Verdict thresholds remain the ones agreed in turn 2:
`<5%` = `VAE_OK`, `5–10%` = `VAE_MARGINAL`, `≥10%` = `VAE_BINDING`,
keyed off the **Charbonnier excess ratio mean** (VGG ratio is reported
in parallel for sanity).

Smoke-test on 3 clips ran end-to-end successfully. With only 3 clips the
mean is dominated by one outlier (one clip at +55%, two at −12%); the
script exits with a verdict but the recommended floor is `--num_clips 100`
for a stable mean. To run:

```bash
bash scripts/diagnostic_vae_temporal.sh --num_clips 100
```

#### A2 result — 2026-05-29 — `VAE_OK` at 300 clips

Ran the diagnostic at `--num_clips 300` (2100 frame pairs, occlusion-valid
fraction 0.795). Headline statistics:

| Metric | Mean | Median | p75 | p95 |
|--------|------|--------|-----|-----|
| `gt_charbonnier` | 0.0181 | 0.0125 | 0.0240 | (—) |
| `decoded_charbonnier` | 0.0161 | 0.0115 | 0.0208 | (—) |
| `excess_ratio_charbonnier` | **−5.3%** | −9.5% | −5.8% | +30.0% |
| `excess_ratio_vgg_feat` | **+2.0%** | −0.9% | +0.4% | +5.8% |

Tail statistics:

- 9.0% of clips exceed `excess_ratio_charbonnier > 0.10`.
- 8.7% of clips exceed `excess_ratio_vgg_feat > 0.10`.
- 71% of clips have **negative** Charbonnier excess (VAE smooths slightly).

**Verdict**: `VAE_OK`.

The negative mean is interesting evidence that the SDXL VAE acts as a mild
low-pass on per-frame high-frequency texture: real-video Charbonnier on
high-pass RGB residuals is *higher* than after a VAE round-trip, because
some of the original high-frequency content is what makes consecutive
frames disagree after flow-warping (sensor noise, compression, real
texture noise). The VAE removes some of that. This is a single-frame
fidelity property, not flicker, and is out of A2's scope.

**Content stratification of the binding tail.** Inspection of the top-10
worst clips (highest `excess_ratio_charbonnier`) shows a clear pattern:
fine-texture, high-detail scenes (beaches with sand, mountains with
foliage, cityscapes at night, hummingbird-in-rain). The most VAE-friendly
clips (most negative excess) are smooth aerial views and large
low-detail regions. So the residual ~9% of binding clips is concentrated
in **high-frequency-texture content**, not in any specific motion regime.

**Decision.** Close A2 as `VAE_OK`. No training-side decoder intervention
(R4 pred_x0 consistency, R2 flow-warp consistency, or temporal VAE
replacement) is justified at this point. The 2–3 weeks of training
compute that codex flagged in §2 A2 goes to A1 + A3 instead.

**Re-open condition.** If a future evaluation step (e.g., post-E2)
reports content-specific flicker concentrated on high-frequency texture,
re-run the diagnostic with a stratified sampler (e.g., split by an
aesthetic / texture-density predictor) to confirm the pattern before
investing in decoder-side work.

---

### A3 — Attention adapter capacity allocation

**Concern.** "Adapter is too heavy" is not actionable; the right question is
*where* the 1B in attention modules actually buys capacity.

**Discussion path (2 turns + code verification).**

1. Codex turn 1 — first cut (corrected later).
   - Codex initially assumed a standard FFN multiplier (8–12 `C²`) and
     concluded FFN might be the largest single submodule.
   - Recommended order: remove cross-attn → shrink FFN → restrict placement →
     LoRA on self-attn.

2. Claude verified the actual implementation
   (`framegen/video_attention.py:113-149`) — the temporal FFN is **already
   rank-factorised** at `rank = C/4`:
   - `Linear(C → C/4)` + `Conv1d(rank → rank, k=3)` + `Linear(C/4 → C)`
     ≈ `C²/4 + 3C²/16 + C²/4 ≈ 0.69 C²` per block.
   - Per-block budget at hidden width `C`:
     - temporal self-attn (Q/K/V/O): ~4 `C²`  (46%)
     - temporal cross-attn (Q/K/V/O): ~4 `C²`  (46%)
     - temporal FFN: ~0.69 `C²`  (8%)
     - Total: ~8.69 `C²`.

3. Codex turn 2 — revised plan, agreed.
   - Removing cross-attn saves ~46% (not 20–25%). FFN compression is marginal.
   - SDXL block split: down 24 / mid 10 / up 36, weighted by `C²`. Mid+up
     restriction saves ~30–35% more, not exactly 50%.
   - Self-attn compression order: **(a) full mid + same-width sharing in up
     blocks** → (b) same-width sharing inside mid+up → (c) shared base +
     per-block LoRA delta → (d) pure LoRA. Pure head reduction does **not**
     save params unless inner projection width is also reduced.

**Decision (v1).**

| Step | Action | Cumulative trainable |
|------|--------|----------------------|
| 0 | Baseline | ~1.0–1.1 B |
| 1 | Remove temporal cross-attn (`use_temporal_cross_attention: false`) | ~0.54 B |
| 2 | Restrict adapter placement to mid+up blocks (new config flag) | ~0.35–0.40 B |
| 3 | (Later) self-attn weight sharing inside same-width up blocks | conditional |

Note: A1 anchor branch must also obey the mid+up restriction; if it were
added at full `4C²` in all 70 blocks it would undo A3.

---

## 3. Researcher-raised Agendas (deep-dived in turn)

### U1 — Adapter weight, structural alternatives

**Question to codex.** Beyond incremental compression (A3), are there
structural alternatives — single temporal pre-net, MM-DiT-style bottleneck
joint attention, side-network, or reusing SDXL's existing spatial cross-attn
by appending frame tokens to K/V — worth pursuing now?

**Codex (1 turn).**

- Strongest theoretical case for image-first: **(γ) side network** (a small
  parallel UNet injecting residuals at chosen layers; like SCEdit / T2I-Adapter).
- (δ) frame-token reuse of SDXL's cross-attn is *parameter-free* but does change
  the activations seen by frozen layers — frozen-weights promise is kept,
  frozen-forward-path promise is weakened.
- Cleanest removability: (α) pre-net > (γ) side-net > (β) bottleneck attention >
  (δ) frame-token append.
- **Pragmatic call: incremental compression first.** A structural redesign
  confounds capacity / placement / training dynamics / conditioning interface
  all at once. Treat (γ) as the next-generation architecture.

**Decision.** Stay on A3's incremental path. Keep (γ) in the backlog.

---

### U2 — Train/inference manifold mismatch (`latent_calibrator` redesign)

**Question.** The calibrator currently has `apply_mode: switch_only` and is
keyed off the hard SNR mask. With smooth_snr active, what does the principled
version look like?

**Discussion path (2 turns).**

1. Codex turn 1.
   - Input `concat[z_t, E(a), z_t − E(a)]` is fine but the delta is linearly
     redundant. Bridge gate `g(t)` and `log SNR` should enter via **FiLM
     conditioning**, not as input channels.
   - Residual scale `clipped_snr` capped at 0.5 is a trust-region knob, not a
     theorem. Better: `scale(t) = g(t) · min(√SNR(t), s_cap)` with `s_cap`
     ablated (0.5 vs 1.0).
   - `switch_only` is not aligned with smooth_snr's continuous gate. New
     `gate_scaled` apply mode: residual magnitude continuous across the bridge
     zone (`δz = g(t)·clipped_snr·calibrator(...)`).
   - Aux loss `||Down(z_t,calib) − Down(z_t,video)||²` is meaningful at
     intermediate `g(t)` **only if the same noise tensor is shared** between
     `z_t,calib` and `z_t,video`. Otherwise the calibrator is being asked to
     predict stochastic noise it cannot know.
   - With Agenda A1's persistent anchor branch in play, the calibrator is
     **not redundant**: anchor branch gives the network a clean reference
     inside attention, calibrator fixes the pre-UNet input distribution
     before early convs and norms. Different failure modes; 2×2 ablation
     warranted.

2. Codex turn 2 — code-level change list.
   - Conditioning plumbing: small MLP / scalar embed for `g(t)` and `log SNR`
     before joining FiLM.
   - `gate_scaled` aux loss should be **gate-weighted** (e.g. `w = g(t)` or
     `g(t)²`, normalised by `mean(w) + ε`) so `g≈0` samples don't dominate.
   - Gate the `norm_weight` regulariser similarly.
   - Update **every** calibrator call site, not only `train.py` — inference,
     validation, eval preview.
   - Defaults: `switch_only` keeps `g = bridge_mask.float()`; no-bridge mode
     uses `g = 0` (skip).
   - Checkpoint compatibility: when `apply_mode` field is missing default to
     `switch_only`.
   - Add tests: identity at `g=0`, switch_only unchanged, shared-noise
     verified, continuous residual scale at smooth gate.

**Claude verification of the shared-noise concern.** Read
`train.py:2030–2095`:
- `noise = bridge_noise` for both `always` and `smooth_snr` modes (lines 2070,
  2084).
- `noisy_latents = add_noise(noising_latents, noise, t)` uses `noise = bridge_noise`.
- `latent_calibrator_alignment_target = add_noise(clean_latents, bridge_noise, t)`
  uses the **same** tensor.
- `boundary_noise = noise.detach()` (line 2307) is still that same tensor.
- ✅ All three losses (`L_diff`, `L_boundary`, `L_calib_map`) share `bridge_noise`
  in the `smooth_snr` path. No bug. Tested only with smooth_snr; under hard
  `snr` mode `noise = where(mask, bridge_noise, standard_noise)` may diverge
  from the calibrator target — verify before re-enabling calibrator there.

**Decision (v1).** Implement `gate_scaled` apply mode, with FiLM-injected
`g(t) / log SNR`, gate-weighted aux loss, and a new
`configs/train/image_first_smooth_snr_renoise_boundary_calib.yaml`. Ablate as
2×2 against A1's anchor branch.

---

### U3 — Forward/reverse process consistency (SNR endpoints, boundary, rollout)

**Question.** The `[SNR 1, SNR 5]` cosine window, the single-point boundary
loss, and the `pred_x0_renoise` validation — are any of these principled or
just hyperparameters?

**Discussion path (2 turns).**

1. Codex turn 1 — sharp theoretical answers.
   - Define `δ = anchor − z*` and `c(t) = √SNR(t)·g(t)·RMS(δ)`. This is the
     anchor correction in noise-floor units. The smooth-SNR window should be
     *calibrated from data*, not guessed: switch the gate off once `c(t)`
     dominates noise.
   - One boundary point at SNR=5 is theoretically thin but pragmatically
     justifiable: the standard denoising loss already supervises the interior;
     only the stage-1/stage-2 handoff at the upper edge is the fragile point.
     A spread-out version would be "sample `t` in the gate, predict `x0`,
     re-noise to a second `s`, match `q_s(z* | same ε)`" — paired-noise
     consistency dominates MMD/moment matching.
   - A unified objective exists at the level of **marginal score matching**:
     train the score of `q_g(z_t | z*, a) = N(α_t[z* + g(t)(a−z*)], σ_t² I)`.
     But the boundary loss is **not** automatic from that objective; it is a
     paired `x0` consistency term that a real ELBO would have to derive from
     a forward process with drift `g'(t)·(a−z*)`. Current design mixes a
     marginal score fit with a hand-crafted handoff aux loss.
   - `pred_x0_renoise` is exact for VP/DDPM schedulers. It can be wrong for
     EDM, `v_prediction`, ancestral schedulers, off-by-one switch index, or
     custom sigma scaling. The real residual mismatch is **not** noise
     independence but the fact that *inference re-noises a stage-1 predicted
     `x0`, while training bridges from the **true** anchor latent*.

2. Codex turn 2 — rollout, calibration, versioning.
   - **Rollout source** closes the residual U3 gap IF the rolled latent is
     fed in as an `x0`-like quantity. The existing `rollout` mode
     (`train.py:608`) returns a **noisy-at-target_t** latent (verified by
     reading the code), so plugging it directly into the smooth-SNR formula
     would double-count noise. Two options: (a) compute `pred_x0` from the
     rollout endpoint and use that as the bridge source, or (b) redefine the
     bridge in noisy-latent space. `K` (rollout steps) should be **fixed** to
     match inference's switch depth, not sampled.
   - **Principled SNR endpoints**:
     - `c_low ≈ 0.2–0.5` for `g=1` region (anchor correction negligible vs noise)
     - `c_high ≈ 1.0–2.0` for `g=0` region (anchor correction dominates noise)
     - Compute `RMS(δ)` from OpenVid at **p50/p75/p90 by frame distance**.
       Start with a scalar gate from a high quantile; move to frame-distance
       buckets only if far frames remain bad.
   - **Versioning correction** — Claude proposed v1→calibrator→rollout, codex
     **swapped** the order:
     - v1: smooth_snr + boundary + pred_x0_renoise + persistent anchor (A1)
     - v2: rollout source replacing clean anchor
     - v3: calibrated SNR gate using v2's rollout-source `δ` distribution
     - Reason: the gate must be calibrated on the same δ distribution that
       training actually uses; calibrating before rollout optimises the wrong
       bridge.
   - Each step should own ONE metric: switch re-noise consistency, first-frame
     preservation, FVD/temporal coherence, boundary `x0` low-frequency error.

**Claude verification of rollout return type.**
`rollout_image_first_anchor_latents` (`train.py:607-687`) returns the *latent
after `scheduler.step` from `target_t + K` down to `target_t`*. This is a
noisy-at-`target_t` quantity, **not** `pred_x0`. ✅ Codex's concern is real.

**Decision (v1).**
- v1 = smooth_snr + boundary + pred_x0_renoise + persistent anchor (A1).
- v2 follow-up requires adding a `pred_x0`-extraction step to the rollout
  function (call it `rollout_image_first_anchor_pred_x0`) before plugging it
  into the smooth_snr bridge as the source.
- v3 calibration runs only after v2 is live.

---

## 4. Integrated Roadmap (snapshot — update on revision)

| Phase | Scope | Files |
|-------|-------|-------|
| **Week 1** | A3.1 remove temporal cross-attn; A2 diagnostic in parallel | `framegen/video_attention.py`, `configs/train/*.yaml`, ad-hoc diagnostic script |
| **Week 2–3** | A3.2 mid+up placement; A1 v1 persistent anchor branch; continuation fine-tune; 2×2 ablation with U2 redesigned calibrator | `framegen/video_attention.py`, `framegen/latent_calibrator.py`, `train.py`, `framegen/image_first_generation.py`, new configs |
| **Week 4+** | If A2 diagnostic positive: R4 pred_x0 consistency; A3.3 self-attn weight sharing | conditional |
| **Backlog** | U3 v2 rollout `pred_x0` source; U3 v3 data-calibrated SNR gate; U1 (γ) side-network architecture | future |

---

## 5. Next Experiment Design

Two experiments are queued. **E1 ships first** because it is config-only and
isolates a single falsifiable claim. **E2 follows** once E1 confirms (or
falsifies) the cheap version of A3.

---

**Files generated for this section** (2026-05-29):
- `configs/train/image_first_smooth_snr_renoise_boundary_xnocross.yaml` (E1)
- `scripts/train_image_first_smooth_snr_renoise_boundary_xnocross.sh` (E1)
- `configs/train/image_first_smooth_snr_renoise_boundary_anchor.yaml` (E2, header
  explicitly notes the required code changes; spec-precise but not runnable
  until the anchor branch + placement filter land)
- `scripts/train_image_first_smooth_snr_renoise_boundary_anchor.sh` (E2 launcher)

#### E1 fix — 2026-05-29 — `use_temporal_cross_attention=False` was a no-op

The first E1 launch hit a DDP `find_unused_parameters` failure on all four
ranks. Trainable param count at startup was still **1.140B** (identical to
baseline), confirming the temporal cross-attn weights were *instantiated and
treated as trainable* even though their forward was skipped — DDP then
complained that they did not receive gradients.

Root cause (`framegen/video_attention.py`): `VideoBasicTransformerBlock.__init__`
always instantiated `temporal_cross_norm` / `temporal_cross_attn` regardless
of the `use_temporal_cross_attention` flag, and `adapter_parameters` always
yielded them. `set_video_attention_adapter_requires_grad(unet, True)` then
flipped them to `requires_grad=True`, which (a) made the trainable-param
counter still report the full 1.140B and (b) registered them with DDP's
all-reduce expectations.

Fix (minimal, keeps state-dict format compatible with old baseline
checkpoints):
1. In `__init__`, freeze parameters of any sub-module whose `use_*` flag is
   `False` (symmetric across self-attn / cross-attn / FFN).
2. Update `adapter_parameters` to skip yielding parameters from disabled
   sub-modules so `train: true` cannot accidentally re-enable them.

Sub-modules are still instantiated so checkpoints saved under the default
config still load unchanged.

Verified at SDXL scale (per-block dim=1280, cross_attention_dim=2048):
per-block adapter trainable drops **52.6%** when cross-attn is disabled.
Extrapolated to 70 transformer blocks: **~1.135B → ~0.538B** trainable,
matching the v1 target in §2 A3.

#### E1 perf — 2026-05-29 — removed per-micro-step CUDA syncs

The user reported the smooth-SNR family (E1 included) trained noticeably
slower than earlier experiments. Root cause was **GPU→CPU synchronization
on every micro-batch**, not the adapter compute:

The whole training body runs inside `accelerator.accumulate(...)` and
repeats `gradient_accumulation_steps` (=4) times per optimizer step. Three
diagnostics-only values called `.item()` inside that loop:

- `sample_image_first_noise(...)` returned `mask.float().mean().item()`
  (`train.py:429`).
- `image_first_bridge_fraction = bridge_mask.float().mean().item()`
  (`train.py:2048`).
- the smooth_snr branch `image_first_bridge_fraction = gate...mean().item()`
  (`train.py:2061`).

So **~3 syncs × 4 accumulation = ~12 forced CUDA syncs per optimizer step**,
all before the UNet forward, draining the kernel queue and killing
forward/all-reduce overlap. These values are consumed only by the
once-only `logged_shapes` block and the `logging_steps`-gated metrics block.

Fix (no semantic change to training):
- `sample_image_first_noise` now returns the fraction as a **0-dim tensor**
  (drops `.item()`).
- The two `image_first_bridge_fraction` assignments keep the value as a
  tensor.
- New module-level `_scalar(x)` helper converts tensor-or-float → float; it
  is called **only** in the two logging consumers (first-step shapes log and
  the gated metrics dict), where a sync is already acceptable.

Verified: `train.py` parses, `_scalar` handles float/CPU-tensor/GPU-tensor,
`sample_image_first_noise` returns a tensor fraction, and no `.item()`
remains in the per-micro-step path (the only residual `.item()` calls in
1845–2340 are inside the `if not logged_shapes:` once-only block).

This optimization helps **every image-first config**, not just E1. Other
shared costs (fp32 VAE encode every micro-step, ffmpeg decode in the
dataloader) were left as-is since they are not specific to the implemented
methodology and affect all runs equally.

### Experiment E1 — "Cross-attn-free temporal adapter" (A3.1 isolated)

#### Hypothesis

> Under `token_embedding_mode: add_to_text`, the per-block temporal
> cross-attention adds **no measurable training-loss or validation-quality
> improvement** beyond what temporal self-attention + frame tokens in the
> text path already provide. Removing it saves ~46% of the temporal
> adapter trainable parameters at near-zero quality cost.

#### Theoretical basis

1. Frame-position information already enters the network through two
   independent channels with the current configuration:
   - `add_to_text` adds the sinusoidal frame embedding to the SDXL text
     encoder hidden states (`framegen/temporal.py`,
     `framegen/sdxl.py::add_temporal_embedding_to_pooled_prompt_embeds`),
     so every SDXL spatial cross-attn already sees a frame-conditioned text
     stream.
   - Temporal self-attention over `[B*N, F, C]`
     (`framegen/video_attention.py:113`) gives every spatial location a
     direct attention window across frames, which is the actual mechanism
     by which motion is learned.

2. The temporal cross-attention block reads K/V from `frame_attention_tokens`
   produced by the frame position encoder
   (`framegen/temporal.py::FramePositionEncoder`). With
   `frame_position_encoder.train: false` and the encoder set to
   `sinusoidal`, those K/V tokens are **deterministic functions of frame
   index** — there is no learned content for the cross-attention to bind
   to beyond positional re-injection. This is the textbook condition under
   which a cross-attention layer collapses into a more expensive form of
   the bias path.

3. Per-block parameter accounting at hidden width `C`:
   - temporal self-attn ≈ `4 C²`
   - temporal cross-attn ≈ `4 C²` ← removed
   - temporal FFN ≈ `0.69 C²` (rank=C/4, already low-rank)
   - Total before: `~8.69 C²`; after: `~4.69 C²` → **46% temporal-adapter
     reduction at constant placement**.

4. Failure-mode prediction: if the validation FVD/temporal coherence drops
   noticeably, the cross-attn was carrying real positional information that
   self-attn could not recover from add-to-text alone. In that case the
   correct response is to *upgrade* the frame token producer
   (make `frame_position_encoder` trainable, or move to learnable tokens)
   rather than restore cross-attn.

#### Implementation plan

This experiment is **config-only**. The infrastructure already exposes the
toggle (`VideoAttentionAdapterConfig.use_temporal_cross_attention`,
`framegen/video_attention.py:42`).

1. New config:
   `configs/train/image_first_smooth_snr_renoise_boundary_xnocross.yaml`,
   identical to `image_first_smooth_snr_renoise_boundary.yaml` except:

   ```yaml
   video_adapters:
     attention:
       use_temporal_cross_attention: false
   training:
     output_dir: "outputs/image-first-smooth-snr-renoise-boundary-xnocross"
   logging:
     run_name: "image-first-smooth-snr-renoise-boundary-xnocross"
     tags: [..., "ablation-no-cross-attn"]
   ```

2. New launcher
   `scripts/train_image_first_smooth_snr_renoise_boundary_xnocross.sh`
   (mirror of the existing smooth-SNR launcher with the new `TRAIN_CONFIG`).

3. Training run: same data, seed (`42`), optimizer, LR schedule, batch,
   precision, validation cadence, max_train_steps (`15000`) as the
   baseline. Run on the same 4 GPU configuration
   (`configs/accelerate/default.yaml`).

4. Optional but recommended: log per-step trainable param count and VRAM
   peak through the existing `wandb` metric path so the parameter saving
   is recorded next to loss curves.

#### Success / falsification criteria

- **Primary metric** (training): `train/loss` curve on the same data should
  stay within the noise band of the baseline run after the first 500 steps.
- **Secondary metric** (validation): CFG=8 validation samples at
  `t1 ∈ {0, 0.25, 0.5, 0.75}` over the same prompt
  (`"A dog running through a grassy field, cinematic lighting"`) should not
  degrade qualitatively. We will eyeball-compare against the baseline run's
  saved videos at the same global_step.
- **Tertiary metric** (compute): VRAM peak should drop ~3–5% (cross-attn is
  not the biggest VRAM consumer because activations dominate, but
  parameter count drops 46% of the temporal adapter).

If primary and secondary criteria pass at step 10k, **E1 succeeds** and we
move to E2 with `use_temporal_cross_attention: false` baked in.

If they fail, fall back to **either** (a) restore cross-attn and switch
A3.2 (mid+up restriction) first, or (b) make the frame position encoder
trainable and re-test.

#### Estimated cost

15k steps at the existing settings on 4 GPUs. ~1.5 days at the current rate.

---

### Experiment E2 — "Persistent anchor conditioning v1" (A1 + A3.2)

Run only after E1 results are in. E2 starts from the E1 checkpoint
(`use_temporal_cross_attention: false`) and adds two changes simultaneously:
A1 v1 anchor branch + A3.2 mid+up placement restriction.

#### Hypothesis

> Adding gated, spatial-aligned anchor K/V tokens to temporal cross-attn in
> the mid + up SDXL blocks improves first-frame identity preservation and
> stage-1 → stage-2 handoff quality, while the mid+up restriction prevents
> the anchor branch from undoing A3's compression. Specifically:
>
> - validation video's frame-0 should be visibly closer to the stage-1 image
>   than the E1 baseline,
> - and the trainable parameter count after both changes should drop further
>   to **~0.4 B** (vs ~0.55 B post-E1 and ~1.0 B baseline).

#### Theoretical basis

(See `§2 A1` and `§2 A3` of this document for the full discussion. Summary:)

1. The image-first contract is "SDXL defines scene identity; video adapters
   add motion residual." Currently this contract is only enforced at the
   **noising source** through the smooth_snr blend. The video adapters in
   stage 2 have **no persistent reference** to the stage-1 image after the
   re-noise step; they could regenerate frame 0 from any prior.

2. Spatial-aligned anchor K/V tokens give the temporal cross-attn at every
   mid+up block a direct, deterministic reference to the stage-1 anchor at
   the corresponding spatial site. The temporal cross-attn already attends
   along the frame axis at `[B*N, F, C]`; adding anchor tokens to the K/V
   set is "site `(h, w)` of frame `f` whispers to site `(h, w)` of the
   anchor."

3. Gated zero-init is required for two reasons:
   - The current temporal cross-attn has softmax-normalised K/V over
     `frame_attention_tokens`. Naive K/V append would change the softmax
     mass even at zero values. A separate gated anchor-attn branch
     (`output = temporal_cross + gate · anchor_attn(...)`) starting at
     `gate = 0` gives **exact checkpoint identity** at step 0 so a
     continuation fine-tune does not regress.
   - The projector that produces anchor K/V can be normally initialised so
     `gate` receives meaningful gradients from step 1.

4. Mid + up restriction is justified by A3's parameter accounting (mid+up
   carries the majority of `C=1280` motion-relevant blocks; down blocks
   are higher-resolution local detail and less motion-critical). Combined
   with A1's anchor branch placed at the same blocks, total trainable
   stays in the same A3.2 budget envelope.

#### Implementation plan

##### B-1. New config

`configs/train/image_first_smooth_snr_renoise_boundary_anchor.yaml`:

```yaml
video_adapters:
  attention:
    use_temporal_cross_attention: false     # inherit from E1
    use_temporal_self_attention: true
    use_temporal_ffn: true
    placement: "mid_up"                     # NEW — defaults to "all"
    anchor_conditioning:                    # NEW group
      enabled: true
      mode: "spatial"                       # spatial | global (v1 = spatial)
      projector_hidden: 128                 # mid-channel of the anchor projector
      gate_init: 0.0                        # zero-init scalar gate per block
      gate_per_channel: false               # v1: scalar per block
training:
  output_dir: "outputs/image-first-smooth-snr-renoise-boundary-anchor"
  resume_from_checkpoint:
    "outputs/image-first-smooth-snr-renoise-boundary-xnocross/checkpoint-last"
  max_train_steps: 18000                    # 3000 extra steps from E1's 15k
logging:
  run_name: "image-first-smooth-snr-renoise-boundary-anchor"
  tags: [..., "anchor-conditioning-v1", "placement-mid-up"]
```

##### B-2. Code changes — `framegen/video_attention.py`

Add an optional anchor-attn sub-branch inside `VideoBasicTransformerBlock`:

```python
class VideoBasicTransformerBlock(nn.Module):
    def __init__(self, ..., use_anchor_conditioning: bool = False,
                 anchor_projector_hidden: int = 128,
                 anchor_gate_init: float = 0.0,
                 anchor_gate_per_channel: bool = False):
        ...
        if use_anchor_conditioning:
            # Anchor K/V projector: from anchor latent feature -> attn dim
            self.anchor_proj_in = nn.Conv2d(
                latent_channels, anchor_projector_hidden, kernel_size=1,
            )
            self.anchor_proj_out = nn.Linear(anchor_projector_hidden, dim)
            self.anchor_attn = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=heads, dim_head=dim_head,
                bias=False,
            )
            _zero_attention_output(self.anchor_attn)  # follow existing pattern
            self.anchor_norm = nn.LayerNorm(dim)
            if anchor_gate_per_channel:
                self.anchor_gate = nn.Parameter(
                    torch.full((dim,), float(anchor_gate_init))
                )
            else:
                self.anchor_gate = nn.Parameter(
                    torch.tensor(float(anchor_gate_init))
                )
```

Forward path: after the existing temporal cross-attn output (or in its
place, when cross-attn is disabled), the per-frame temporal stream
`[B*N, F, C]` attends with Q=stream, K/V=spatial-aligned anchor tokens
projected from `anchor_latent_feature[:, :, n, :]`. Output multiplied by
`anchor_gate` and added to the existing temporal cross output (or to the
self-attn output if cross-attn is disabled).

##### B-3. Placement restriction

Add `block_filter` parameter to the injection entry point (around
`framegen/video_attention.py:480`). Resolve based on the top-level SDXL
UNet block hierarchy (`down_blocks.*`, `mid_block`, `up_blocks.*`). The
config flag `placement: "mid_up"` produces a filter that admits only
descendants of `mid_block` and `up_blocks.*`. Default `"all"` preserves
current behaviour.

The same `block_filter` is reused for `VideoResnetBlock2D` injection in
`framegen/video_resnet.py` if the user opts in symmetrically.

##### B-4. Anchor latent source

Training (`train.py` around the existing image-first branch ~line 1930):

```python
# anchor_clean = clean first frame latent z*_1, broadcast across F
anchor_clean = clean_latents.view(B, F, *clean_latents.shape[1:])[:, 0]
# Pass anchor_clean into set_video_attention_context(...)
```

Inference (`framegen/image_first_generation.py` around the switch ~line
390): use the `pred_x0` already computed for `pred_x0_renoise`:

```python
anchor_clean = anchor_clean_latents  # pred_x0 of stage-1 image latent
set_video_attention_context(pipe.unet, ..., anchor_features=anchor_clean)
```

Multi-resolution downsample: cache projected anchor feature once per
UNet resolution. The projector is shared across blocks at the same
resolution to keep parameters small.

##### B-5. Parameter budget check

With `placement: "mid_up"` and the anchor branch added only at mid+up
transformer blocks (~46 of the 70 SDXL transformer blocks):

| Component | per-block | block count | sub-total |
|-----------|-----------|-------------|-----------|
| temporal self-attn (existing) | `4 C²` | 46 | core |
| temporal cross-attn (disabled in E1) | 0 | — | — |
| temporal FFN (existing) | `0.69 C²` | 46 | small |
| anchor attn (new) | `4 C²` × gate=0 init | 46 | grows during training |
| anchor projector (shared by resolution) | small | ~3 | small |

Even if the anchor attn ends up at full `4 C²` per block, the mid+up
restriction keeps the **net trainable below 0.5 B** — a clean improvement
over the 1.0–1.1 B baseline and a small overhead on top of E1's ~0.55 B.

##### B-6. Continuation fine-tune

- Resume from E1's checkpoint with the new anchor parameters initialised
  fresh (zero gate). Existing temporal self-attn / FFN weights load
  unchanged; the new anchor projector + anchor attn weights initialise
  from scratch.
- LR for the new parameters: same as the base LR for the first 2000 steps,
  then merge into the standard schedule for the remaining 1000.
- Validation: same `t1 ∈ {0, 0.25, 0.5, 0.75}`, same CFG list.

#### Success / falsification criteria

- **Primary metric** (qualitative + quantitative): frame-0 similarity to
  the stage-1 image. Compute LPIPS between `validation/t1_0p25/cfg_8/frame_000.png`
  and the corresponding stage-1 image latent decoded separately. Expected:
  E2 stage-1↔frame-0 LPIPS is **≥ 30% lower** than E1.
- **Secondary metric**: training loss should not increase (the anchor branch
  starts at gate=0 and only adds capacity; the existing path is preserved).
- **Tertiary metric** (parameter budget): trainable param count after E2
  should be reported as ≤ 0.50 B by the existing wandb metric path.

If primary criterion fails (frame-0 still drifts away from stage-1 image),
either (a) the gate didn't open during fine-tune (check `anchor_gate` value
trajectory and bump LR), or (b) spatial anchor tokens are insufficient and
we need to add global pooled anchor tokens too. Either is a v2 follow-up.

#### Estimated cost

3k continuation steps on 4 GPUs ≈ 6–8 hours from E1's checkpoint, vs a
full retrain.

---

### Experiments **NOT** queued this cycle (rationale)

- **A2 VAE diagnostic** — runs in parallel with E1 as a 1–2 day standalone
  analysis script, not as a training run. Output is a single number
  (flow-warped excess error) plus a 20-clip contact sheet. Decision gate
  for whether to invest further.
- **U2 calibrator redesign** — implementation is ready in spec (see §3 U2),
  but it must be ablated against E2's anchor branch as a 2×2 study, not
  shipped first. After E2 closes successfully, the calibrator redesign
  becomes E3.
- **U3 v2 rollout `pred_x0` source** — requires extracting `pred_x0` from
  the rollout endpoint (a new function in `train.py` adjacent to
  `rollout_image_first_anchor_latents`). Defer until E2 settles, then run
  as E4.
- **A3.3 self-attn weight sharing** — risky compression; ship only if E1
  passed but post-E2 trainable count is still considered too high.

---

## 6. Parallel-run plan (5 slots × 4 GPU = 20 GPU)

Decided 2026-05-29 after the user confirmed a 20-GPU budget at 4 GPUs/run.
E1 is currently in slot 1. The remaining 4 slots fill the U3 ablation
lattice with config-only runs that have never been trained before, plus
one missing 2×2 corner. None of the parallel slots require new code.

Already-trained reference checkpoints in `outputs/` (used as additional
data points; do NOT re-run):

- `image-first-smooth-snr-renoise-boundary` — the (B=yes, R=yes, cross-attn=on)
  baseline, up to step 14000. Serves as E1's reference.
- `image-first-snr` — hard-SNR gate with `repeat_add_noise` validation.
- `image-first-rollout`, `image-first-rollout-snr` — rollout-source variants.
- `image-first-snr-Ea` — SNR gate with the latent calibrator (snr_ea).

### Phase 1 (≈1.5 days wall-clock from 2026-05-29) — fill all 5 slots

| Slot | GPUs | Accelerate config | Launcher | Hypothesis isolated |
|------|------|-------------------|----------|---------------------|
| 1 (running) | 0–3 | `configs/accelerate/default.yaml` | `scripts/train_image_first_smooth_snr_renoise_boundary_xnocross.sh` | **A3.1**: cross-attn redundancy under `add_to_text` |
| 2 | 4–7 | `configs/accelerate/gpus_4_7.yaml` | `scripts/train_image_first_smooth_snr_boundary.sh` | **U3.a**: vs E1's `pred_x0_renoise` baseline — does pred_x0 re-noise help at inference? |
| 3 | 8–11 | `configs/accelerate/gpus_8_11.yaml` | `scripts/train_image_first_smooth_snr.sh` | **U3.b**: vs slot 2 — does the boundary loss help? |
| 4 | 12–15 | `configs/accelerate/gpus_12_15.yaml` | `scripts/train_image_first_snr_renoise.sh` | **U3.c**: vs trained baseline — smooth vs hard SNR gate |
| 5 | 16–19 | `configs/accelerate/gpus_16_19.yaml` | `scripts/train_image_first_smooth_snr_renoise.sh` (NEW, B=no R=yes corner) | **U3.d**: completes 2×2 over {boundary, pred_x0_renoise} |

Launch syntax (each in its own terminal / tmux pane):

```bash
ACCELERATE_CONFIG=configs/accelerate/gpus_4_7.yaml \
  bash scripts/train_image_first_smooth_snr_boundary.sh

ACCELERATE_CONFIG=configs/accelerate/gpus_8_11.yaml \
  bash scripts/train_image_first_smooth_snr.sh

ACCELERATE_CONFIG=configs/accelerate/gpus_12_15.yaml \
  bash scripts/train_image_first_snr_renoise.sh

ACCELERATE_CONFIG=configs/accelerate/gpus_16_19.yaml \
  bash scripts/train_image_first_smooth_snr_renoise.sh
```

Each `gpus_*.yaml` pins:
- `gpu_ids` to its 4-GPU group
- `main_process_port` to a unique value (29500/29501/29502/29503/29504)
  to avoid TCPStore collisions.

### Reading the lattice once phase 1 finishes

Across the 5 fresh runs + the 14k-step baseline, the following questions
are now answered jointly:

1. **A3.1** = (E1 vs baseline) — Δquality at trainable ≈ 0.54B vs 1.14B.
2. **U3 boundary loss** = (smooth_snr_boundary vs smooth_snr) and (baseline vs smooth_snr_renoise) — does boundary loss help across both switch modes?
3. **U3 pred_x0_renoise** = (smooth_snr_boundary vs baseline) and (smooth_snr vs smooth_snr_renoise) — does re-noise help across both boundary settings? Two-way cross-check.
4. **U3 smooth-vs-hard gate** = (snr_renoise vs baseline) — was the smooth gate worth the extra design surface?
5. **First-stage rollout** — already covered by the trained `image-first-rollout-snr` checkpoint vs baseline; reused.

### Phase 2 (when slot 1 frees, ≈end of day +1.5)

- **Slot 1 → E2**. Implement the anchor branch + mid+up placement filter
  (§5 E2 B-2/B-3/B-4) during phase 1 and start E2 from
  `image-first-smooth-snr-renoise-boundary-xnocross/checkpoint-last`
  as soon as E1 completes.
- **Slots 2–5**: as each phase-1 run completes, repurpose the slot for an
  experiment that depends on phase-1 results:
  - E3 = calibrator `gate_scaled` (§3 U2 implementation, requires code).
  - Boundary-weight sensitivity sweep (config-only; `image_first_boundary_loss_weight`
    ∈ {0.02, 0.05, 0.10}).
  - Smooth-SNR gate window sensitivity (`snr_full / snr_zero` variations)
    once the principled-calibration data analysis from §3 U3 is done.

Files generated for §6 (2026-05-29):
- `configs/accelerate/gpus_{4_7, 8_11, 12_15, 16_19}.yaml` — per-group launcher configs.
- `configs/train/image_first_smooth_snr_renoise.yaml` — the missing 2×2 corner.
- `scripts/train_image_first_smooth_snr_renoise.sh` — its launcher.

## 7. Update Protocol

When a new round of discussion happens or an earlier conclusion is revised:

1. Append a **new dated subsection** under the relevant agenda
   (e.g. `### A1 update — 2026-06-XX`).
2. Reference the prior decision and state **what changed** and **why**.
3. Update the agenda status table in §1 if scope opens/closes.
4. Update the roadmap in §4 if priorities or files shift.
5. Update the experiment queue in §5 if a new experiment is added or an
   existing one is reordered/cancelled.
6. Update the parallel-run plan in §6 if the slot assignment changes (e.g.,
   a slot completes and is repurposed, or a new run is added).
7. Cross-link related agendas with `[[A1]]` / `[[U3]]` style references so
   future updates stay grounded.

Do **not** delete prior content — keep the history readable. Strikethrough
or "superseded by" notes are preferred over rewrites.
