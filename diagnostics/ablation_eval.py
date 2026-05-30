#!/usr/bin/env python
"""Ablation evaluation harness (Minimum Viable Eval).

Evaluates trained image-first checkpoints against each other on a fixed prompt
set with paired seeds, computes quantitative video metrics, applies a paired
bootstrap decision rule, and logs everything to a SINGLE wandb run so all
models are compared side by side.

Design + rationale: information/claude-codex-discussion.md §8 (feasibility) and
information/13_ablation_eval.md.

Models are read from `outputs/<name>/checkpoint-<step>` (default step = "last").
Each model is generated with the operating point fixed (CFG, t1) for
comparability, but with its OWN inference switch settings from its config.yaml
(switch_mode / renoise mode are part of what the ablation tests).

Metrics (per generated clip):
  clip_t          frame<->text CLIP cosine (open_clip; NaN if not installed)
  clip_anchor     stage-1 anchor <-> frame0 CLIP image cosine
  vgg_anchor      VGG feature L1, anchor <-> frame0  (lower better)
  pix_anchor_low  64x64 L1, anchor <-> frame0        (lower better)
  warp_vgg        RAFT flow-warped masked VGG error over adjacent frames (lower better)
  motion_mag      median RAFT flow magnitude
  motion_cov      fraction of pixels with flow > 2px (static-video guardrail)

Decision rule (paired vs a baseline model): per-prompt delta at identical seeds
-> 20% trimmed mean across prompts -> 10k-bootstrap 95% CI. With enough baseline
seeds, sigma_null is estimated from baseline-vs-baseline splits and an ablation
is "different" iff CI excludes 0 AND |delta| > 2*sigma_null.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.config import get_torch_dtype, load_config
from framegen.env import apply_hf_env_aliases, load_env_file
from framegen.image_first_generation import generate_image_first_video_frames
from framegen.temporal import normalize_frame_token_mode
from infer import resolve_checkpoint_dir, resolve_config_path
from infer_image_first import load_image_first_pipeline


# ---------------------------------------------------------------------------
# Metric backbones
# ---------------------------------------------------------------------------

_VGG_INDICES = (8, 15, 22)  # relu2_2, relu3_3, relu4_3


class VGGFeatures(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        slices = []
        prev = 0
        for idx in _VGG_INDICES:
            slices.append(torch.nn.Sequential(*list(backbone[prev : idx + 1])))
            prev = idx + 1
        self.slices = torch.nn.ModuleList(slices)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval().to(device)

    @torch.no_grad()
    def feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self._mean) / self._std
        out = []
        for s in self.slices:
            x = s(x)
            out.append(x)
        return out

    @torch.no_grad()
    def l1(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        total = 0.0
        fa, fb = self.feats(a), self.feats(b)
        for x, y in zip(fa, fb):
            d = (x - y).abs().mean(dim=1, keepdim=True)
            if mask is None:
                total += float(d.mean())
            else:
                m = F.adaptive_avg_pool2d(mask, output_size=d.shape[-2:])
                total += float((d * m).sum() / m.sum().clamp_min(1e-8))
        return total / len(fa)


def load_raft(device: torch.device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.transforms()


@torch.no_grad()
def raft_flow(raft, transforms, src, dst):
    s, d = transforms(src, dst)
    return raft(s, d)[-1]


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base = torch.stack([xs, ys], 0).unsqueeze(0).expand(b, -1, -1, -1)
    samp = base + flow
    sx = 2.0 * samp[:, 0] / max(w - 1, 1) - 1.0
    sy = 2.0 * samp[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([sx, sy], -1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def occlusion_mask(fwd, bwd, threshold: float = 1.0) -> torch.Tensor:
    warped = warp(bwd, fwd)
    disc = (fwd + warped).pow(2).sum(1, keepdim=True).clamp_min(0).sqrt()
    return (disc < threshold).float()


class CLIPScorer:
    """OpenCLIP text/image scorer. Disabled (returns NaN) if open_clip missing."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.ok = False
        try:
            import open_clip

            self.model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model = self.model.eval().to(device)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.register_buffers()
            self.ok = True
        except Exception as exc:  # pragma: no cover - optional dep
            print(f"[eval] open_clip unavailable ({exc.__class__.__name__}); clip_* metrics = NaN. "
                  f"Install with: pip install open_clip_torch")

    def register_buffers(self):
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _embed_images(self, imgs01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(imgs01, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        f = self.model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_score(self, frames01: torch.Tensor, prompt: str) -> float:
        if not self.ok:
            return float("nan")
        idx = [0, 2, 4, 6] if frames01.shape[0] >= 7 else list(range(frames01.shape[0]))
        img = self._embed_images(frames01[idx].to(self.device))
        tok = self.tokenizer([prompt]).to(self.device)
        t = self.model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
        return float((img @ t.T).mean())

    @torch.no_grad()
    def image_score(self, a01: torch.Tensor, b01: torch.Tensor) -> float:
        if not self.ok:
            return float("nan")
        fa = self._embed_images(a01.to(self.device))
        fb = self._embed_images(b01.to(self.device))
        return float((fa * fb).sum(-1).mean())


# ---------------------------------------------------------------------------
# Per-clip metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def clip_metrics_for(
    frames01: torch.Tensor,       # [F,3,H,W] in [0,1] on device
    anchor01: torch.Tensor | None,  # [1,3,H,W] in [0,1] on device or None
    prompt: str,
    vgg: VGGFeatures,
    raft,
    raft_transforms,
    clip: CLIPScorer,
    occlusion_threshold: float,
) -> dict:
    F_ = frames01.shape[0]
    m: dict[str, float] = {}
    m["clip_t"] = clip.text_score(frames01, prompt)

    frame0 = frames01[:1]
    if anchor01 is not None:
        m["clip_anchor"] = clip.image_score(anchor01, frame0)
        m["vgg_anchor"] = vgg.l1(anchor01, frame0)
        a_low = F.avg_pool2d(anchor01, 8)
        f_low = F.avg_pool2d(frame0, 8)
        m["pix_anchor_low"] = float((a_low - f_low).abs().mean())
    else:
        m["clip_anchor"] = float("nan")
        m["vgg_anchor"] = float("nan")
        m["pix_anchor_low"] = float("nan")

    # temporal: RAFT over adjacent frames
    if F_ >= 2:
        src, dst = frames01[:-1], frames01[1:]
        fwd = raft_flow(raft, raft_transforms, src, dst)
        bwd = raft_flow(raft, raft_transforms, dst, src)
        valid = occlusion_mask(fwd, bwd, occlusion_threshold)
        dst_warp = warp(src, fwd)
        m["warp_vgg"] = vgg.l1(dst, dst_warp, valid)
        mag = fwd.pow(2).sum(1, keepdim=True).clamp_min(0).sqrt()
        m["motion_mag"] = float(mag.median())
        m["motion_cov"] = float((mag > 2.0).float().mean())
    else:
        m["warp_vgg"] = float("nan")
        m["motion_mag"] = float("nan")
        m["motion_cov"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _load_frames(output_dir: Path, num_frames: int, device, dtype) -> torch.Tensor:
    frames = []
    for i in range(num_frames):
        p = output_dir / f"frame_{i:03d}.png"
        img = Image.open(p).convert("RGB")
        t = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(t)
    return torch.stack(frames, 0).to(device=device, dtype=dtype)


def _load_image(path: Path, device, dtype) -> torch.Tensor | None:
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device=device, dtype=dtype)


def _to_array(img):
    import numpy as np

    return np.asarray(img)


def generate_one(
    pipe_bundle,
    model_config,
    temporal_config,
    frame_encoder_config,
    latent_calibrator_config,
    inf,
    prompt: str,
    seed: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    pipe, temporal_mlp, frame_position_encoder, latent_calibrator = pipe_bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = out_dir / "anchor.png"
    generate_image_first_video_frames(
        pipe=pipe,
        temporal_mlp=temporal_mlp,
        frame_position_encoder=frame_position_encoder,
        latent_calibrator=latent_calibrator,
        latent_calibrator_config=latent_calibrator_config,
        prompt=prompt,
        num_frames=inf["num_frames"],
        output_dir=out_dir,
        resolution=int(model_config["resolution"]),
        temporal_alpha=float(temporal_config.get("alpha", 1.0)),
        t1_ratio=inf["t1"],
        guidance_scale=inf["cfg"],
        injection_mode=temporal_config.get("injection_mode", "add_to_pooled_prompt_embeds"),
        frame_token_embedding_mode=frame_encoder_config["token_embedding_mode"],
        frame_token_alpha=float(frame_encoder_config.get("alpha", 1.0)),
        num_inference_steps=inf["num_inference_steps"],
        seed=seed,
        save_grid=False,
        save_video=True,
        fps=inf["fps"],
        switch_noise_scale=inf["switch_noise_scale"],
        switch_mode=inf["switch_mode"],
        renoise_noise_mode=inf["renoise_noise_mode"],
        renoise_noise_scale=inf["renoise_noise_scale"],
        anchor_image_path=anchor_path,
    )
    return out_dir / "video.mp4", anchor_path


def build_model(name: str, step: str, device, project_root: Path):
    args = SimpleNamespace(
        checkpoint=None, name=name, step=step, config=None,
        disable_video_resnet_adapters=False,
        disable_vae_decoder_adapters=False,
        disable_video_attention_adapters=False,
    )
    ckpt = resolve_checkpoint_dir(args, project_root)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found for model {name!r}: {ckpt}")
    cfg = load_config(resolve_config_path(args, ckpt, project_root))
    model_config = cfg["model"]
    torch_dtype = get_torch_dtype(model_config.get("dtype", "bf16"))
    vae_dtype = get_torch_dtype(model_config.get("vae_dtype", model_config.get("dtype", "bf16")))
    (
        pipe, temporal_mlp, temporal_config, frame_position_encoder,
        frame_encoder_config, latent_calibrator, latent_calibrator_config,
    ) = load_image_first_pipeline(
        args=args, config=cfg, checkpoint_dir=ckpt,
        torch_dtype=torch_dtype, vae_dtype=vae_dtype, device=device,
    )
    frame_encoder_config.setdefault("token_embedding_mode", "add_to_text")
    frame_encoder_config["token_embedding_mode"] = normalize_frame_token_mode(
        frame_encoder_config["token_embedding_mode"]
    )
    val = cfg.get("validation", {})
    inf = dict(
        switch_mode=val.get("image_first_switch_mode", "repeat_add_noise"),
        renoise_noise_mode=val.get("image_first_renoise_noise_mode", "independent"),
        renoise_noise_scale=float(val.get("image_first_renoise_noise_scale", 1.0)),
        switch_noise_scale=float(val.get("switch_noise_scale", 0.1)),
    )
    return dict(
        pipe_bundle=(pipe, temporal_mlp, frame_position_encoder, latent_calibrator),
        model_config=model_config, temporal_config=temporal_config,
        frame_encoder_config=frame_encoder_config,
        latent_calibrator_config=latent_calibrator_config,
        inf=inf, checkpoint=str(ckpt),
    )


def free_model(bundle):
    pipe = bundle["pipe_bundle"][0]
    del bundle["pipe_bundle"]
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def trimmed_mean(x, trim: float = 0.2) -> float:
    xs = sorted(x)
    n = len(xs)
    k = int(math.floor(n * trim))
    core = xs[k : n - k] if n - 2 * k > 0 else xs
    return sum(core) / len(core)


def bootstrap_ci(deltas, n_boot: int, trim: float, seed: int = 0):
    if not deltas:
        return (float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    n = len(deltas)
    t = torch.tensor(deltas, dtype=torch.float64)
    stats = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        stats.append(trimmed_mean(t[idx].tolist(), trim))
    s = torch.tensor(stats)
    return (float(s.quantile(0.025)), float(s.quantile(0.975)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_PROMPTS = [
    "A dog running through a grassy field, cinematic lighting",
    "A red sports car driving down a coastal highway at sunset",
    "Aerial drone shot flying over a snowy mountain range",
    "A chef chopping vegetables on a wooden board, close up",
    "Ocean waves crashing on a rocky shore, slow motion",
    "A hummingbird hovering near a bright pink flower",
    "Time-lapse of clouds moving over a city skyline at dusk",
    "A person walking through a busy neon-lit street at night",
]

LOWER_BETTER = {"vgg_anchor", "pix_anchor_low", "warp_vgg"}
METRIC_KEYS = [
    "clip_t", "clip_anchor", "vgg_anchor", "pix_anchor_low",
    "warp_vgg", "motion_mag", "motion_cov",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", required=True,
                   help="Comma-separated model names under outputs/ (first = baseline unless --baseline).")
    p.add_argument("--baseline", default=None, help="Baseline model name for paired deltas (default: first).")
    p.add_argument("--step", default="last", help="Checkpoint step or 'last' (default).")
    p.add_argument("--prompts_file", default=None, help="Text file, one prompt per line. Default: built-in 8.")
    p.add_argument("--num_prompts", type=int, default=None, help="Cap number of prompts.")
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds for every model.")
    p.add_argument("--baseline_extra_seeds", default="3,4,5",
                   help="Extra baseline-only seeds to estimate run-to-run noise (sigma_null). Empty to skip.")
    p.add_argument("--cfg", type=float, default=8.0)
    p.add_argument("--t1", type=float, default=0.5)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--occlusion_threshold", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--trim", type=float, default=0.2)
    p.add_argument("--num_log_videos", type=int, default=4,
                   help="Prompts (x all models) to log as comparable videos in wandb.")
    p.add_argument("--output_dir", default=None, help="Default: outputs/eval/<timestamp>.")
    p.add_argument("--from_per_clip", default=None,
                   help="Skip generation; re-aggregate + re-log from an existing per_clip.json "
                        "(no GPU/model load needed).")
    p.add_argument("--wandb", action="store_true", help="Log everything to one wandb run.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--env_file", default=".env")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    apply_hf_env_aliases()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_dtype = torch.float32

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    baseline = args.baseline or models[0]
    if baseline not in models:
        models = [baseline] + models
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    extra = [int(s) for s in args.baseline_extra_seeds.split(",") if s.strip() != ""]

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [
            l.strip() for l in Path(args.prompts_file).read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "eval" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[eval] models={models} baseline={baseline} prompts={len(prompts)} seeds={seeds} "
          f"extra_baseline_seeds={extra} cfg={args.cfg} t1={args.t1} -> {out_root}")

    # Re-aggregation mode: load existing rows, skip all GPU/model work.
    reaggregate = args.from_per_clip is not None
    clip_ok = False
    if reaggregate:
        rows = json.loads(Path(args.from_per_clip).read_text())
        models = list(dict.fromkeys([r["model"] for r in rows]))
        if baseline not in models:
            baseline = models[0]
        print(f"[eval] re-aggregating {len(rows)} rows from {args.from_per_clip} (no generation)")
        vgg = raft = raft_transforms = clip = None
    else:
        print("[eval] loading metric backbones ...")
        vgg = VGGFeatures(device)
        raft, raft_transforms = load_raft(device)
        clip = CLIPScorer(device)
        clip_ok = clip.ok

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=__import__("os").environ.get("WANDB_PROJECT", "sdxl-frame-generator"),
            name=args.wandb_run_name or f"ablation-eval-{ts}",
            job_type="ablation-eval",
            config=dict(models=models, baseline=baseline, seeds=seeds,
                        extra_baseline_seeds=extra, cfg=args.cfg, t1=args.t1,
                        num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
                        prompts=prompts, clip_available=clip_ok, reaggregated=reaggregate),
        )

    # rows: list of dicts {model, prompt_idx, prompt, seed, **metrics}
    video_log: dict[tuple[int, str], Path] = {}  # (prompt_idx, model) -> mp4
    if not reaggregate:
        rows = []
        for model_name in models:
            model_seeds = list(seeds)
            if model_name == baseline:
                model_seeds = list(seeds) + [s for s in extra if s not in seeds]
            print(f"\n[eval] === model {model_name} (seeds {model_seeds}) ===")
            bundle = build_model(model_name, args.step, device, PROJECT_ROOT)
            inf = dict(bundle["inf"])
            inf.update(num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
                       fps=args.fps, t1=args.t1, cfg=args.cfg)
            print(f"[eval]   switch_mode={inf['switch_mode']} renoise={inf['renoise_noise_mode']} "
                  f"ckpt={bundle['checkpoint']}")
            for pi, prompt in enumerate(prompts):
                for seed in model_seeds:
                    gdir = out_root / model_name / f"p{pi:03d}_s{seed}"
                    try:
                        mp4, anchor_path = generate_one(
                            bundle["pipe_bundle"], bundle["model_config"], bundle["temporal_config"],
                            bundle["frame_encoder_config"], bundle["latent_calibrator_config"],
                            inf, prompt, seed, gdir,
                        )
                    except Exception as exc:
                        print(f"[eval]   FAILED {model_name} p{pi} s{seed}: {exc}")
                        continue
                    frames = _load_frames(gdir, inf["num_frames"], device, metric_dtype)
                    anchor = _load_image(anchor_path, device, metric_dtype)
                    met = clip_metrics_for(frames, anchor, prompt, vgg, raft, raft_transforms,
                                           clip, args.occlusion_threshold)
                    rows.append(dict(model=model_name, prompt_idx=pi, prompt=prompt, seed=seed, **met))
                    if pi < args.num_log_videos and seed == model_seeds[0]:
                        video_log[(pi, model_name)] = mp4
            free_model(bundle)

        # ---- persist raw rows ----
        (out_root / "per_clip.json").write_text(json.dumps(rows, indent=2))
        print(f"\n[eval] wrote {len(rows)} per-clip rows -> {out_root/'per_clip.json'}")

    # ---- aggregate per model ----
    def model_seed_prompt_mean(model, metric):
        # average over seeds within each prompt -> dict prompt_idx -> value
        per = {}
        for r in rows:
            if r["model"] == model and not math.isnan(r[metric]):
                per.setdefault(r["prompt_idx"], []).append(r[metric])
        return {pi: sum(v) / len(v) for pi, v in per.items()}

    summary = {}
    for model in models:
        summary[model] = {}
        for metric in METRIC_KEYS:
            pm = model_seed_prompt_mean(model, metric)
            vals = list(pm.values())
            summary[model][metric] = trimmed_mean(vals, args.trim) if vals else float("nan")

    # ---- sigma_null from baseline extra seeds (split-half) ----
    # Derive baseline seeds from the actual rows (robust to --from_per_clip).
    sigma_null = {}
    base_all_seeds = sorted({r["seed"] for r in rows if r["model"] == baseline})
    if len(base_all_seeds) >= 4:
        half = len(base_all_seeds) // 2
        gA, gB = base_all_seeds[:half], base_all_seeds[half:]
        for metric in METRIC_KEYS:
            # per-prompt: mean over gA - mean over gB
            byp = {}
            for r in rows:
                if r["model"] == baseline and not math.isnan(r[metric]):
                    byp.setdefault(r["prompt_idx"], {}).setdefault("A" if r["seed"] in gA else "B", []).append(r[metric])
            deltas = []
            for pi, d in byp.items():
                if "A" in d and "B" in d:
                    deltas.append(sum(d["A"]) / len(d["A"]) - sum(d["B"]) / len(d["B"]))
            sigma_null[metric] = float(torch.tensor(deltas).std(unbiased=False)) if len(deltas) > 1 else float("nan")
    else:
        sigma_null = {k: float("nan") for k in METRIC_KEYS}

    # ---- paired deltas vs baseline + decision ----
    decisions = []
    base_pm = {metric: model_seed_prompt_mean(baseline, metric) for metric in METRIC_KEYS}
    for model in models:
        if model == baseline:
            continue
        for metric in METRIC_KEYS:
            mp = model_seed_prompt_mean(model, metric)
            shared = sorted(set(mp) & set(base_pm[metric]))
            deltas = [mp[pi] - base_pm[metric][pi] for pi in shared]
            if not deltas:
                continue
            delta = trimmed_mean(deltas, args.trim)
            lo, hi = bootstrap_ci(deltas, args.bootstrap, args.trim)
            sig = sigma_null.get(metric, float("nan"))
            ci_excludes_0 = (lo > 0) or (hi < 0)
            beats_noise = (not math.isnan(sig)) and abs(delta) > 2 * sig
            different = bool(ci_excludes_0 and (beats_noise or math.isnan(sig)))
            direction = "lower=better" if metric in LOWER_BETTER else "higher=better"
            decisions.append(dict(
                model=model, metric=metric, delta=delta, ci_low=lo, ci_high=hi,
                sigma_null=sig, ci_excludes_0=ci_excludes_0, beats_2sigma=beats_noise,
                different=different, direction=direction, n_prompts=len(deltas),
            ))

    (out_root / "summary.json").write_text(json.dumps(
        dict(summary=summary, sigma_null=sigma_null, decisions=decisions,
             baseline=baseline, models=models), indent=2))

    # ---- console report ----
    print("\n[eval] ===== PER-MODEL (20% trimmed mean over prompts) =====")
    header = "model".ljust(46) + "".join(k.rjust(13) for k in METRIC_KEYS)
    print(header)
    for model in models:
        line = model.ljust(46) + "".join(f"{summary[model][k]:13.4f}" for k in METRIC_KEYS)
        print(line)
    print("\n[eval] ===== PAIRED DECISIONS vs baseline (%s) =====" % baseline)
    for d in decisions:
        flag = "DIFFERENT" if d["different"] else "ns"
        print(f"  {d['model'][:38]:38s} {d['metric']:14s} d={d['delta']:+.4f} "
              f"CI[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] σ0={d['sigma_null']:.4f} -> {flag} ({d['direction']})")

    # ---- wandb: one run, comparable ----
    if run is not None:
        import wandb

        per_clip_tbl = wandb.Table(columns=["model", "prompt_idx", "prompt", "seed", *METRIC_KEYS])
        for r in rows:
            per_clip_tbl.add_data(r["model"], r["prompt_idx"], r["prompt"], r["seed"],
                                  *[r[k] for k in METRIC_KEYS])
        summary_tbl = wandb.Table(columns=["model", *METRIC_KEYS])
        for model in models:
            summary_tbl.add_data(model, *[summary[model][k] for k in METRIC_KEYS])
        dec_tbl = wandb.Table(columns=["model", "metric", "delta", "ci_low", "ci_high",
                                       "sigma_null", "ci_excludes_0", "beats_2sigma", "different", "direction"])
        for d in decisions:
            dec_tbl.add_data(d["model"], d["metric"], d["delta"], d["ci_low"], d["ci_high"],
                             d["sigma_null"], d["ci_excludes_0"], d["beats_2sigma"], d["different"], d["direction"])

        # side-by-side sample videos: rows=prompt, columns=models
        sample_cols = ["prompt_idx", "prompt", *models]
        sample_tbl = wandb.Table(columns=sample_cols)
        for pi in range(min(args.num_log_videos, len(prompts))):
            cells = [pi, prompts[pi]]
            for model in models:
                mp4 = video_log.get((pi, model))
                cells.append(wandb.Video(str(mp4)) if mp4 and Path(mp4).exists() else None)
            sample_tbl.add_data(*cells)

        log = {
            "eval/per_clip": per_clip_tbl,
            "eval/per_model_summary": summary_tbl,
            "eval/decisions": dec_tbl,
            "eval/samples": sample_tbl,
        }
        # grouped summary scalars for bar charts
        for model in models:
            for k in METRIC_KEYS:
                v = summary[model][k]
                if not math.isnan(v):
                    log[f"summary/{k}/{model}"] = v
        run.log(log)
        run.summary["baseline"] = baseline
        run.summary["clip_available"] = clip_ok
        run.finish()
        print("[eval] logged to wandb run.")

    print(f"\n[eval] done. artifacts in {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
