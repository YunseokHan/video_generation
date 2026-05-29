#!/usr/bin/env python
"""A2 — VAE temporal coherence diagnostic.

Spec: ``information/12_vae_temporal_diagnostic.md`` and
``information/claude-codex-discussion.md`` §2 A2 + §5.

Measures whether the frozen SDXL VAE's frame-independent decode introduces
temporal flicker beyond what the underlying video already has, using
flow-warped consistency:

    excess = E(decode_{t+1}, warp(decode_t, flow_GT))
           - E(gt_{t+1},     warp(gt_t,     flow_GT))

with occlusion masks from forward-backward flow consistency. Two error
functionals E:

* ``charbonnier`` on high-pass RGB residuals — pixel-level shimmer signal.
* ``vgg_feat`` (LPIPS-style multi-layer VGG16 feature L2) — perceptual signal.

Verdict thresholds follow the agenda decision in §5:

* excess_ratio < 5%  → VAE is *not* the binding bottleneck; stop here.
* 5% ≤ excess_ratio < 10% → marginal; revisit after [[A1]].
* excess_ratio ≥ 10% → real flicker source; consider an R2/R4 training-side
  intervention or a temporal VAE decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.data import OpenVidVideoDataset
from framegen.env import (
    apply_hf_env_aliases,
    get_hf_cache_dir,
    get_hf_token,
    get_openvid_csv,
    get_openvid_root,
    load_env_file,
)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_vae(device: torch.device) -> torch.nn.Module:
    from diffusers import AutoencoderKL

    cache_dir = get_hf_cache_dir()
    token = get_hf_token()
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        token=token,
    )
    vae.eval().to(device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def load_raft(device: torch.device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.C_T_SKHT_V2
    raft = raft_large(weights=weights, progress=False).eval().to(device)
    for parameter in raft.parameters():
        parameter.requires_grad_(False)
    return raft, weights.transforms()


_VGG_INDICES = (3, 8, 15, 22, 29)  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3


class VGGFeatureDistance(torch.nn.Module):
    """LPIPS-style perceptual distance using channel-normalized VGG16 features.

    Not calibrated against humans like full LPIPS; install ``lpips`` if you
    need the calibrated version. The relative ordering of clips is consistent
    with LPIPS in practice.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        slices: list[torch.nn.Sequential] = []
        prev = 0
        for index in _VGG_INDICES:
            slices.append(torch.nn.Sequential(*list(backbone[prev : index + 1])))
            prev = index + 1
        self.slices = torch.nn.ModuleList(slices)
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval().to(device)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    @torch.no_grad()
    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self._normalize_input(a)
        b = self._normalize_input(b)
        total = a.new_zeros(())
        for slice_module in self.slices:
            a = slice_module(a)
            b = slice_module(b)
            diff = (
                _channel_unit_normalize(a) - _channel_unit_normalize(b)
            ).pow(2).sum(dim=1, keepdim=True)
            if mask is None:
                total = total + diff.mean()
            else:
                m = F.adaptive_avg_pool2d(mask, output_size=diff.shape[-2:])
                total = total + (diff * m).sum() / m.sum().clamp_min(1.0e-8)
        return total / len(self.slices)


def _channel_unit_normalize(x: torch.Tensor, eps: float = 1.0e-10) -> torch.Tensor:
    return x / x.pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()


# ---------------------------------------------------------------------------
# Flow + warp utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_flow(raft, transforms, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Flow that warps `src` toward `dst` (RAFT forward flow). Inputs in [0, 1]."""
    src_t, dst_t = transforms(src, dst)
    flow_pyramid = raft(src_t, dst_t)
    return flow_pyramid[-1]


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``image`` by ``flow`` so output[u, v] ≈ image[u+fx, v+fy]."""
    batch_size, _, height, width = image.shape
    device = image.device
    dtype = image.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xs, ys], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    sample = base + flow
    sx = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    sy = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([sx, sy], dim=-1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def occlusion_mask(
    fwd_flow: torch.Tensor, bwd_flow: torch.Tensor, threshold: float = 1.0
) -> torch.Tensor:
    """Forward-backward flow consistency mask (1 = valid, 0 = occluded)."""
    warped_bwd = warp_with_flow(bwd_flow, fwd_flow)
    discrepancy = (
        (fwd_flow + warped_bwd).pow(2).sum(dim=1, keepdim=True).clamp_min(0).sqrt()
    )
    return (discrepancy < threshold).to(fwd_flow.dtype)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.view(1, 1, size, size)


def highpass(image: torch.Tensor, sigma: float = 1.5, size: int = 9) -> torch.Tensor:
    kernel = gaussian_kernel(size, sigma, image.device, image.dtype)
    kernel = kernel.expand(image.shape[1], 1, size, size)
    padded = F.pad(image, (size // 2,) * 4, mode="reflect")
    blurred = F.conv2d(padded, kernel, groups=image.shape[1])
    return image - blurred


def charbonnier(
    residual: torch.Tensor, mask: torch.Tensor | None, eps: float = 1.0e-3
) -> torch.Tensor:
    per_pixel = (residual.pow(2) + eps * eps).sqrt().mean(dim=1, keepdim=True)
    if mask is None:
        return per_pixel.mean()
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0e-8)


# ---------------------------------------------------------------------------
# VAE round-trip
# ---------------------------------------------------------------------------


@torch.no_grad()
def vae_roundtrip(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode then decode each frame independently. Input/output in [-1, 1]."""
    latent_dist = vae.encode(frames).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    decoded = vae.decode(latent / vae.config.scaling_factor).sample
    return decoded.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    n_pairs: int
    gt_charbonnier: float
    decoded_charbonnier: float
    gt_vgg_feat: float
    decoded_vgg_feat: float
    valid_fraction: float

    @property
    def excess_charbonnier(self) -> float:
        return self.decoded_charbonnier - self.gt_charbonnier

    @property
    def excess_vgg_feat(self) -> float:
        return self.decoded_vgg_feat - self.gt_vgg_feat

    @property
    def excess_ratio_charbonnier(self) -> float:
        return self.excess_charbonnier / max(self.gt_charbonnier, 1.0e-8)

    @property
    def excess_ratio_vgg_feat(self) -> float:
        return self.excess_vgg_feat / max(self.gt_vgg_feat, 1.0e-8)


@torch.no_grad()
def evaluate_clip(
    frames_gt: torch.Tensor,
    frames_decoded: torch.Tensor,
    raft,
    raft_transforms,
    vgg: VGGFeatureDistance,
    occlusion_threshold: float,
    highpass_sigma: float,
) -> ClipResult:
    """Compute per-clip flow-warped consistency metrics. Frames in [-1, 1]."""
    gt = (frames_gt + 1.0) * 0.5
    dec = (frames_decoded + 1.0) * 0.5
    assert gt.shape[0] >= 2

    src_gt, dst_gt = gt[:-1], gt[1:]
    src_dec, dst_dec = dec[:-1], dec[1:]
    fwd = compute_flow(raft, raft_transforms, src_gt, dst_gt)
    bwd = compute_flow(raft, raft_transforms, dst_gt, src_gt)
    valid = occlusion_mask(fwd, bwd, threshold=occlusion_threshold)
    gt_warp = warp_with_flow(src_gt, fwd)
    dec_warp = warp_with_flow(src_dec, fwd)

    gt_resid_hp = highpass(dst_gt, sigma=highpass_sigma) - highpass(
        gt_warp, sigma=highpass_sigma
    )
    dec_resid_hp = highpass(dst_dec, sigma=highpass_sigma) - highpass(
        dec_warp, sigma=highpass_sigma
    )
    gt_charb = float(charbonnier(gt_resid_hp, valid))
    dec_charb = float(charbonnier(dec_resid_hp, valid))
    gt_vgg = float(vgg(dst_gt, gt_warp, valid))
    dec_vgg = float(vgg(dst_dec, dec_warp, valid))

    return ClipResult(
        n_pairs=int(src_gt.shape[0]),
        gt_charbonnier=gt_charb,
        decoded_charbonnier=dec_charb,
        gt_vgg_feat=gt_vgg,
        decoded_vgg_feat=dec_vgg,
        valid_fraction=float(valid.mean()),
    )


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------


def save_contact_sheet(
    frames_gt: torch.Tensor,
    frames_decoded: torch.Tensor,
    output_path: Path,
    title: str,
    diff_clip: float = 0.2,
) -> None:
    import matplotlib.pyplot as plt

    f = frames_gt.shape[0]
    fig, axes = plt.subplots(3, f, figsize=(2.0 * f, 6.0))
    if f == 1:
        axes = axes[:, None]
    for i in range(f):
        gt = ((frames_gt[i].float().cpu() + 1.0) * 0.5).clamp(0, 1)
        dec = ((frames_decoded[i].float().cpu() + 1.0) * 0.5).clamp(0, 1)
        diff = (gt - dec).abs().mean(dim=0)
        axes[0][i].imshow(gt.permute(1, 2, 0).numpy())
        axes[1][i].imshow(dec.permute(1, 2, 0).numpy())
        axes[2][i].imshow(diff.numpy(), cmap="hot", vmin=0.0, vmax=diff_clip)
        for row in range(3):
            axes[row][i].set_xticks([])
            axes[row][i].set_yticks([])
        axes[0][i].set_title(f"t={i}", fontsize=8)
    axes[0][0].set_ylabel("GT", fontsize=10)
    axes[1][0].set_ylabel("VAE decoded", fontsize=10)
    axes[2][0].set_ylabel(f"|diff| (clip {diff_clip:g})", fontsize=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[ClipResult]) -> dict:
    if not results:
        return {}
    fields = (
        "gt_charbonnier",
        "decoded_charbonnier",
        "gt_vgg_feat",
        "decoded_vgg_feat",
        "excess_charbonnier",
        "excess_vgg_feat",
        "excess_ratio_charbonnier",
        "excess_ratio_vgg_feat",
        "valid_fraction",
    )
    output: dict[str, float] = {}
    for field in fields:
        values = torch.tensor([getattr(r, field) for r in results], dtype=torch.float64)
        output[f"{field}/mean"] = float(values.mean())
        output[f"{field}/median"] = float(values.median())
        output[f"{field}/p25"] = float(values.quantile(0.25))
        output[f"{field}/p75"] = float(values.quantile(0.75))
        output[f"{field}/p90"] = float(values.quantile(0.90))
        output[f"{field}/std"] = float(values.std(unbiased=False))
    output["n_clips"] = len(results)
    output["n_pairs_total"] = sum(r.n_pairs for r in results)
    return output


def verdict(excess_ratio_mean: float) -> str:
    if excess_ratio_mean < 0.05:
        return "VAE_OK"
    if excess_ratio_mean < 0.10:
        return "VAE_MARGINAL"
    return "VAE_BINDING"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--openvid_root",
        type=str,
        default=None,
        help="OpenVid root. If omitted, resolved from OPENVID_ROOT in .env.",
    )
    parser.add_argument("--openvid_csv", type=str, default=None)
    parser.add_argument("--num_clips", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact_sheet_count", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diagnostics/vae_temporal",
    )
    parser.add_argument("--occlusion_threshold", type=float, default=1.0)
    parser.add_argument("--highpass_sigma", type=float, default=1.5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 32 clips, 4 contact sheets.",
    )
    parser.add_argument(
        "--ffmpeg_path", type=str, default="ffmpeg",
        help="Path to ffmpeg binary; required by OpenVidVideoDataset.",
    )
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.num_clips = 32
        args.contact_sheet_count = 4

    load_env_file(args.env_file)
    apply_hf_env_aliases()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)

    print(
        f"[A2] device={device} num_clips={args.num_clips} resolution={args.resolution}"
        f" num_frames={args.num_frames} seed={args.seed}"
    )

    print("[A2] loading SDXL VAE ...")
    vae = load_vae(device)
    print("[A2] loading RAFT-Large ...")
    raft, raft_transforms = load_raft(device)
    print("[A2] loading VGG16 feature extractor (LPIPS substitute) ...")
    vgg = VGGFeatureDistance(device)

    openvid_root = get_openvid_root(args.openvid_root)
    if openvid_root is None:
        raise SystemExit(
            "OpenVid root not set. Pass --openvid_root or set OPENVID_ROOT in .env."
        )
    openvid_csv = get_openvid_csv(args.openvid_csv)
    print(f"[A2] loading OpenVid dataset from {openvid_root} ...")
    dataset = OpenVidVideoDataset(
        root=openvid_root,
        csv_path=openvid_csv,
        num_frames_per_video=args.num_frames,
        resolution=args.resolution,
        frame_sampling="uniform",
        min_frames=args.num_frames,
        max_videos=None,
        ffmpeg_path=args.ffmpeg_path,
        center_crop=True,
        random_flip=False,
        image_interpolation_mode="lanczos",
    )
    print(f"[A2] dataset size = {len(dataset)} clips")

    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
        : args.num_clips
    ].tolist()

    results: list[ClipResult] = []
    per_clip_log: list[dict] = []
    start_time = time.time()

    for clip_position, dataset_index in enumerate(indices):
        try:
            sample = dataset[int(dataset_index)]
        except Exception as exc:  # pragma: no cover — runtime IO
            print(f"[A2] skipping index {dataset_index}: {exc}")
            continue
        frames = sample["frames"].to(device=device, dtype=torch.float32)
        if frames.shape[0] < 2:
            continue
        decoded = vae_roundtrip(vae, frames)
        result = evaluate_clip(
            frames_gt=frames,
            frames_decoded=decoded,
            raft=raft,
            raft_transforms=raft_transforms,
            vgg=vgg,
            occlusion_threshold=args.occlusion_threshold,
            highpass_sigma=args.highpass_sigma,
        )
        results.append(result)
        per_clip_log.append(
            {
                "dataset_index": int(dataset_index),
                "caption": str(sample.get("caption", ""))[:200],
                **asdict(result),
                "excess_charbonnier": result.excess_charbonnier,
                "excess_vgg_feat": result.excess_vgg_feat,
                "excess_ratio_charbonnier": result.excess_ratio_charbonnier,
                "excess_ratio_vgg_feat": result.excess_ratio_vgg_feat,
            }
        )

        if clip_position < args.contact_sheet_count:
            sheet_path = contact_dir / f"clip_{clip_position:03d}_idx{dataset_index}.png"
            title = (
                f"clip #{clip_position} (idx {dataset_index})"
                f" | excessRatio Charb={result.excess_ratio_charbonnier:+.3f}"
                f" VGG={result.excess_ratio_vgg_feat:+.3f}"
                f"\n{per_clip_log[-1]['caption'][:100]}"
            )
            try:
                save_contact_sheet(frames, decoded, sheet_path, title=title)
            except Exception as exc:  # pragma: no cover
                print(f"[A2] contact sheet save failed for clip {clip_position}: {exc}")

        if (clip_position + 1) % max(1, args.num_clips // 10) == 0:
            elapsed = time.time() - start_time
            mean_excess_charb = sum(r.excess_ratio_charbonnier for r in results) / len(results)
            mean_excess_vgg = sum(r.excess_ratio_vgg_feat for r in results) / len(results)
            print(
                f"[A2] {clip_position + 1}/{args.num_clips} clips"
                f" | excessRatio mean Charb={mean_excess_charb:+.3f}"
                f" VGG={mean_excess_vgg:+.3f}"
                f" | elapsed={elapsed:.1f}s"
            )

    if not results:
        print("[A2] no clips evaluated — check OpenVid dataset configuration.")
        return 1

    summary = aggregate(results)
    summary_verdict = verdict(summary["excess_ratio_charbonnier/mean"])
    summary["verdict"] = summary_verdict
    summary["args"] = vars(args)
    summary["elapsed_seconds"] = time.time() - start_time

    metrics_path = output_dir / "metrics.json"
    per_clip_path = output_dir / "per_clip.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    per_clip_path.write_text(json.dumps(per_clip_log, indent=2))

    print("\n[A2] ===== SUMMARY =====")
    print(f"  clips evaluated         : {summary['n_clips']}")
    print(f"  pairs evaluated         : {summary['n_pairs_total']}")
    print(f"  GT Charbonnier mean     : {summary['gt_charbonnier/mean']:.6f}")
    print(f"  Decoded Charbonnier mean: {summary['decoded_charbonnier/mean']:.6f}")
    print(f"  Excess Charb ratio mean : {summary['excess_ratio_charbonnier/mean']:+.4f}")
    print(f"  Excess Charb ratio p75  : {summary['excess_ratio_charbonnier/p75']:+.4f}")
    print(f"  GT VGG-feat mean        : {summary['gt_vgg_feat/mean']:.6f}")
    print(f"  Decoded VGG-feat mean   : {summary['decoded_vgg_feat/mean']:.6f}")
    print(f"  Excess VGG ratio mean   : {summary['excess_ratio_vgg_feat/mean']:+.4f}")
    print(f"  Excess VGG ratio p75    : {summary['excess_ratio_vgg_feat/p75']:+.4f}")
    print(f"  occlusion-valid frac    : {summary['valid_fraction/mean']:.3f}")
    print(f"  verdict                 : {summary_verdict}")
    print(f"  metrics.json            : {metrics_path}")
    print(f"  per_clip.json           : {per_clip_path}")
    print(f"  contact_sheets/         : {contact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
