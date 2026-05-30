#!/usr/bin/env python
"""Build the ablation-eval prompt set by mixing curated synthetic prompts with
held-out OpenVid captions.

Why: codex's MVE spec (information/claude-codex-discussion.md §8) recommends
evaluating on a mix of controlled synthetic prompts AND held-out, in-distribution
OpenVid-style captions, ~96 total across motion buckets.

"Held-out" here = a deterministic partition by a stable hash of the clip's video
filename (`hash(video) % 100 < --holdout_pct`). It is reproducible, so the SAME
partition can be excluded from future training to make it *truly* unseen.

  IMPORTANT: the currently-running models were trained on the full OpenVid index
  (no holdout filter was applied), so for THOSE checkpoints these captions are
  "OpenVid-style in-distribution", not strictly unseen. To make the partition
  genuinely held-out for future runs, filter training rows with the same hash
  rule (the recipe is printed at the end of this script).

Output: a prompt file (default diagnostics/prompts_eval.txt) with a tagged
synthetic block followed by a tagged held-out OpenVid block. `# ...` lines are
comments; `ablation_eval.py --prompts_file` ignores them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framegen.env import apply_hf_env_aliases, get_openvid_csv, get_openvid_root, load_env_file

CAMERA_BUCKETS = ("static", "pan", "tilt", "zoom", "other", "unknown")


def camera_bucket(raw: str) -> str:
    """Map OpenVid 'camera motion' values (e.g. 'pan_left+tilt_up', 'static',
    'Undetermined') to a coarse bucket for diversity sampling."""
    v = (raw or "").strip().lower()
    if not v or v in {"undetermined", "unknown"}:
        return "unknown"
    if v == "static":
        return "static"
    for key in ("pan", "tilt", "zoom"):  # first matching primary motion wins
        if key in v:
            return key
    return "other"


def stable_holdout(video: str, pct: int) -> bool:
    h = int(hashlib.sha1(video.encode("utf-8")).hexdigest(), 16) % 100
    return h < int(pct)


def clean_caption(text: str, max_chars: int) -> str:
    t = " ".join(str(text).split())  # collapse whitespace/newlines
    # Prefer a sentence-aware cut so prompts stay coherent.
    if len(t) > max_chars:
        head = t[:max_chars]
        cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        t = head[: cut + 1] if cut > max_chars // 2 else head.rstrip() + "..."
    return t.strip()


def _f(row: dict, key: str):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num_openvid", type=int, default=72,
                   help="Held-out OpenVid captions to sample (default 72; +24 synthetic = 96).")
    p.add_argument("--holdout_pct", type=int, default=10,
                   help="Percent of clips in the held-out partition (by stable hash).")
    p.add_argument("--min_aesthetic", type=float, default=None)
    p.add_argument("--min_motion", type=float, default=2.0,
                   help="Min motion score so prompts imply real motion (video-relevant).")
    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--max_chars", type=int, default=220)
    p.add_argument("--per_camera_cap", type=int, default=None,
                   help="Optional cap per camera-motion bucket for diversity.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic_file", default="diagnostics/prompts_ablation.txt",
                   help="Curated synthetic prompts to include (set '' to skip).")
    p.add_argument("--output", default="diagnostics/prompts_eval.txt")
    p.add_argument("--openvid_root", default=None)
    p.add_argument("--openvid_csv", default=None)
    p.add_argument("--env_file", default=".env")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    apply_hf_env_aliases()

    root = get_openvid_root(args.openvid_root)
    csv_path = get_openvid_csv(args.openvid_csv)
    if csv_path is None:
        if root is None:
            raise SystemExit("Set OPENVID_ROOT in .env or pass --openvid_root/--openvid_csv.")
        csv_path = str(Path(root) / "OpenVid.csv")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"OpenVid CSV not found: {csv_path}")

    print(f"[prompts] reading {csv_path} (holdout {args.holdout_pct}% by stable hash)")
    by_bucket: dict[str, list[str]] = {b: [] for b in CAMERA_BUCKETS}
    seen = set()
    n_rows = n_heldout = n_kept = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_rows += 1
            video = row.get("video") or row.get("filepath") or ""
            if not stable_holdout(video, args.holdout_pct):
                continue
            n_heldout += 1
            cap = clean_caption(row.get("caption", ""), args.max_chars)
            if not (args.min_chars <= len(cap) <= args.max_chars):
                continue
            key = cap.lower()[:80]
            if key in seen:
                continue
            if args.min_aesthetic is not None:
                a = _f(row, "aesthetic score")
                if a is None or a < args.min_aesthetic:
                    continue
            if args.min_motion is not None:
                mo = _f(row, "motion score")
                if mo is None or mo < args.min_motion:
                    continue
            seen.add(key)
            by_bucket[camera_bucket(row.get("camera motion", ""))].append(cap)
            n_kept += 1

    print(f"[prompts] rows={n_rows} held-out={n_heldout} passed-filters={n_kept}")
    for b in CAMERA_BUCKETS:
        print(f"           {b:12s}: {len(by_bucket[b])}")

    rng = random.Random(args.seed)
    for b in by_bucket:
        rng.shuffle(by_bucket[b])
        if args.per_camera_cap is not None:
            by_bucket[b] = by_bucket[b][: args.per_camera_cap]

    # Round-robin across camera buckets for diversity, up to num_openvid.
    pool = [b for b in CAMERA_BUCKETS if by_bucket[b]]
    sampled: list[str] = []
    idx = {b: 0 for b in pool}
    while len(sampled) < args.num_openvid and pool:
        for b in list(pool):
            if idx[b] >= len(by_bucket[b]):
                pool.remove(b)
                continue
            sampled.append(by_bucket[b][idx[b]])
            idx[b] += 1
            if len(sampled) >= args.num_openvid:
                break
    if len(sampled) < args.num_openvid:
        print(f"[prompts] WARNING: only {len(sampled)} held-out captions passed filters "
              f"(< requested {args.num_openvid}); relax --min_motion/--min_chars or raise --holdout_pct.")

    synthetic: list[str] = []
    if args.synthetic_file:
        sp = PROJECT_ROOT / args.synthetic_file
        if sp.exists():
            synthetic = [
                l.strip() for l in sp.read_text().splitlines()
                if l.strip() and not l.lstrip().startswith("#")
            ]

    out = PROJECT_ROOT / args.output
    lines = [
        "# Ablation eval prompt set = curated synthetic + held-out OpenVid captions.",
        f"# Built by diagnostics/build_eval_prompts.py (seed={args.seed}, "
        f"holdout_pct={args.holdout_pct}, min_motion={args.min_motion}).",
        f"# synthetic={len(synthetic)}  openvid_heldout={len(sampled)}  total={len(synthetic)+len(sampled)}",
        "#",
        "# --- curated synthetic ---",
        *synthetic,
        "",
        "# --- held-out OpenVid captions (deterministic hash partition) ---",
        *sampled,
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"[prompts] wrote {len(synthetic)+len(sampled)} prompts -> {out}")
    print("\n[prompts] To make this partition TRULY held-out for FUTURE training, exclude")
    print("           the same hash partition in the dataloader, e.g. keep a row only when:")
    print(f"           int(hashlib.sha1(video.encode()).hexdigest(),16) % 100 >= {args.holdout_pct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
