#!/usr/bin/env bash
# 2x2 ablation corner (boundary=no, pred_x0_renoise=yes).
# See information/claude-codex-discussion.md §7 (Parallel-run plan).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
