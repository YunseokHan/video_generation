#!/usr/bin/env bash
# E3/E4 launcher. Warm-starts from E1 (xnocross) checkpoint-last. See
# information/claude-codex-discussion.md §5. Launch only after E1 finished.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise_boundary_anchor_calib.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
