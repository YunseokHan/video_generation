#!/usr/bin/env bash
# Experiment E1 launcher. See information/claude-codex-discussion.md §5 E1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise_boundary_xnocross.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
