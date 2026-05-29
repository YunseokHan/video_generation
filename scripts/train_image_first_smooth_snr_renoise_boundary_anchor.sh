#!/usr/bin/env bash
# Experiment E2 launcher. See information/claude-codex-discussion.md §5 E2.
#
# NOTE: this launcher will not produce a meaningful E2 run until the code
# changes listed in the E2 config header (anchor branch, mid+up placement)
# are implemented. Until then it would silently behave as an E1 continuation
# fine-tune. Implement the code first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise_boundary_anchor.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
