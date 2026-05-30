#!/usr/bin/env bash
# E4 launcher: INDEPENDENT fresh 15k run (no warm-start), baseline-matched, so
# E4 vs baseline isolates the rollout pred_x0 source effect. Does NOT depend on
# E1; can run as soon as a GPU group is free. See
# information/claude-codex-discussion.md §3 U3 (E4) + §9.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise_boundary_rollout.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
