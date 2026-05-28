#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_smooth_snr_renoise_boundary.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
