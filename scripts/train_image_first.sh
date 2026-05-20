#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/image_first_sinusoidal.yaml}"
exec "${SCRIPT_DIR}/train.sh" "$@"
