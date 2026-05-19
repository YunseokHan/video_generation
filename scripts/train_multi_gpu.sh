#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-accelerate}" exec "${SCRIPT_DIR}/train.sh" "$@"
