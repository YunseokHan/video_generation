#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

load_shell_env "${ENV_FILE}"

exec "${PYTHON_BIN}" \
  training/train_sdxl_frame_generator.py \
  --config "${CONFIG}" \
  --env_file "${ENV_FILE}" \
  "$@"
