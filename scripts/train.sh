#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

SHELL_TRAIN_CONFIG="${TRAIN_CONFIG:-}"
SHELL_CONFIG="${CONFIG:-}"
SHELL_ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
SHELL_TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-}"

load_shell_env "${ENV_FILE}"

TRAIN_CONFIG="${SHELL_TRAIN_CONFIG:-${TRAIN_CONFIG:-${SHELL_CONFIG:-${CONFIG:-configs/train/default.yaml}}}}"
ACCELERATE_CONFIG="${SHELL_ACCELERATE_CONFIG:-${ACCELERATE_CONFIG:-configs/accelerate/default.yaml}}"
TRAIN_LAUNCHER="${SHELL_TRAIN_LAUNCHER:-${TRAIN_LAUNCHER:-accelerate}}"

case "${TRAIN_LAUNCHER}" in
  accelerate)
    exec "${ACCELERATE_BIN}" launch \
      --config_file "${ACCELERATE_CONFIG}" \
      train.py \
      --config "${TRAIN_CONFIG}" \
      --env_file "${ENV_FILE}" \
      "$@"
    ;;
  python | single | single_process)
    exec "${PYTHON_BIN}" \
      train.py \
      --config "${TRAIN_CONFIG}" \
      --env_file "${ENV_FILE}" \
      "$@"
    ;;
  *)
    echo "Unsupported TRAIN_LAUNCHER=${TRAIN_LAUNCHER}. Use accelerate or python." >&2
    exit 2
    ;;
esac
