#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON_BIN="/NHNHOME/WORKSPACE/26moe001_D/miniconda3/envs/video/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}"

ENV_FILE="${ENV_FILE:-.env}"

cd "${PROJECT_ROOT}"

load_shell_env() {
  local env_path="${1:-${ENV_FILE}}"
  if [[ -f "${env_path}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_path}"
    set +a
  fi
}
