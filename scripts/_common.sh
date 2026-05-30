#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-.env}"

# Hardcoded last-resort fallback. The authoritative per-server value belongs in
# .env (PYTHON_BIN=...). Capture any caller/CLI override now, before .env is
# loaded, so precedence stays: CLI override > .env > this fallback.
DEFAULT_PYTHON_BIN="/home/work/data/miniconda3/envs/video/bin/python"
_CLI_PYTHON_BIN="${PYTHON_BIN:-}"
_CLI_ACCELERATE_BIN="${ACCELERATE_BIN:-}"

_resolve_bins() {
  PYTHON_BIN="${_CLI_PYTHON_BIN:-${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}}"
  ACCELERATE_BIN="${_CLI_ACCELERATE_BIN:-${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}}"
}

load_shell_env() {
  local env_path="${1:-${ENV_FILE}}"
  # Resolve relative env files against the project root so this works no matter
  # what directory the launcher was invoked from.
  if [[ "${env_path}" != /* ]]; then
    env_path="${PROJECT_ROOT}/${env_path}"
  fi
  if [[ -f "${env_path}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_path}"
    set +a
  fi
  # Re-resolve after every (re)load so an in-.env PYTHON_BIN/ACCELERATE_BIN and
  # the empty ACCELERATE_BIN= placeholder never clobber a CLI override or the
  # derived accelerate path on repeated calls.
  _resolve_bins
}

# Load .env up front so PYTHON_BIN is available to scripts that read it right
# after sourcing this file; launchers may call load_shell_env again safely.
load_shell_env "${ENV_FILE}"

cd "${PROJECT_ROOT}"
