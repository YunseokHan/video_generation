#!/usr/bin/env bash
# A2 — VAE temporal coherence diagnostic launcher.
# See diagnostics/vae_temporal_diagnostic.py and
# information/12_vae_temporal_diagnostic.md.
#
# Examples:
#   bash scripts/diagnostic_vae_temporal.sh --smoke
#   bash scripts/diagnostic_vae_temporal.sh --num_clips 100
#   bash scripts/diagnostic_vae_temporal.sh --num_clips 300 \
#       --output_dir outputs/diagnostics/vae_temporal_robust

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

ENV_FILE="${ENV_FILE:-.env}"
load_shell_env "${ENV_FILE}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/../diagnostics/vae_temporal_diagnostic.py" \
  --env_file "${ENV_FILE}" \
  "$@"
