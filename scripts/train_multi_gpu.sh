#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

load_shell_env "${ENV_FILE}"

NUM_PROCESSES="$(resolve_num_processes)"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"

launch_args=(
  "--num_processes" "${NUM_PROCESSES}"
  "--num_machines" "${NUM_MACHINES}"
  "--mixed_precision" "${MIXED_PRECISION}"
  "--dynamo_backend" "${DYNAMO_BACKEND}"
  "--main_process_port" "${MAIN_PROCESS_PORT}"
)

if (( NUM_PROCESSES > 1 )); then
  launch_args=("--multi_gpu" "${launch_args[@]}")
fi

exec "${ACCELERATE_BIN}" launch \
  "${launch_args[@]}" \
  training/train_sdxl_frame_generator.py \
  --config "${CONFIG}" \
  --env_file "${ENV_FILE}" \
  "$@"
