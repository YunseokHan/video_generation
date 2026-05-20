#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

load_shell_env "${ENV_FILE}"

INFER_CONFIG="${INFER_CONFIG:-${CONFIG:-}}"
NAME="${NAME:-}"
STEP="${STEP:-}"
CHECKPOINT="${CHECKPOINT:-}"
PROMPT="${PROMPT:-Astronaut walking through a jungle, cold color palette}"
NUM_FRAMES="${NUM_FRAMES:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SAVE_MP4="${SAVE_MP4:-}"
NO_GRID="${NO_GRID:-0}"

cmd=(
  "${PYTHON_BIN}"
  "infer.py"
  "--env_file" "${ENV_FILE}"
  "--prompt" "${PROMPT}"
  "--num_frames" "${NUM_FRAMES}"
)

if [[ -n "${INFER_CONFIG}" ]]; then
  cmd+=("--config" "${INFER_CONFIG}")
fi

if [[ -n "${NAME}" && -n "${STEP}" ]]; then
  cmd+=("--name" "${NAME}" "--step" "${STEP}")
elif [[ -n "${CHECKPOINT}" ]]; then
  cmd+=("--checkpoint" "${CHECKPOINT}")
else
  cmd+=("--checkpoint" "outputs/sdxl-resnet-attention-sinusoidal/checkpoint-last")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=("--output_dir" "${OUTPUT_DIR}")
fi

if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  cmd+=("--num_inference_steps" "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${GUIDANCE_SCALE:-}" ]]; then
  cmd+=("--guidance_scale" "${GUIDANCE_SCALE}")
fi
if [[ -n "${GUIDANCE_SCALES:-}" ]]; then
  # shellcheck disable=SC2206
  scales=(${GUIDANCE_SCALES})
  cmd+=("--guidance_scales" "${scales[@]}")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=("--seed" "${SEED}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  cmd+=("--batch_size" "${BATCH_SIZE}")
fi
if [[ "${SAVE_MP4}" == "1" || "${SAVE_MP4}" == "true" ]]; then
  cmd+=("--save_mp4")
elif [[ "${SAVE_MP4}" == "0" || "${SAVE_MP4}" == "false" ]]; then
  cmd+=("--no_mp4")
fi
if [[ "${NO_GRID}" == "1" || "${NO_GRID}" == "true" ]]; then
  cmd+=("--no_grid")
fi

exec "${cmd[@]}" "$@"
