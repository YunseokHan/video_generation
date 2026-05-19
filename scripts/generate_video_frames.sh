#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

load_shell_env "${ENV_FILE}"

INFER_CONFIG="${INFER_CONFIG:-${CONFIG:-configs/train/default.yaml}}"
CHECKPOINT="${CHECKPOINT:-outputs/sdxl_frame_generator/checkpoint-last}"
PROMPT="${PROMPT:-Astronaut walking through a jungle, cold color palette}"
NUM_FRAMES="${NUM_FRAMES:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/example_video}"
SAVE_MP4="${SAVE_MP4:-0}"
NO_GRID="${NO_GRID:-0}"

cmd=(
  "${PYTHON_BIN}"
  "infer.py"
  "--config" "${INFER_CONFIG}"
  "--env_file" "${ENV_FILE}"
  "--checkpoint" "${CHECKPOINT}"
  "--prompt" "${PROMPT}"
  "--num_frames" "${NUM_FRAMES}"
  "--output_dir" "${OUTPUT_DIR}"
)

if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  cmd+=("--num_inference_steps" "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${GUIDANCE_SCALE:-}" ]]; then
  cmd+=("--guidance_scale" "${GUIDANCE_SCALE}")
fi
if [[ -n "${SEED:-}" ]]; then
  cmd+=("--seed" "${SEED}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  cmd+=("--batch_size" "${BATCH_SIZE}")
fi
if [[ "${SAVE_MP4}" == "1" || "${SAVE_MP4}" == "true" ]]; then
  cmd+=("--save_mp4")
fi
if [[ "${NO_GRID}" == "1" || "${NO_GRID}" == "true" ]]; then
  cmd+=("--no_grid")
fi

exec "${cmd[@]}" "$@"
