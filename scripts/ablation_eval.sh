#!/usr/bin/env bash
# Ablation evaluation harness launcher.
# See diagnostics/ablation_eval.py and information/13_ablation_eval.md.
#
# Examples:
#   # smoke (2 models, 1 prompt, 1 seed, no wandb)
#   bash scripts/ablation_eval.sh \
#     --models image-first-smooth-snr-renoise-boundary,image-first-smooth-snr-renoise-boundary-xnocross \
#     --num_prompts 1 --seeds 0 --baseline_extra_seeds "" --num_log_videos 1
#
#   # full MVE sweep over the 6 runs, logged to one wandb run.
#   # First build the 96-prompt benchmark (synthetic + held-out OpenVid):
#   #   python diagnostics/build_eval_prompts.py --num_openvid 72 --output diagnostics/prompts_eval.txt
#   bash scripts/ablation_eval.sh \
#     --models image-first-smooth-snr-renoise-boundary,image-first-smooth-snr-renoise-boundary-xnocross,image-first-smooth-snr-boundary,image-first-smooth-snr,image-first-snr-renoise,image-first-smooth-snr-renoise \
#     --baseline image-first-smooth-snr-renoise-boundary \
#     --prompts_file diagnostics/prompts_eval.txt \
#     --seeds 0,1,2 --baseline_extra_seeds 3,4,5 --cfg 8 --t1 0.5 --wandb

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
ENV_FILE="${ENV_FILE:-.env}"
load_shell_env "${ENV_FILE}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/../diagnostics/ablation_eval.py" \
  --env_file "${ENV_FILE}" \
  "$@"
