#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

load_shell_env "${ENV_FILE}"

exec "${PYTHON_BIN}" - <<'PY'
from tests.test_core import (
    test_env_file_loads_hf_aliases,
    test_flatten_video_batch_repeats_captions_and_positions,
    test_frame_positions_single_and_uniform,
    test_sdxl_time_ids_shape_and_values,
    test_temporal_mlp_matches_pooled_shape,
)

shape_tests = [
    test_frame_positions_single_and_uniform,
    test_flatten_video_batch_repeats_captions_and_positions,
    test_temporal_mlp_matches_pooled_shape,
    test_sdxl_time_ids_shape_and_values,
]

for test_fn in shape_tests:
    test_fn()

print("core shape tests ok")
print("env alias test requires pytest tmp_path fixture; run it with pytest when available")
PY
