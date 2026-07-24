#!/usr/bin/env bash
set -euo pipefail

# Set this to a full optimizer checkpoint, not an EMA-only weights file.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EXP_ID="${EXP_ID:-urban_soundscape_small44k_nonstride_v1_continue}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to the checkpoint to resume." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/colab_finetune_common.sh"

require_path "${CHECKPOINT_PATH}"
run_mmaudio_training "null" "${CHECKPOINT_PATH}"
