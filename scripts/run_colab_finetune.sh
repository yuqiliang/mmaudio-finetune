#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:-urban_soundscape_small44k_nonstride_v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/colab_finetune_common.sh"

run_mmaudio_training "${BASE_WEIGHTS}" "null"
