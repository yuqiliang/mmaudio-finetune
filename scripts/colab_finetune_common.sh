#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/MMAudio}"
MYDRIVE_ROOT="${MYDRIVE_ROOT:-/content/drive/MyDrive}"

if [[ -z "${DRIVE_ROOT:-}" && -n "${MMAUDIO_DRIVE_ROOT:-}" ]]; then
  DRIVE_ROOT="${MMAUDIO_DRIVE_ROOT}"
fi

if [[ -z "${DRIVE_ROOT:-}" ]]; then
  for candidate in \
    "${MYDRIVE_ROOT}/MMAudio_Yuqi" \
    "${MYDRIVE_ROOT}/fine-tune"; do
    if [[ -d "${candidate}/data" \
      || -d "${candidate}/weights" \
      || -d "${candidate}/latents" \
      || -d "${candidate}/outputs" ]]; then
      DRIVE_ROOT="${candidate}"
      break
    fi
  done
  DRIVE_ROOT="${DRIVE_ROOT:-${MYDRIVE_ROOT}/MMAudio_Yuqi}"
fi

MODE="44k"
MODEL="small_44k"
LATENT_SEQ_LEN="345"
CLIP_SEQ_LEN="64"
SYNC_SEQ_LEN="192"

BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-18000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
SAVE_EVAL_INTERVAL="${SAVE_EVAL_INTERVAL:-1000000}"
SAVE_WEIGHTS_INTERVAL="${SAVE_WEIGHTS_INTERVAL:-500}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-500}"
AMP="${AMP:-True}"
COMPILE="${COMPILE:-False}"
PIN_MEMORY="${PIN_MEMORY:-False}"

VIDEO_ROOT="${DRIVE_ROOT}/data/soundscape_nonoverlap_8s"
MANIFEST_ROOT="${VIDEO_ROOT}/metadata"
LATENT_ROOT="${DRIVE_ROOT}/latents/soundscape_nonoverlap_small_44k"
WEIGHTS_DIR="${DRIVE_ROOT}/weights"
EXT_WEIGHTS_DIR="${DRIVE_ROOT}/ext_weights"
OUTPUT_DIR="${DRIVE_ROOT}/outputs"
LOG_DIR="${DRIVE_ROOT}/logs"
export HF_HOME="${HF_HOME:-${DRIVE_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${DRIVE_ROOT}/cache/torch}"

TRAIN_TSV="${MANIFEST_ROOT}/video_train_ft.tsv"
VAL_TSV="${MANIFEST_ROOT}/video_val_ft.tsv"
TEST_TSV="${MANIFEST_ROOT}/video_test_ft.tsv"
TRAIN_MEMMAP="${LATENT_ROOT}/train/training_latents"
VAL_MEMMAP="${LATENT_ROOT}/val/training_latents"
TEST_MEMMAP="${LATENT_ROOT}/test/training_latents"

BASE_WEIGHTS="${WEIGHTS_DIR}/mmaudio_small_44k.pth"
VAE_44K_CKPT="${EXT_WEIGHTS_DIR}/v1-44.pth"
SYNCHFORMER_CKPT="${EXT_WEIGHTS_DIR}/synchformer_state_dict.pth"

RUN_DIR="${OUTPUT_DIR}/${EXP_ID}"
LOG_FILE="${LOG_DIR}/train_${EXP_ID}.log"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

link_repo_file() {
  local repo_rel="$1"
  local drive_path="$2"
  local repo_path="${REPO_ROOT}/${repo_rel}"
  mkdir -p "$(dirname "${repo_path}")"
  if [[ -e "${repo_path}" || -L "${repo_path}" ]]; then
    return
  fi
  ln -s "${drive_path}" "${repo_path}"
}

prepare_colab_training() {
  for path in \
    "${REPO_ROOT}" \
    "${TRAIN_TSV}" \
    "${VAL_TSV}" \
    "${TEST_TSV}" \
    "${TRAIN_MEMMAP}" \
    "${VAL_MEMMAP}" \
    "${TEST_MEMMAP}" \
    "${BASE_WEIGHTS}" \
    "${VAE_44K_CKPT}" \
    "${SYNCHFORMER_CKPT}"; do
    require_path "${path}"
  done

  link_repo_file "weights/mmaudio_small_44k.pth" "${BASE_WEIGHTS}"
  link_repo_file "ext_weights/v1-44.pth" "${VAE_44K_CKPT}"
  link_repo_file "ext_weights/synchformer_state_dict.pth" "${SYNCHFORMER_CKPT}"

  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${HF_HOME}" "${TORCH_HOME}"
}

run_mmaudio_training() {
  local initial_weights="$1"
  local checkpoint="$2"
  prepare_colab_training
  cd "${REPO_ROOT}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

  echo "Starting MMAudio fine-tuning"
  echo "  exp_id=${EXP_ID}"
  echo "  model=${MODEL}"
  echo "  mode=${MODE}"
  echo "  sequences=${LATENT_SEQ_LEN}/${CLIP_SEQ_LEN}/${SYNC_SEQ_LEN}"
  echo "  batch_size=${BATCH_SIZE}"
  echo "  train_memmap=${TRAIN_MEMMAP}"
  echo "  output=${RUN_DIR}"

  torchrun --standalone --nproc_per_node=1 train.py \
    "exp_id=${EXP_ID}" \
    "model=${MODEL}" \
    "++mode=${MODE}" \
    "weights=${initial_weights}" \
    "checkpoint=${checkpoint}" \
    "batch_size=${BATCH_SIZE}" \
    "eval_batch_size=${EVAL_BATCH_SIZE}" \
    "learning_rate=${LEARNING_RATE}" \
    "num_iterations=${NUM_ITERATIONS}" \
    "linear_warmup_steps=${WARMUP_STEPS}" \
    "log_text_interval=50" \
    "val_interval=${VAL_INTERVAL}" \
    "eval_interval=${EVAL_INTERVAL}" \
    "save_eval_interval=${SAVE_EVAL_INTERVAL}" \
    "save_weights_interval=${SAVE_WEIGHTS_INTERVAL}" \
    "save_checkpoint_interval=${SAVE_CHECKPOINT_INTERVAL}" \
    "num_workers=${NUM_WORKERS}" \
    "amp=${AMP}" \
    "compile=${COMPILE}" \
    "pin_memory=${PIN_MEMORY}" \
    "hydra.run.dir=${RUN_DIR}" \
    "hydra.output_subdir=train-hydra" \
    "ema.checkpoint_folder=${RUN_DIR}/ema_ckpts" \
    "ema.enable=True" \
    "ema.checkpoint_every=${SAVE_CHECKPOINT_INTERVAL}" \
    "vae_44k_ckpt=${VAE_44K_CKPT}" \
    "synchformer_ckpt=${SYNCHFORMER_CKPT}" \
    "++data_dim.latent_seq_len=${LATENT_SEQ_LEN}" \
    "++data_dim.clip_seq_len=${CLIP_SEQ_LEN}" \
    "++data_dim.sync_seq_len=${SYNC_SEQ_LEN}" \
    "data.ExtractedVGG.tsv=${TRAIN_TSV}" \
    "data.ExtractedVGG.memmap_dir=${TRAIN_MEMMAP}" \
    "data.ExtractedVGG_val.tsv=${VAL_TSV}" \
    "data.ExtractedVGG_val.memmap_dir=${VAL_MEMMAP}" \
    "data.ExtractedVGG_val.gt_cache=null" \
    "data.ExtractedVGG_val.output_subdir=val" \
    "data.ExtractedVGG_test.tsv=${TEST_TSV}" \
    "data.ExtractedVGG_test.memmap_dir=${TEST_MEMMAP}" \
    "data.ExtractedVGG_test.gt_cache=null" \
    "data.ExtractedVGG_test.output_subdir=null" \
    2>&1 | tee "${LOG_FILE}"
}
