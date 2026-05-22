#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/MMAudio}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/MMAudio_Yuqi}"

EXP_ID="urban_ft_8s_stride4_18k"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DRIVE_ROOT}/outputs/${EXP_ID}/${EXP_ID}_ckpt_last.pth}"

MODE="16k"
MODEL="small_16k"
STRIDE="4"

BATCH_SIZE="2"
LEARNING_RATE="2e-5"
NUM_ITERATIONS="18000"
WARMUP_STEPS="500"

LATENT_SEQ_LEN="250"
CLIP_SEQ_LEN="64"
SYNC_SEQ_LEN="192"

NUM_WORKERS="2"
EVAL_BATCH_SIZE="2"
VAL_INTERVAL="500"
EVAL_INTERVAL="2000"
SAVE_EVAL_INTERVAL="4000"
SAVE_WEIGHTS_INTERVAL="1000"
SAVE_CHECKPOINT_INTERVAL="1000"
AMP="True"
COMPILE="False"
PIN_MEMORY="True"

LATENT_ROOT="${DRIVE_ROOT}/latents/stride${STRIDE}"
WEIGHTS_DIR="${DRIVE_ROOT}/weights"
EXT_WEIGHTS_DIR="${DRIVE_ROOT}/ext_weights"
OUTPUT_DIR="${DRIVE_ROOT}/outputs"
LOG_DIR="${DRIVE_ROOT}/logs"
RUN_DIR="${OUTPUT_DIR}/${EXP_ID}"

TSV_ROOT="${DRIVE_ROOT}/tsv/stride${STRIDE}"
if [[ ! -f "${TSV_ROOT}/video_train_ft.tsv" ]]; then
  TSV_ROOT="${DRIVE_ROOT}/tsv"
fi

TRAIN_TSV="${TSV_ROOT}/video_train_ft.tsv"
VAL_TSV="${TSV_ROOT}/video_val_ft.tsv"
TEST_TSV="${TSV_ROOT}/video_test_ft.tsv"

TRAIN_MEMMAP="${LATENT_ROOT}/train/training_latents"
VAL_MEMMAP="${LATENT_ROOT}/val/training_latents"
TEST_MEMMAP="${LATENT_ROOT}/test/training_latents"

VAE_16K_CKPT="${EXT_WEIGHTS_DIR}/v1-16.pth"
BIGVGAN_CKPT="${EXT_WEIGHTS_DIR}/best_netG.pt"
SYNCHFORMER_CKPT="${EXT_WEIGHTS_DIR}/synchformer_state_dict.pth"
EMPTY_STRING="${EXT_WEIGHTS_DIR}/empty_string.pth"

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

for path in \
  "${REPO_ROOT}" \
  "${CHECKPOINT_PATH}" \
  "${TRAIN_TSV}" \
  "${VAL_TSV}" \
  "${TEST_TSV}" \
  "${TRAIN_MEMMAP}" \
  "${VAL_MEMMAP}" \
  "${TEST_MEMMAP}" \
  "${VAE_16K_CKPT}" \
  "${BIGVGAN_CKPT}" \
  "${SYNCHFORMER_CKPT}" \
  "${EMPTY_STRING}"; do
  require_path "${path}"
done

link_repo_file "ext_weights/v1-16.pth" "${VAE_16K_CKPT}"
link_repo_file "ext_weights/best_netG.pt" "${BIGVGAN_CKPT}"
link_repo_file "ext_weights/synchformer_state_dict.pth" "${SYNCHFORMER_CKPT}"
link_repo_file "ext_weights/empty_string.pth" "${EMPTY_STRING}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "Continuing fine-tuning:"
echo "  exp_id=${EXP_ID}"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  mode=${MODE}"
echo "  model=${MODEL}"
echo "  latents=${LATENT_ROOT}"
echo "  output=${RUN_DIR}"

torchrun --standalone --nproc_per_node=1 train.py \
  "exp_id=${EXP_ID}" \
  "model=${MODEL}" \
  "++mode=${MODE}" \
  "checkpoint=${CHECKPOINT_PATH}" \
  "weights=null" \
  "batch_size=${BATCH_SIZE}" \
  "eval_batch_size=${EVAL_BATCH_SIZE}" \
  "learning_rate=${LEARNING_RATE}" \
  "num_iterations=${NUM_ITERATIONS}" \
  "linear_warmup_steps=${WARMUP_STEPS}" \
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
  "vae_16k_ckpt=${VAE_16K_CKPT}" \
  "bigvgan_vocoder_ckpt=${BIGVGAN_CKPT}" \
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
  "data.ExtractedVGG_test.output_subdir=null"
