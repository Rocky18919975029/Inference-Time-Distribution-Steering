#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CONFIG="${CONFIG:-${ROOT_DIR}/configs/itds_qwen25_7b_math.yaml}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/full_train_subtb_with_ref.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/itds_tb_qwen25_7b}"
OBJECTIVE="${OBJECTIVE:-tb}"
TOP_K="${TOP_K:-64}"
RANK="${RANK:-32}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.1}"
NUM_GPUS="${NUM_GPUS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-0}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-itds}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-itds_${OBJECTIVE}_topk${TOP_K}_r${RANK}}"
WANDB_MODE="${WANDB_MODE:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${WANDB_MODE}" ]]; then
  WANDB_MODE_OVERRIDE="--override=wandb_mode=${WANDB_MODE}"
else
  WANDB_MODE_OVERRIDE="--override=wandb_mode=null"
fi

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  -m itds.train \
  --config "${CONFIG}" \
  --override "model_name_or_path=${MODEL_NAME_OR_PATH}" \
  --override "train_data_path=${TRAIN_DATA}" \
  --override "output_dir=${OUTPUT_DIR}" \
  --override "objective=${OBJECTIVE}" \
  --override "top_k=${TOP_K}" \
  --override "rank=${RANK}" \
  --override "alpha=${ALPHA}" \
  --override "beta=${BETA}" \
  --override "max_steps=${MAX_STEPS}" \
  --override "num_epochs=${NUM_EPOCHS}" \
  --override "save_steps=${SAVE_STEPS}" \
  --override "logging_steps=${LOGGING_STEPS}" \
  --override "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --override "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  --override "learning_rate=${LEARNING_RATE}" \
  --override "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}" \
  --override "wandb_project=${WANDB_PROJECT}" \
  --override "wandb_run_name=${WANDB_RUN_NAME}" \
  "${WANDB_MODE_OVERRIDE}"
