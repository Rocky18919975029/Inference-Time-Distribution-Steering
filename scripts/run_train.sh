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

CONFIG="${CONFIG:-${ROOT_DIR}/configs/qwen25_7b_math.yaml}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/data/mini_train_subtb_with_ref.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/mini_train_newline_subtb}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"

NUM_GPUS="${NUM_GPUS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero2_cpu_offload.json}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-0}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.000002}"
MAX_PAIRS_PER_TRACE="${MAX_PAIRS_PER_TRACE:-16}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

WANDB_PROJECT="${WANDB_PROJECT:-one_step_posttrain}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mini_train_newline_subtb}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_MODE="${WANDB_MODE:-}"

LIMIT_OF_RLVR_DIR="${LIMIT_OF_RLVR_DIR:-${ROOT_DIR}/limit-of-RLVR}"
LIMIT_DATA_DIR="${LIMIT_DATA_DIR:-}"

EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
EVAL_DATA="${EVAL_DATA:-${ROOT_DIR}/data/test.parquet}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval}"
EVAL_BENCHMARK="${EVAL_BENCHMARK:-eval_test}"
EVAL_PROMPT_TYPE="${EVAL_PROMPT_TYPE:-qwen-boxed}"
EVAL_SEED="${EVAL_SEED:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
EVAL_N_SAMPLING="${EVAL_N_SAMPLING:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-1024}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Training data not found: ${TRAIN_DATA}" >&2
  echo "Expected output from scripts/run_rollout.sh, usually data/mini_train_subtb_with_ref.jsonl." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
python - <<PY
import json
from pathlib import Path

payload = {
    "CONFIG": "${CONFIG}",
    "TRAIN_DATA": "${TRAIN_DATA}",
    "OUTPUT_DIR": "${OUTPUT_DIR}",
    "MODEL_NAME_OR_PATH": "${MODEL_NAME_OR_PATH}",
    "NCCL_P2P_DISABLE": "${NCCL_P2P_DISABLE}",
    "NCCL_IB_DISABLE": "${NCCL_IB_DISABLE}",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "${TORCH_NCCL_ASYNC_ERROR_HANDLING}",
    "TORCH_NCCL_BLOCKING_WAIT": "${TORCH_NCCL_BLOCKING_WAIT}",
    "OMP_NUM_THREADS": "${OMP_NUM_THREADS}",
    "NUM_GPUS": "${NUM_GPUS}",
    "MIXED_PRECISION": "${MIXED_PRECISION}",
    "USE_DEEPSPEED": "${USE_DEEPSPEED}",
    "DEEPSPEED_CONFIG": "${DEEPSPEED_CONFIG}",
    "MAX_STEPS": "${MAX_STEPS}",
    "NUM_EPOCHS": "${NUM_EPOCHS}",
    "SAVE_STEPS": "${SAVE_STEPS}",
    "LOGGING_STEPS": "${LOGGING_STEPS}",
    "PER_DEVICE_TRAIN_BATCH_SIZE": "${PER_DEVICE_TRAIN_BATCH_SIZE}",
    "GRADIENT_ACCUMULATION_STEPS": "${GRADIENT_ACCUMULATION_STEPS}",
    "LEARNING_RATE": "${LEARNING_RATE}",
    "MAX_PAIRS_PER_TRACE": "${MAX_PAIRS_PER_TRACE}",
    "RESUME_FROM_CHECKPOINT": "${RESUME_FROM_CHECKPOINT}",
    "FINETUNE_MODE": "${FINETUNE_MODE}",
    "LORA_R": "${LORA_R}",
    "LORA_ALPHA": "${LORA_ALPHA}",
    "LORA_DROPOUT": "${LORA_DROPOUT}",
    "LORA_TARGET_MODULES": "${LORA_TARGET_MODULES}",
    "WANDB_PROJECT": "${WANDB_PROJECT}",
    "WANDB_RUN_NAME": "${WANDB_RUN_NAME}",
    "WANDB_ENTITY": "${WANDB_ENTITY}",
    "WANDB_RUN_ID": "${WANDB_RUN_ID}",
    "WANDB_MODE": "${WANDB_MODE}",
    "LIMIT_OF_RLVR_DIR": "${LIMIT_OF_RLVR_DIR}",
    "LIMIT_DATA_DIR": "${LIMIT_DATA_DIR}",
    "EVAL_EVERY_STEPS": "${EVAL_EVERY_STEPS}",
    "EVAL_DATA": "${EVAL_DATA}",
    "EVAL_OUTPUT_DIR": "${EVAL_OUTPUT_DIR}",
    "EVAL_BENCHMARK": "${EVAL_BENCHMARK}",
    "EVAL_PROMPT_TYPE": "${EVAL_PROMPT_TYPE}",
    "EVAL_SEED": "${EVAL_SEED}",
    "EVAL_MAX_SAMPLES": "${EVAL_MAX_SAMPLES}",
    "EVAL_N_SAMPLING": "${EVAL_N_SAMPLING}",
    "EVAL_TEMPERATURE": "${EVAL_TEMPERATURE}",
    "EVAL_TOP_P": "${EVAL_TOP_P}",
    "EVAL_MAX_TOKENS": "${EVAL_MAX_TOKENS}",
}
Path("${OUTPUT_DIR}", "launch_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

TRAIN_DATA_OFFSETS="${TRAIN_DATA}.offsets"
if [[ ! -f "${TRAIN_DATA_OFFSETS}" || "${TRAIN_DATA}" -nt "${TRAIN_DATA_OFFSETS}" ]]; then
  echo "Building training data offset index: ${TRAIN_DATA_OFFSETS}"
  python scripts/build_jsonl_offsets.py \
    --input "${TRAIN_DATA}" \
    --output "${TRAIN_DATA_OFFSETS}"
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_OVERRIDE="--override=wandb_entity=${WANDB_ENTITY}"
else
  WANDB_ENTITY_OVERRIDE="--override=wandb_entity=null"
fi

if [[ -n "${WANDB_RUN_ID}" ]]; then
  WANDB_RUN_ID_OVERRIDE="--override=wandb_run_id=${WANDB_RUN_ID}"
else
  WANDB_RUN_ID_OVERRIDE="--override=wandb_run_id=null"
fi

if [[ -n "${WANDB_MODE}" ]]; then
  WANDB_MODE_OVERRIDE="--override=wandb_mode=${WANDB_MODE}"
else
  WANDB_MODE_OVERRIDE="--override=wandb_mode=null"
fi

LAUNCH_ARGS=(
  --num_processes "${NUM_GPUS}"
  --num_machines 1
  --mixed_precision "${MIXED_PRECISION}"
  --dynamo_backend no
)

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  if ! python -c "import deepspeed" >/dev/null 2>&1; then
    echo "DeepSpeed is not installed in this environment. Install it with: pip install deepspeed" >&2
    exit 1
  fi
  if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
    exit 1
  fi
  LAUNCH_ARGS+=(
    --use_deepspeed
    --deepspeed_config_file "${DEEPSPEED_CONFIG}"
  )
fi

accelerate launch \
  "${LAUNCH_ARGS[@]}" \
  -m offline_subtb.train \
  --config "${CONFIG}" \
  --override "model_name_or_path=${MODEL_NAME_OR_PATH}" \
  --override "train_data_path=${TRAIN_DATA}" \
  --override "output_dir=${OUTPUT_DIR}" \
  --override "max_steps=${MAX_STEPS}" \
  --override "num_epochs=${NUM_EPOCHS}" \
  --override "save_steps=${SAVE_STEPS}" \
  --override "logging_steps=${LOGGING_STEPS}" \
  --override "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --override "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  --override "learning_rate=${LEARNING_RATE}" \
  --override "max_pairs_per_trace=${MAX_PAIRS_PER_TRACE}" \
  --override "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}" \
  --override "finetune_mode=${FINETUNE_MODE}" \
  --override "lora_r=${LORA_R}" \
  --override "lora_alpha=${LORA_ALPHA}" \
  --override "lora_dropout=${LORA_DROPOUT}" \
  --override "lora_target_modules=${LORA_TARGET_MODULES}" \
  --override "wandb_project=${WANDB_PROJECT}" \
  --override "wandb_run_name=${WANDB_RUN_NAME}" \
  "${WANDB_ENTITY_OVERRIDE}" \
  "${WANDB_RUN_ID_OVERRIDE}" \
  "${WANDB_MODE_OVERRIDE}" \
  --override "limit_of_rlvr_dir=${LIMIT_OF_RLVR_DIR}" \
  --override "limit_data_dir=${LIMIT_DATA_DIR}" \
  --override "eval_every_steps=${EVAL_EVERY_STEPS}" \
  --override "eval_data_path=${EVAL_DATA}" \
  --override "eval_output_dir=${EVAL_OUTPUT_DIR}" \
  --override "eval_benchmark=${EVAL_BENCHMARK}" \
  --override "eval_prompt_type=${EVAL_PROMPT_TYPE}" \
  --override "eval_seed=${EVAL_SEED}" \
  --override "eval_max_samples=${EVAL_MAX_SAMPLES}" \
  --override "eval_n_sampling=${EVAL_N_SAMPLING}" \
  --override "eval_temperature=${EVAL_TEMPERATURE}" \
  --override "eval_top_p=${EVAL_TOP_P}" \
  --override "eval_max_tokens=${EVAL_MAX_TOKENS}"
