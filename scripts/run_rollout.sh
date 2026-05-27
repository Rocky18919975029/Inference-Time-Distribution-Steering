#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

RAW_DATA="${RAW_DATA:-${ROOT_DIR}/data/train.parquet}"
MINI_DATA="${MINI_DATA:-${ROOT_DIR}/data/mini_train.parquet}"
SHARD_DIR="${SHARD_DIR:-${ROOT_DIR}/data/shards}"

LIMIT_OF_RLVR_DIR="${LIMIT_OF_RLVR_DIR:-${ROOT_DIR}/limit-of-RLVR}"
LIMIT_DATA_DIR="${LIMIT_DATA_DIR:-}"
BENCHMARK_PREFIX="${BENCHMARK_PREFIX:-mini_train_shard}"
OFFICIAL_OUTPUT_ROOT="${OFFICIAL_OUTPUT_ROOT:-${ROOT_DIR}/data/rollouts/mini_train/eval_results/global_step_0}"
LOG_DIR="${LOG_DIR:-${OFFICIAL_OUTPUT_ROOT}/logs}"
MERGED_ROLLOUT="${MERGED_ROLLOUT:-${OFFICIAL_OUTPUT_ROOT}/mini_train/test_qwen-boxed_-1_seed42_t0.6_s0_e-1.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/data/mini_train_subtb.jsonl}"
TRAIN_JSONL_WITH_REF="${TRAIN_JSONL_WITH_REF:-${ROOT_DIR}/data/mini_train_subtb_with_ref.jsonl}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
NUM_SHARDS="${NUM_SHARDS:-4}"
N_SAMPLING="${N_SAMPLING:-256}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.05}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen-boxed}"
COMPUTE_REF_LOGPROBS="${COMPUTE_REF_LOGPROBS:-1}"
REF_CONFIG="${REF_CONFIG:-${ROOT_DIR}/configs/qwen25_7b_math.yaml}"
REF_GPU="${REF_GPU:-0}"
REF_NUM_SHARDS="${REF_NUM_SHARDS:-${NUM_SHARDS}}"
REF_LOGPROB_DIR="${REF_LOGPROB_DIR:-${OFFICIAL_OUTPUT_ROOT}/ref_logprobs}"
ROLLOUT_LIVE_PROGRESS="${ROLLOUT_LIVE_PROGRESS:-1}"
ROLLOUT_PROGRESS_INTERVAL="${ROLLOUT_PROGRESS_INTERVAL:-5}"

MODE="${1:-all}"

abs_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return
  fi
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${ROOT_DIR}" "${path}"
  fi
}

RAW_DATA="$(abs_path "${RAW_DATA}")"
MINI_DATA="$(abs_path "${MINI_DATA}")"
SHARD_DIR="$(abs_path "${SHARD_DIR}")"
LIMIT_OF_RLVR_DIR="$(abs_path "${LIMIT_OF_RLVR_DIR}")"
if [[ -n "${LIMIT_DATA_DIR}" ]]; then
  LIMIT_DATA_DIR="$(abs_path "${LIMIT_DATA_DIR}")"
fi
OFFICIAL_OUTPUT_ROOT="$(abs_path "${OFFICIAL_OUTPUT_ROOT}")"
LOG_DIR="$(abs_path "${LOG_DIR}")"
MERGED_ROLLOUT="$(abs_path "${MERGED_ROLLOUT}")"
TRAIN_JSONL="$(abs_path "${TRAIN_JSONL}")"
TRAIN_JSONL_WITH_REF="$(abs_path "${TRAIN_JSONL_WITH_REF}")"
REF_CONFIG="$(abs_path "${REF_CONFIG}")"
REF_LOGPROB_DIR="$(abs_path "${REF_LOGPROB_DIR}")"

require_limit_of_rlvr() {
  if [[ -z "${LIMIT_OF_RLVR_DIR}" ]]; then
    echo "LIMIT_OF_RLVR_DIR must point to a local clone of https://github.com/LeapLabTHU/limit-of-RLVR" >&2
    exit 2
  fi
  if [[ ! -f "${LIMIT_OF_RLVR_DIR}/math/examples/math_eval/math_eval.py" ]]; then
    echo "Cannot find limit-of-RLVR math_eval.py under: ${LIMIT_OF_RLVR_DIR}" >&2
    exit 2
  fi
  if [[ -z "${LIMIT_DATA_DIR}" ]]; then
    LIMIT_DATA_DIR="${LIMIT_OF_RLVR_DIR}/math/examples/math_eval/data"
  fi
}

prepare() {
  python scripts/rollout_pipeline.py prepare \
    --input "${RAW_DATA}" \
    --output "${MINI_DATA}" \
    --shard-dir "${SHARD_DIR}" \
    --fraction "${SAMPLE_FRACTION}" \
    --seed "${SEED}" \
    --num-shards "${NUM_SHARDS}"
}

export_limit_inputs() {
  require_limit_of_rlvr
  python scripts/rollout_pipeline.py export-limit-of-rlvr-data \
    --shard-dir "${SHARD_DIR}" \
    --output-data-dir "${LIMIT_DATA_DIR}" \
    --benchmark-prefix "${BENCHMARK_PREFIX}" \
    --num-shards "${NUM_SHARDS}"
}

official_command_for_shard() {
  local shard_id="$1"
  local benchmark="${BENCHMARK_PREFIX}_${shard_id}"
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q python -u math_eval.py --model_name_or_path %q --data_names %q --data_dir %q --output_dir %q --split test --prompt_type %q --num_test_sample -1 --max_tokens_per_call %q --seed %q --temperature %q --n_sampling %q --top_p %q --start 0 --end -1 --use_vllm --save_outputs\n' \
    "${LIMIT_OF_RLVR_DIR}/math/examples/math_eval" \
    "${shard_id}" \
    "${MODEL_NAME_OR_PATH}" \
    "${benchmark}" \
    "${LIMIT_DATA_DIR}" \
    "${OFFICIAL_OUTPUT_ROOT}" \
    "${PROMPT_TYPE}" \
    "${MAX_TOKENS}" \
    "${SEED}" \
    "${TEMPERATURE}" \
    "${N_SAMPLING}" \
    "${TOP_P}"
}

print_rollout_commands() {
  require_limit_of_rlvr
  for shard_id in $(seq 0 "$((NUM_SHARDS - 1))"); do
    official_command_for_shard "${shard_id}"
  done
}

latest_progress_line() {
  local log_file="$1"
  local line=""
  if [[ -f "${log_file}" ]]; then
    line="$(tr '\r' '\n' < "${log_file}" | grep 'Processed prompts:' | tail -n 1 | perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g' || true)"
    if [[ -z "${line}" ]]; then
      line="$(tr '\r' '\n' < "${log_file}" | grep -E 'data:|remain samples|Epoch|[0-9]+/[0-9]+' | tail -n 1 | perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g' || true)"
    fi
  fi
  if [[ -z "${line}" ]]; then
    line="waiting for first progress update"
  fi
  printf '%s' "${line}"
}

pid_is_running() {
  local pid="$1"
  local stat=""
  stat="$(ps -p "${pid}" -o stat= 2>/dev/null || true)"
  [[ -n "${stat}" && "${stat}" != Z* ]]
}

any_rollout_running() {
  local pid
  for pid in "${pids[@]}"; do
    if pid_is_running "${pid}"; then
      return 0
    fi
  done
  return 1
}

print_rollout_progress() {
  local clear_screen="${1:-0}"
  local i shard_id gpu_id pid status line
  if [[ "${clear_screen}" == "1" ]]; then
    printf '\033[2J\033[H'
  fi
  printf 'Rollout progress by shard/GPU. Full logs: %s\n' "${LOG_DIR}"
  printf '%s\n' '============================================================'
  for i in "${!pids[@]}"; do
    shard_id="${shard_ids[$i]}"
    gpu_id="${gpu_ids[$i]}"
    pid="${pids[$i]}"
    if pid_is_running "${pid}"; then
      status="RUNNING"
    else
      status="FINISHED"
    fi
    line="$(latest_progress_line "${log_files[$i]}")"
    printf '[shard_%s | gpu_%s | %s] %s\n' "${shard_id}" "${gpu_id}" "${status}" "${line}"
  done
}

monitor_rollout_progress() {
  if [[ "${ROLLOUT_LIVE_PROGRESS}" != "1" ]]; then
    return
  fi
  while any_rollout_running; do
    print_rollout_progress 1
    sleep "${ROLLOUT_PROGRESS_INTERVAL}"
  done
  print_rollout_progress 1
}

compute_ref_logprobs_parallel() {
  local total_rows rows_per_shard shard_id start_index end_index output_part log_file failed
  mkdir -p "${REF_LOGPROB_DIR}" "$(dirname "${TRAIN_JSONL_WITH_REF}")"
  total_rows="$(wc -l < "${TRAIN_JSONL}" | tr -d ' ')"
  if [[ "${total_rows}" == "0" ]]; then
    echo "No rows found in ${TRAIN_JSONL}; cannot compute ref logprobs." >&2
    exit 1
  fi

  if [[ "${REF_NUM_SHARDS}" -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${REF_GPU}" python -m offline_subtb.compute_ref_logprobs \
      --config "${REF_CONFIG}" \
      --input "${TRAIN_JSONL}" \
      --output "${TRAIN_JSONL_WITH_REF}"
    return
  fi

  rows_per_shard="$(((total_rows + REF_NUM_SHARDS - 1) / REF_NUM_SHARDS))"
  pids=()
  ref_parts=()
  echo "Computing reference logprobs with ${REF_NUM_SHARDS} GPUs over ${total_rows} rows."
  for shard_id in $(seq 0 "$((REF_NUM_SHARDS - 1))"); do
    start_index="$((shard_id * rows_per_shard))"
    end_index="$((start_index + rows_per_shard))"
    if [[ "${start_index}" -ge "${total_rows}" ]]; then
      break
    fi
    if [[ "${end_index}" -gt "${total_rows}" ]]; then
      end_index="${total_rows}"
    fi
    output_part="${REF_LOGPROB_DIR}/part_${shard_id}.jsonl"
    log_file="${REF_LOGPROB_DIR}/part_${shard_id}.log"
    ref_parts+=("${output_part}")
    (
      export CUDA_VISIBLE_DEVICES="${shard_id}"
      python -m offline_subtb.compute_ref_logprobs \
        --config "${REF_CONFIG}" \
        --input "${TRAIN_JSONL}" \
        --output "${output_part}" \
        --start-index "${start_index}" \
        --end-index "${end_index}"
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "  shard ${shard_id} on GPU ${shard_id}: rows [${start_index}, ${end_index}) -> ${output_part}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one ref logprob shard failed. Check logs under: ${REF_LOGPROB_DIR}" >&2
    exit 1
  fi

  : > "${TRAIN_JSONL_WITH_REF}"
  for output_part in "${ref_parts[@]}"; do
    cat "${output_part}" >> "${TRAIN_JSONL_WITH_REF}"
  done
  echo "wrote $(wc -l < "${TRAIN_JSONL_WITH_REF}" | tr -d ' ') rows to ${TRAIN_JSONL_WITH_REF}"
}

rollout() {
  require_limit_of_rlvr
  mkdir -p "${OFFICIAL_OUTPUT_ROOT}" "${LOG_DIR}"
  pids=()
  shard_ids=()
  gpu_ids=()
  log_files=()
  for shard_id in $(seq 0 "$((NUM_SHARDS - 1))"); do
    log_file="${LOG_DIR}/shard_${shard_id}_gpu_${shard_id}.log"
    : > "${log_file}"
    (
      cd "${LIMIT_OF_RLVR_DIR}/math/examples/math_eval"
      export CUDA_VISIBLE_DEVICES="${shard_id}"
      python -u math_eval.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_names "${BENCHMARK_PREFIX}_${shard_id}" \
        --data_dir "${LIMIT_DATA_DIR}" \
        --output_dir "${OFFICIAL_OUTPUT_ROOT}" \
        --split test \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample -1 \
        --max_tokens_per_call "${MAX_TOKENS}" \
        --seed "${SEED}" \
        --temperature "${TEMPERATURE}" \
        --n_sampling "${N_SAMPLING}" \
        --top_p "${TOP_P}" \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    shard_ids+=("${shard_id}")
    gpu_ids+=("${shard_id}")
    log_files+=("${log_file}")
  done

  monitor_rollout_progress

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  print_rollout_progress 0
  if [[ "${failed}" != "0" ]]; then
    echo "At least one rollout shard failed. Check logs under: ${LOG_DIR}" >&2
    exit 1
  fi
}

merge_and_convert() {
  python scripts/rollout_pipeline.py merge-limit-of-rlvr-outputs \
    --output-root "${OFFICIAL_OUTPUT_ROOT}" \
    --output "${MERGED_ROLLOUT}" \
    --benchmark-prefix "${BENCHMARK_PREFIX}" \
    --num-shards "${NUM_SHARDS}" \
    --template "${PROMPT_TYPE}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}"

  python scripts/convert_limit_of_rlvr_outputs.py \
    --input "${MERGED_ROLLOUT}" \
    --output "${TRAIN_JSONL}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" \
    --n-sampling "${N_SAMPLING}"

  if [[ "${COMPUTE_REF_LOGPROBS}" == "1" ]]; then
    compute_ref_logprobs_parallel
  fi
}

case "${MODE}" in
  prepare-only)
    prepare
    ;;
  export-limit-inputs)
    export_limit_inputs
    ;;
  dry-run)
    prepare
    export_limit_inputs
    print_rollout_commands
    ;;
  rollout-only)
    rollout
    ;;
  merge-only)
    merge_and_convert
    ;;
  ref-logprobs-only)
    compute_ref_logprobs_parallel
    ;;
  all)
    prepare
    export_limit_inputs
    rollout
    merge_and_convert
    ;;
  *)
    echo "Usage: $0 [all|prepare-only|export-limit-inputs|dry-run|rollout-only|merge-only|ref-logprobs-only]" >&2
    exit 2
    ;;
esac
