# Inference-Time Distribution Steering

This project implements Top-k Low-Rank Steering for frozen-base decoding.
The base language model is never updated. Training learns only a lightweight
state projector, token steering basis, and optional value head.

The rollout and verification workflow is copied from the previous
`limit-of-RLVR`-based project. Existing rollout JSONL files can be reused
directly as training data.

## Method

At each token position, the frozen base model produces hidden state `h_t` and
base logits `z_t^0`. The steering policy restricts actions to the base top-k
tokens and adds a low-rank score:

```text
pi(a | x, y_<t) = softmax_{a in TopK(z_t^0)} [
  z_t^0(a) + alpha * B[a]^T f_psi(h_t)
]
```

Implemented objectives:

- `tb`: VarGrad trajectory balance
- `grpo`: group-relative clipped policy optimization
- `ac`: actor-critic with a learned value baseline

The recommended first experiment is `tb`.

## Layout

```text
src/itds/                 New Top-k low-rank steering implementation
src/offline_subtb/        Copied compatibility utilities for rollout/eval conversion
scripts/run_itds_train.sh Main ITDS training launcher
scripts/run_rollout.sh    Copied limit-of-RLVR rollout pipeline
scripts/eval_checkpoints.py
limit-of-RLVR/            Copied verifier/eval dependency
configs/itds_qwen25_7b_math.yaml
```

## Install

```bash
conda activate simplerl
pip install -r requirements.txt
pip install -e .
```

## Train

Reuse an existing verified rollout file:

```bash
USE_DEEPSPEED=0 \
WANDB_PROJECT=itds \
WANDB_RUN_NAME=itds_tb_topk64_r32 \
OBJECTIVE=tb \
TRAIN_DATA=$PWD/data/full_train_subtb_with_ref.jsonl \
OUTPUT_DIR=$PWD/outputs/itds_tb_topk64_r32 \
NUM_GPUS=4 \
MAX_STEPS=1000 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=2 \
LEARNING_RATE=0.0001 \
TOP_K=64 \
RANK=32 \
BETA=0.1 \
bash scripts/run_itds_train.sh
```

The dataloader batches responses by `problem_id`, so TB/GRPO group statistics
are computed over multiple responses for the same prompt.

## Reuse Rollout And Verify

Rollout generation and exact-answer verification are delegated to the copied
`limit-of-RLVR` pipeline:

```bash
bash scripts/run_rollout.sh
```

For CPU-safe command inspection:

```bash
COMPUTE_REF_LOGPROBS=0 bash scripts/run_rollout.sh dry-run
```
