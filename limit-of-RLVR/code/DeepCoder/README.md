## Installation 🎯

```bash
# Installing Python 3.10 Environment.
conda create -n rllm python=3.10 -y
conda activate rllm

# Installing RLLM dependencies.
cd rllm
pip install -e ./verl
pip install -e .
```

### Data

```bash
# Download datasets from GDrive, populates /data/*.json
python scripts/data/download_datasets.py
```

## Evaluation ⚖️

Our evaluation scripts automatically runs many replicas of vLLM. To run our evaluation scripts, run:

```bash
./scripts/eval/eval_model.sh --model [CHECKPOINT_PATH] --datasets [DATASET1] [DATASET2] --output-dir [OUTPUT_DIR] --n [N_PASSES] --tp [TENSOR_PARALLEL_SIZE] --max-length [MAX_CONTEXT_LENGTH] --seed [SEED]
```

Example:

```bash
./scripts/eval/eval_model.sh --model DeepCoder-14B-Preview --datasets test_livecodebench --output-dir ./eval/LCB-DeepCoder-14B-Preview-1 --tp 2 --max-length 65536 --n 1 --seed 101
```

For this example, generated responses are saved to `./eval/LCB-DeepCoder-14B-Preview-1/test_livecodebench.parquet` and detailed scores are saved to `./eval/LCB-DeepCoder-14B-Preview-1/results.json`.

More information, see `scripts/eval/README.md`.

## Acknowledgements

This framework is built on [rllm](https://github.com/agentica-project/rllm). To accelerate the computation, we rewrote it using the `vllm` framework.
