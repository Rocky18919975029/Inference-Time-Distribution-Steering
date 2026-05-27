#!/usr/bin/env bash
set -euo pipefail

python -m offline_subtb.train --config configs/qwen25_7b_simplerl.yaml
