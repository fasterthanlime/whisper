#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:?missing model id}"
ADAPTER_PATH="${2:-}"

uv run --python 3.11 \
  --with mlx-lm==0.31.1 \
  python scripts/prototype_reranker_sidecar.py "$MODEL_ID" "$ADAPTER_PATH"
