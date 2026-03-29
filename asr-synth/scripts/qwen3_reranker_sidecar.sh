#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --python 3.11 \
  --with torch \
  --with transformers \
  --with sentencepiece \
  python scripts/qwen3_reranker_sidecar.py
