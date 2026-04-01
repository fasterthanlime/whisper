#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run \
  --with torch \
  --with transformers \
  --with sentencepiece \
  training/train_byt5.py "$@"
