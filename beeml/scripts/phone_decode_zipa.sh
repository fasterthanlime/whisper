#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --python 3.11 \
  --with huggingface_hub \
  --with onnxruntime \
  --with librosa \
  scripts/phone_decode_zipa.py "$@"
