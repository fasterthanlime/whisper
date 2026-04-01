#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --python 3.11 \
  --with huggingface_hub \
  --with onnxruntime \
  --with librosa \
  --with numpy \
  scripts/phone_decode_zipa_sidecar.py "$@"
