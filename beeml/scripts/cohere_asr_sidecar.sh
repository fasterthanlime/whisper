#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --python 3.11 \
  --with torch \
  --with transformers \
  --with soundfile \
  --with librosa \
  --with sentencepiece \
  scripts/cohere_asr_sidecar.py "$@"
