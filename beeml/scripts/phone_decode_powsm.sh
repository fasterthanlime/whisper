#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --python 3.11 \
  --with torch \
  --with torchaudio \
  --with soundfile \
  --with huggingface_hub \
  --with git+https://github.com/espnet/espnet.git \
  scripts/phone_decode_powsm.py "$@"
