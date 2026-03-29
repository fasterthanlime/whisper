#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --python 3.11 \
  --with allophant \
  --with soundfile \
  --with torchaudio \
  scripts/phone_decode_allophant.py "$@"
