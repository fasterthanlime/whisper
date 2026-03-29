#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

find_default_model_dir() {
  local cache_root="${HOME}/Library/Caches/qwen3-asr"
  local preferred="${cache_root}/Alkd--qwen3-asr-gguf--qwen3_asr_0_6b_q4_k_gguf"
  if [[ -d "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi

  local first=""
  while IFS= read -r candidate; do
    first="${candidate}"
    break
  done < <(compgen -G "${cache_root}/Alkd--qwen3-asr-gguf--qwen3_asr_*_gguf" || true)

  if [[ -n "${first}" && -d "${first}" ]]; then
    printf '%s\n' "${first}"
    return 0
  fi
  return 1
}

MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="$(find_default_model_dir || true)"
fi

if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  cat >&2 <<'EOF'
error: model directory not found.
Set QWEN3_ASR_MODEL_DIR, for example:
  export QWEN3_ASR_MODEL_DIR="$HOME/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_0_6b_q4_k_gguf"
EOF
  exit 1
fi

cd "${REPO_ROOT}"
echo "[asr-pipeline] using model dir: ${MODEL_DIR}"

echo "[asr-pipeline] running ignored full streaming integration suite..."
QWEN3_ASR_MODEL_DIR="${MODEL_DIR}" \
  cargo nextest run \
  --test streaming_integration \
  --run-ignored ignored-only \
  --status-level all

echo "[asr-pipeline] running dtype mixed regression suite..."
cargo nextest run \
  --test dtype_mixed_regressions \
  --status-level all

echo "[asr-pipeline] all checks passed"
