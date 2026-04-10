#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  scripts/compare-cut-modes.sh <audio.wav> [out_dir]

Examples:
  scripts/compare-cut-modes.sh /Users/amos/bearcove/bee/.artifacts/repros/9BE3E21F.wav
  scripts/compare-cut-modes.sh /tmp/test.wav /tmp/bee-cut-compare
EOF
  exit 1
fi

WAV_PATH="$1"
OUT_DIR="${2:-/tmp/bee-cut-compare-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "$WAV_PATH" ]]; then
  echo "error: WAV not found: $WAV_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Required by this repo for MLX/HF/etc. env setup.
if command -v direnv >/dev/null 2>&1; then
  eval "$(direnv export bash)"
fi

MODES=("zipa" "qwen3" "uncut")

echo "WAV: $WAV_PATH"
echo "OUT: $OUT_DIR"
echo

for mode in "${MODES[@]}"; do
  log_file="$OUT_DIR/${mode}.log"
  echo "=== Running mode=$mode ==="

  set +e
  (
    export BEE_DISABLE_CORRECTION=1
    export BEE_ROTATION_CUT_MODE="$mode"
    export RUST_LOG="bee_transcribe::session=info,bee_transcribe::decode_session=info"
    # For uncut mode, process as a single feed so logs are concise.
    if [[ "$mode" == "uncut" ]]; then
      export BEE_CHUNK_DURATION=600
    else
      unset BEE_CHUNK_DURATION || true
    fi
    cargo run -q -p bee-transcribe --bin transcribe -- "$WAV_PATH"
  ) 2>&1 | tee "$log_file"
  cmd_status=${PIPESTATUS[0]}
  set -e

  status="ok"
  if [[ "$cmd_status" -ne 0 ]] || rg -n "panicked at|thread 'main' panicked" "$log_file" >/dev/null 2>&1; then
    status="panic"
  fi

  rotations="$( (rg -n "commit: rotation" "$log_file" || true) | wc -l | tr -d ' ')"
  final_text="$( (rg -n '^  text: ' "$log_file" || true) | tail -n 1 | sed -E "s/^[^:]*: //")"
  if [[ -z "$final_text" ]]; then
    final_text="<none>"
  fi

  echo "mode=$mode status=$status rotations=$rotations"
  echo "final_text=$final_text"
  echo "log=$log_file"
  echo
done

echo "Done."
