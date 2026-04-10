#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUST_LOG:-}" ]]; then
  export RUST_LOG=debug
fi

if [[ "${1:-}" == "--lldb" ]]; then
  shift
  lldb -- /Applications/bee.app/Contents/MacOS/bee "$@"
else
  /Applications/bee.app/Contents/MacOS/bee "$@"
fi
