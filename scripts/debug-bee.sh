#!/usr/bin/env bash
set -euo pipefail

export RUST_LOG=debug

if [[ "${1:-}" == "--lldb" ]]; then
  shift
  lldb -- /Applications/bee.app/Contents/MacOS/bee "$@"
else
  /Applications/bee.app/Contents/MacOS/bee "$@"
fi
