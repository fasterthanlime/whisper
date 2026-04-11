#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/bee-roll-traced-auto.sh <wav>" >&2
  exit 1
fi

cd /Users/amos/bearcove/bee
eval "$(direnv export bash)"

export BEE_ZIPA_BUNDLE_DIR="${BEE_ZIPA_BUNDLE_DIR:-$HOME/bearcove/zipa-mlx-hf}"
export BEE_G2P_CHARSIU_MODEL_DIR="${BEE_G2P_CHARSIU_MODEL_DIR:-/tmp/charsiu-g2p}"
export RUST_LOG="${RUST_LOG:-bee_roll::utterance=trace}"

exec cargo run --release -p bee-roll -- --auto "$1"
