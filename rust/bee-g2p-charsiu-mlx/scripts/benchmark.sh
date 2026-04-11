#!/usr/bin/env bash
# Comparative benchmark: Rust/MLX vs Python/PyTorch
#
# Usage: ./scripts/benchmark.sh [num_words]
#   num_words: number of words to benchmark (default: 100)
#
# Requires:
#   - Model at ~/bearcove/charsiu-g2p-mlx/model.safetensors
#   - uv (for Python script)
#   - cargo (for Rust binary)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$HOME/bearcove/charsiu-g2p-mlx"
NUM_WORDS="${1:-100}"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: Model not found at $MODEL_DIR/model.safetensors"
    echo "See README.md for setup instructions."
    exit 1
fi

# Generate word list from dictionary
if [ -f /usr/share/dict/words ]; then
    WORDS=$(grep -E '^[a-zA-Z]{3,12}$' /usr/share/dict/words | shuf -n "$NUM_WORDS" | tr '\n' ' ')
else
    echo "WARNING: /usr/share/dict/words not found, using fallback word list"
    WORDS="hello world beautiful computer language algorithm pronunciation dictionary extraordinary Facet"
fi

echo "=== Benchmark: $NUM_WORDS words ==="
echo ""

# Build Rust binary first (don't count compilation time)
echo "Building Rust binary..."
cargo build -p bee-g2p-charsiu-mlx --release -q
echo ""

echo "=== Rust/MLX — Sequential ==="
cargo run -p bee-g2p-charsiu-mlx --release -q -- "$MODEL_DIR" $WORDS 2>&1 | grep -E '(ms/word|total)'
echo ""

for BS in 12 64; do
    echo "=== Rust/MLX — Batch $BS ==="
    cargo run -p bee-g2p-charsiu-mlx --release -q -- "$MODEL_DIR" --batch "$BS" $WORDS 2>&1 | grep -E '(ms/word|total|batch of)'
    echo ""
done

echo "=== Python/PyTorch (MPS) ==="
uv run "$SCRIPT_DIR/benchmark_python.py" $WORDS 2>&1 | grep -E '(ms/word|total|Device)'
echo ""
