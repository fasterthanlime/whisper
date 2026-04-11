# bee-g2p-charsiu-mlx

Native Rust/MLX inference for the [Charsiu G2P model](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers) (ByT5 encoder-decoder, word → IPA) on Apple Silicon.

## Model setup

The model weights must be in safetensors format. Two options:

### Option A: Use the pre-converted weights

```bash
# The converted model lives at:
~/bearcove/charsiu-g2p-mlx/model.safetensors
```

### Option B: Convert from HuggingFace yourself

```bash
# 1. Download the original pytorch checkpoint
pip install huggingface_hub
huggingface-cli download charsiu/g2p_multilingual_byT5_tiny_16_layers --local-dir /tmp/charsiu-g2p

# 2. Convert to safetensors
uv run rust/bee-g2p-charsiu-mlx/scripts/convert_to_safetensors.py /tmp/charsiu-g2p

# 3. Copy to the expected location
mkdir -p ~/bearcove/charsiu-g2p-mlx
cp /tmp/charsiu-g2p/model.safetensors ~/bearcove/charsiu-g2p-mlx/
```

## Quick start

```bash
# Plain G2P (sequential)
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx Facet hello beautiful

# Batched G2P
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx --batch 12 Facet hello beautiful world computer language

# Cross-attention ownership (shows IPA ↔ input byte alignment)
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx --cross-attention Facet
```

## Benchmarks

### Rust/MLX vs Python/PyTorch

Run both benchmarks on the same word list and compare:

```bash
# Generate a word list
WORDS="hello world beautiful computer language Facet algorithm pronunciation dictionary extraordinary"

# Rust/MLX — sequential
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx $WORDS

# Rust/MLX — batched (batch size 12, the product use case)
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx --batch 12 $WORDS

# Python/PyTorch (MPS) — sequential + batched
uv run rust/bee-g2p-charsiu-mlx/scripts/benchmark_python.py $WORDS
```

### Larger benchmark (500 words)

```bash
# Generate 500 words from /usr/share/dict/words
WORDS=$(shuf -n 500 /usr/share/dict/words | tr '\n' ' ')

# Rust/MLX batched
cargo run -p bee-g2p-charsiu-mlx --release -- ~/bearcove/charsiu-g2p-mlx --batch 64 $WORDS

# Python/PyTorch batched
uv run rust/bee-g2p-charsiu-mlx/scripts/benchmark_python.py $WORDS
```

### Reference numbers (M3 Max, 2026-04-11)

| Mode | Rust/MLX | Python/PyTorch (MPS) | Speedup |
|------|----------|---------------------|---------|
| Sequential (500 words) | 25.9 ms/word | 43.3 ms/word | 1.7× |
| Batch 64 (500 words) | 4.3 ms/word | 64.5 ms/word | 15× |
| Batch 12 (12 words) | 8.8 ms/word, 105 ms total | — | — |

## Library usage

```rust
use bee_g2p_charsiu_mlx::engine::G2pEngine;
use bee_g2p_charsiu_mlx::ownership::ByteSpan;

// Load model
let mut engine = G2pEngine::load(Path::new("~/bearcove/charsiu-g2p-mlx"))?;

// Plain G2P (cached)
let ipa = engine.g2p("hello", "eng-us")?;
assert_eq!(ipa, "ˈhɛɫoʊ");

// Batched G2P
let ipas = engine.g2p_batch(&["hello", "world"], "eng-us")?;

// Cross-attention probe with token-piece spans
let spans = vec![
    ByteSpan { label: "hel".into(), byte_start: 0, byte_end: 3 },
    ByteSpan { label: "lo".into(), byte_start: 3, byte_end: 5 },
];
let output = engine.probe("hello", "eng-us", &spans)?;
for span in &output.ownership {
    println!("{} -> {} (score={:.3})", span.label, span.ipa_text, span.avg_score);
}
```

## Tests

```bash
cargo test -p bee-g2p-charsiu-mlx -- --test-threads=1
```

The `--test-threads=1` is required because MLX Metal callbacks fail when multiple tests run in parallel.

## Architecture

- **model.rs** — Full T5 encoder-decoder with KV cache, batched generation, teacher-forced cross-attention extraction
- **engine.rs** — `G2pEngine`: model + word-level IPA cache + high-level API
- **ownership.rs** — Pure computation: attention matrix → ownership spans (no MLX dependency)
- **tokenize.rs** — ByT5 byte-level tokenization (UTF-8 byte + 3 = token ID)
- **config.rs** — Hardcoded T5 config for the Charsiu G2P model
- **load.rs** — Safetensors weight loading with HuggingFace → our key mapping
