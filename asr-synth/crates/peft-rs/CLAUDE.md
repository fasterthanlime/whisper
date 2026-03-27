# peft-rs - Parameter-Efficient Fine-Tuning Library

## Overview

Rust implementation of PEFT (Parameter-Efficient Fine-Tuning) methods for LLMs. This is the **foundation crate** - other crates in the workspace depend on it.

## Architecture

```
src/
├── lib.rs           # Public API: re-exports all adapters and traits
├── adapters/        # PEFT method implementations
│   ├── mod.rs       # Adapter module exports
│   ├── lora.rs      # LoRA (Low-Rank Adaptation)
│   ├── dora.rs      # DoRA (Weight-Decomposed LoRA)
│   ├── adalora.rs   # AdaLoRA (Adaptive LoRA with SVD)
│   ├── ia3.rs       # IA³ (Inhibiting and Amplifying)
│   ├── loha.rs      # LoHa (Hadamard product)
│   ├── lokr.rs      # LoKr (Kronecker product)
│   ├── oft.rs       # OFT (Orthogonal Fine-Tuning)
│   ├── boft.rs      # BOFT (Butterfly OFT)
│   ├── vera.rs      # VeRA (Vector-based Random Matrix)
│   ├── prefix.rs    # Prefix Tuning
│   └── prompt.rs    # Prompt Tuning
├── traits.rs        # Core traits: Adapter, AdapterConfig
├── config.rs        # Configuration structures
├── model.rs         # Model integration utilities
├── registry.rs      # Adapter registry for dynamic loading
├── training.rs      # Training loop utilities
├── inference.rs     # Inference utilities
├── io.rs            # Safetensors I/O for adapter weights
└── error.rs         # Error types
```

## Key Traits

```rust
// Core adapter trait - all PEFT methods implement this
pub trait Adapter {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn weights(&self) -> Vec<&Tensor>;  // Required by qlora-rs
    fn config(&self) -> &dyn AdapterConfig;
}

// Configuration trait
pub trait AdapterConfig {
    fn adapter_type(&self) -> &str;
    fn rank(&self) -> usize;
}
```

## Downstream Dependencies

**qlora-rs** depends on:
- `Adapter` trait and its `weights()` method
- `LoraAdapter` struct for QLoRA inference
- `LoraConfig` for configuration

**axolotl-rs** depends on (when `peft` feature enabled):
- All adapter types for YAML-driven configuration
- `AdapterRegistry` for dynamic adapter loading

## Development Commands

```bash
# Check
cargo check -p peft-rs

# Test (unit tests are in src/ files)
cargo test -p peft-rs

# Test with CUDA
cargo test -p peft-rs --features cuda

# Benchmarks
cargo bench -p peft-rs

# Docs
cargo doc -p peft-rs --open

# Clippy
cargo clippy -p peft-rs -- -W clippy::pedantic
```

## Critical Code Paths

### LoRA Forward Pass (most used)
`src/adapters/lora.rs` - The `forward()` method is performance-critical:
```rust
// y = x @ W + (x @ A @ B) * scale
// Ensure this remains efficient - qlora-rs depends on it
```

### Adapter Weights Access
`weights()` method must return all trainable parameters. qlora-rs uses this for quantization.

### Safetensors I/O
`src/io.rs` - Must maintain compatibility with HuggingFace PEFT format for interoperability.

## Breaking Change Checklist

Before modifying public API:

1. [ ] Check qlora-rs still compiles: `cargo check -p qlora-rs`
2. [ ] Check axolotl-rs with feature: `cargo check -p axolotl-rs --features peft`
3. [ ] Update CHANGELOG.md
4. [ ] Bump version if needed (semver)

## Testing Strategy

- Unit tests: In each source file (`#[cfg(test)]` modules)
- Property tests: Using `proptest` for adapter math correctness
- Integration: Adapter serialization round-trips

## 1.0 Checklist

- [x] Core LoRA implementation
- [x] DoRA, AdaLoRA variants
- [x] Alternative methods (IA³, LoHa, LoKr, OFT, BOFT, VeRA)
- [x] Prefix/Prompt tuning
- [x] Safetensors I/O
- [x] Published to crates.io
- [ ] 100% doc coverage on public items
- [ ] Property-based tests for all adapters
- [ ] Benchmark suite
- [ ] HuggingFace format compatibility tests
- [x] Examples directory

## Common Issues

### "weights() returns empty"
Ensure adapter is initialized with actual parameters, not just config.

### Dimension mismatch in forward
Check that input tensor shape matches adapter's expected `in_features`.

### Safetensors loading fails
Verify tensor names match HuggingFace PEFT naming convention.
