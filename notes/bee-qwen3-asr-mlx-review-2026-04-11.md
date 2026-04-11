# bee-qwen3-asr MLX Review

Date: 2026-04-11

## Scope

Review of the MLX inference path in:

- [`rust/bee-qwen3-asr/src/generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs)
- [`rust/bee-qwen3-asr/src/decoder.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/decoder.rs)
- [`rust/bee-qwen3-asr/src/encoder.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/encoder.rs)
- [`rust/bee-qwen3-asr/src/model.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/model.rs)

Note: this stack is no longer the main alignment path, but parts of its runtime
behavior still matter because other drafts and tools build on it.

## Main finding

The biggest remaining inefficiency is graph/setup churn:

- rebuilding prompt tensors
- rebuilding position-id tensors
- constructing masks on the host with `Vec<f32>`

The decoder KV cache is already in much better shape than before. The remaining
cost is now the repeated surrounding scaffolding.

## Evidence

### Prompt and position tensor churn

[`prefill_and_decode()`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:225)
rebuilds all of these every call:

- `input_ids` at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:235)
- `positions` host vector at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:237)
- `pos_arr` at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:238)
- `position_ids` broadcast at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:239)

Then each autoregressive step rebuilds:

- `next_ids` at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:283)
- `pos_arr` at [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:284)

For streaming decode, this overhead is paid many times.

### Confidence extraction still forces work

Streaming confidence still does:

- `argpartition` in [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:405)
- explicit `eval()` in [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:409)
- scalar `item()` extraction in [`generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:412)

This is lighter than the old full top-k path, but it is still not free.

### Mask construction is host-side and O(L²)

Decoder causal masks are built with nested Rust loops in:

- [`create_causal_mask()`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/decoder.rs:539)
- [`create_causal_mask_with_prefix()`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/decoder.rs:552)

Encoder window masks do the same in:

- [`create_windowed_mask()`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/encoder.rs:638)

That means shape-derived masks are still being materialized on the host and
copied into MLX arrays instead of being reused or built more cheaply.

## What is already in decent shape

The decoder KV cache itself is much healthier now.

[`KVCache::update()`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/decoder.rs:39)
uses append-by-write into capacity-managed buffers rather than repeated whole
tensor concatenation.

That is no longer the primary bottleneck to look at.

## Recommendation

The highest-value next improvements in this stack are:

1. cache / reuse position-id tensors by shape
2. cache masks by shape instead of rebuilding them from host vectors
3. add harder batch-1 specializations around prompt assembly and step tensors
4. continue trimming confidence-path synchronizations only if profiles show
   they still matter

If this stack matters again, prioritize surrounding graph churn over more KV
cache work.
