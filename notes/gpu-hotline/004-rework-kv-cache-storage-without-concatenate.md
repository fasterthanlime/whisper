# 004: Rework KV cache storage without concatenate

## Goal

Eliminate repeated full-buffer GPU copies inside a single decode call by
replacing concatenating K/V cache growth with appendable storage.

## Problem

`rust/bee-qwen3-asr/src/decoder.rs` currently grows the per-layer KV cache via:

- `concatenate_axis(prev_k, key)`
- `concatenate_axis(prev_v, value)`

for every generated token step.

That is the strongest MLX-side copy smell in the current inference stack.

## Approach options

1. Preallocated cache buffers with explicit write offsets
2. Chunked/page-based cache segments with a final logical view
3. Borrow/adapt an appendable cache pattern from upstream/vLLM/transformers

## Requirements

- Preserve current correctness within a single `prefill_and_decode()` call
- Do not assume cross-chunk cache reuse is valid
- Keep the API simple enough that the rest of the decoder stack stays readable

## Validation

- Same output as current implementation on the fixed corpus
- Lower weight in:
  - `mlx::core::copy_gpu`
  - `mlx::core::concatenate_gpu`
- Lower mlx-worker time per decode step

## Depends on

- 000 baseline
