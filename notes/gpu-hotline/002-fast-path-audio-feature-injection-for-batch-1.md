# 002: Fast path audio feature injection for batch size 1

## Goal

Reduce MLX graph churn and GPU copies in prompt/audio embedding injection for
the single-stream dictation case.

## Problem

The current audio embedding injection path in
`rust/bee-qwen3-asr/src/model.rs` uses:

- `cumsum`
- gather/index
- `expand_dims`
- `concatenate_axis`
- `where`

This is general, but the app runs batch size 1 almost all the time.

## Approach

- Add a dedicated batch-1 path for `inject_audio_features()`
- Avoid building `parts`, batch concatenation, and unnecessary batch-axis
  expansion when `B == 1`
- Investigate whether the audio placeholder region can be filled by slice update
  instead of mask + `where`

## Non-goals

- Changing prompt semantics
- Cross-chunk decoder cache reuse

## Validation

- Same logits / outputs as the current implementation on batch size 1
- No correctness regression on the fixed corpus
- Lower weight in:
  - `mlx::core::copy_gpu`
  - gather/indexing
  - concatenate-related ops during prefill

## Depends on

- 000 baseline
