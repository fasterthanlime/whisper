# 006: Audit and minimize MLX op-graph churn

## Goal

Systematically trim avoidable MLX op construction in the Qwen3-ASR inference
path beyond the headline hotspots.

## Targets

- Repeated mask construction in `rust/bee-qwen3-asr/src/decoder.rs`
- Repeated position-id construction in `rust/bee-qwen3-asr/src/generate.rs`
- Repeated `expand_dims` / `broadcast_to` / `concatenate_axis` chains where a
  batch-1 or shape-specialized path would be simpler
- Any unnecessary host scalar extraction that forces synchronization

## Scope

- Stay inside:
  - `rust/bee-qwen3-asr`
  - vendored `mlx-rs`
- Focus on operations that actually show up in Instruments

## Approach

- Compare the current op graph against the heaviest Instruments stacks
- Remove or specialize only what has visible cost
- Avoid speculative cleanup of code that never shows up in profiles

## Validation

- Every change is justified by a before/after profile diff
- No correctness regression on the fixed corpus

## Depends on

- 000 baseline
- preferably after 001 and 002
