# 007: Delete dead KV experiment code

## Goal

Remove stale code from abandoned decoder-cache reuse experiments so the live
streaming path is easier to reason about and profile.

## Known candidates

- `build_followup_prompt()` in `rust/bee-qwen3-asr/src/generate.rs`
- `KVCache::truncate()` in `rust/bee-qwen3-asr/src/decoder.rs` if it remains
  unused after current work settles

## Why

- These paths suggest alternate architectures that are not actually live
- They make review and future optimization work noisier than necessary

## Approach

- Confirm local non-usage before deletion
- Remove dead code, or explicitly quarantine it behind comments / feature gates
  if it is being kept for future experiments

## Validation

- Build passes
- No behavior change
- The active streaming path becomes easier to trace end-to-end

## Depends on

- Can happen anytime after the corresponding experiment is truly abandoned
