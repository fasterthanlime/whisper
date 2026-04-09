# 003: Tune MLX cache pressure with real measurements

## Goal

Find an MLX cache policy that avoids the 6–8 GB runaway case without paying the
current per-step cache-clear tax.

## Problem

Current behavior:

- engine-level MLX cache limit is set to 2 GB
- streaming decode also calls `mlx_clear_cache()` after every step

The per-step clear is expensive, but removing it previously caused system
memory usage to balloon badly.

## Approach

- Keep `mlx_clear_cache()` on the radar, but do not rip it out blindly
- Measure lower cache caps first:
  - 512 MB
  - 768 MB
  - 1 GB
- Test policies such as:
  - clear only on commit
  - clear only on finalize
  - clear only after crossing a sampled memory threshold

## Questions to answer

- Is the current 2 GB cap ineffective, or just too high for the total resident
  footprint of model + aligner + transient buffers?
- Does lowering the cap reduce runaway growth enough that per-step clear can be
  relaxed?

## Validation

- No runaway memory growth on long dictation runs
- Lower cost attributed to `mlx_clear_cache`
- No regression in final transcript correctness

## Depends on

- 000 baseline
