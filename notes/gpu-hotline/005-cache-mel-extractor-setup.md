# 005: Cache mel extractor setup

## Goal

Remove repeated CPU-side setup work inside the streaming decode loop.

## Problem

`rust/bee-qwen3-asr/src/mel.rs` currently rebuilds immutable setup on every
extract:

- Hann window
- FFT planner / FFT plan

This is not the biggest item in the profile, but it is low-risk cleanup on the
hot path.

## Approach

- Store the Hann window inside `MelExtractor`
- Store a reusable FFT plan inside `MelExtractor`
- Keep output bit-identical to the current implementation

## Validation

- Same mel output on unit tests / fixture audio
- No transcript correctness regression
- Small but measurable reduction in CPU time ahead of MLX work

## Depends on

- 000 baseline
