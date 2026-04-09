# 001: Remove streaming top-k confidence

## Goal

Stop paying the `argpartition` / `argsort` / gather cost for every generated
token during live streaming revisions.

## Problem

The current streaming path computes token alternatives and confidence metrics on
every generated token:

- `rust/bee-qwen3-asr/src/generate.rs`
- `topk_confidence()`

This shows up directly in the hot path and is not required for every live
partial.

## Approach

- Split generation into two modes:
  - streaming: top-1 only
  - commit/final/focused debug: full top-k confidence
- Preserve the existing richer path for commit-time or diagnostics where it is
  actually useful
- Keep the output contract explicit so the rest of `bee-transcribe` knows when
  alternatives are absent

## Questions to answer

- Do we need concentration/margin at all during live streaming?
- If yes, can we compute a cheaper top-2-only metric instead of full top-k?

## Validation

- No regression on final transcript correctness
- Early partial behavior is unchanged or improved
- Instruments shows a clear drop in:
  - `topk_confidence`
  - `argpartition`
  - gather/sort-related MLX activity

## Current status

- Implemented split confidence modes:
  - streaming decode uses top-2-only confidence
  - commit-time refresh rescoring uses full top-k only for the text being committed
- The correction pipeline keeps full alternatives on committed chunks by
  rescoring the committable text prefix immediately before `commit()`.
- Removed the redundant finish-time full-confidence rescore; finalization now
  relies on the final full decode pass instead of paying a second MLX pass on
  the same sub-session.
- Corpus spot-check on the clean correction-free baseline:
  - final transcripts remained clean
  - two previously bad first partials improved in that run
- Follow-up benchmark on the clean correction-free corpus:
  - vs `37b823b`: `total_ms -1.3%`, `finish_ms -3.1%`, `max_feed_ms -2.0%`
  - vs `3aeabb4`: `total_ms -13.1%`, `finish_ms -38.5%`, `max_feed_ms -11.0%`

## Depends on

- 000 baseline
