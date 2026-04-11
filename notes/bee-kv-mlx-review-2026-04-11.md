# bee-kv MLX Review

Date: 2026-04-11

## Scope

Review of the current Qwen3-ASR draft flow in:

- [`rust/bee-kv/src/decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs)
- [`rust/bee-kv/src/alignment.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/alignment.rs)

This note focuses on MLX/runtime inefficiencies, not product design.

## Main finding

`bee-kv` still redoes large amounts of acoustic work on overlapping windows.

That is the dominant MLX-side inefficiency in this draft. It is more important
than decoder-KV micro-optimization inside this stack.

## Evidence

In [`decode_sliding_window_bridge_replay()`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:35),
each window is built from:

- committed / unresolved keep region
- bridge region
- rollback region

See the window geometry at:

- [`decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:141)

Each loop iteration then calls
[`decode_chunk_followup_step()`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:516),
which:

- extracts mel again at [`decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:533)
- rebuilds the mel array at [`decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:536)
- runs `model.encode_audio(&mel)` again at [`decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:537)

Because adjacent windows overlap on purpose, the same audio can be re-encoded
multiple times.

## Secondary findings

### Prompt replay still expands text-side work

`bee-kv` appends carried bridge tokens back into the next prompt at
[`decode.rs`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:555).

That is architecturally fine for the draft, but it means the path still pays
for repeated text-side prefill on replayed material.

### Alignment work is intentionally expensive

Rollback resolution calls ZIPA alignment for each chunk boundary via:

- [`resolve_window_rollback()`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:319)
- [`build_transcript_alignment(...)`](/Users/amos/bearcove/bee/rust/bee-kv/src/decode.rs:332)

That is not a bug, but it means end-to-end timings from `bee-kv` include much
more than pure ASR decode.

## Assessment

For this draft, the key inefficiency is:

- repeated full acoustic encoding of overlapping windows

not:

- decoder cache append cost
- tiny decode-step tensor allocations

Those other costs still exist through the underlying `bee-qwen3-asr` stack, but
the acoustic overlap is the biggest structural issue in `bee-kv` itself.

## Recommendation

If this draft is revived, the first major optimization should be one of:

- incremental / reusable audio encoder state
- reuse of encoded audio features across overlapping windows
- window geometry that minimizes redundant encoder work

Do not spend primary effort on decoder-KV tuning in `bee-kv` before addressing
overlapping acoustic recomputation.
