# bee-vad MLX Review

Date: 2026-04-11

## Scope

Review of the streaming VAD path in:

- [`rust/bee-vad/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs)

## Main finding

This stack is also comparatively clean.

The only obvious inefficiency is that each chunk ends with an explicit sync and
scalar host extraction, which is acceptable because the caller wants one speech
probability value immediately.

## Evidence

`process_chunk()`:

- builds the input MLX array at [`lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:131)
- runs conv/STFT/encoder/LSTM/decoder through MLX ops at
  [`lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:133)
- explicitly calls `prob.eval()` at [`lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:187)
- extracts a scalar with `item::<f32>()` at [`lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:191)

That is a hard host-sync boundary.

## Assessment

For a VAD that produces one probability per 512-sample chunk, this is a
reasonable design.

The model is small, the output is scalar, and the next consumer likely wants
the value on the CPU immediately.

There is no obvious major MLX inefficiency here beyond the unavoidable
terminal synchronization.

## Recommendation

Do not spend time on this stack unless profiling proves VAD is unexpectedly hot.

If it ever becomes hot:

- consider reducing host allocations around the input buffer
- only then revisit whether scalar extraction can be amortized or fused

At the moment, it looks good enough.
