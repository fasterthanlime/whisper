# bee-zipa-mlx Review

Date: 2026-04-11

## Scope

Review of the ZIPA MLX inference path in:

- [`rust/bee-zipa-mlx/src/infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs)

## Main finding

This stack is comparatively clean.

The main inefficiency is not internal graph churn but terminal host readback at
the argmax stage, which is acceptable if ZIPA output is consumed on the CPU
immediately afterward.

## Evidence

The main forward path is:

- features to MLX array at [`infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:167)
- frontend/encoder at [`infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:175)
- CTC head at [`infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:190)

The output path then does:

- `argmax_axis(...)` at [`infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:223)
- `as_slice::<u32>()` at [`infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:225)

That is a synchronization point and device-to-host readback.

## Assessment

For current usage, this is probably fine.

ZIPA is being used as an analysis / alignment-support model, and the next stage
usually wants token ids or decoded tokens on the host anyway.

There is no obvious equivalent here to the heavier graph churn seen in Qwen3-ASR.

## Recommendation

Leave this stack alone unless:

- ZIPA becomes a tighter inner-loop dependency
- you want to fuse downstream operations on-device

If either happens, revisit the output boundary and keep results device-resident
longer before decoding on the host.
