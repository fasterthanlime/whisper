# bee-g2p-charsiu-mlx Review

Date: 2026-04-11

## Scope

Review of the MLX-native Charsiu ByT5 port in:

- [`rust/bee-g2p-charsiu-mlx/src/model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs)
- [`rust/bee-g2p-charsiu-mlx/src/tokenize.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/tokenize.rs)

## Main finding

The most important performance lever in this stack is batching.

The biggest code smell remaining is that the advertised preallocated KV cache
still falls back to concatenation and therefore is not a real append-in-place
cache.

## Evidence

### Fake preallocation

`KvCache::preallocated()` exists at
[`model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:41),
but `update()` still does:

- slice filled prefix
- `concatenate_axis(...)`

at:

- [`model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:73)
- [`model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:75)

So the current KV path still copies accumulated cache state every step.

### Position bias was worth caching

The decoder now caches the full position bias once and indexes rows from it in:

- [`step_cached()`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:813)
- cache initialization at [`model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:823)

That is the right direction and is already more defensible than further small
KV tweaks.

### Batch path is the real operating mode

The batch tokenization path is explicit in:

- [`encode_batch_to_array()`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/tokenize.rs:39)

and the model has a separate batched generation path in:

- [`generate_batch()`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:998)

Recent local measurements indicate batching produces the meaningful latency
improvement for the actual use case.

## Assessment

For this model:

- batching matters more than KV
- position-bias caching is a reasonable small win
- the decoder is small enough that the unfinished KV prealloc path is not the
  main practical limiter right now

That said, the current prealloc code is misleading and should not be left in an
ambiguous half-finished state.

## Recommendation

Choose one:

1. finish real append-in-place KV writes
2. or remove the fake preallocated path and keep the code honest

In parallel:

- continue optimizing for microbatch latency, not giant-batch throughput
- keep using the batched path for real product measurements

For the current product constraints, batching policy and cache strategy around
`stable/carry/preview` matter more than further model-internal micro-tuning.
