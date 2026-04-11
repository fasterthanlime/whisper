# MLX Inference Device Review

Date: 2026-04-11

## Question

Do the MLX-backed inference crates in this workspace run inference on GPU or CPU?

## Conclusion

They are written to run inference on the GPU by default.

The main reason is in [`mlx-rs/mlx-rs/src/stream.rs`](/Users/amos/bearcove/bee/mlx-rs/mlx-rs/src/stream.rs:41): `StreamOrDevice::default()` uses the default MLX device, and that default is documented there as `Device::gpu()` unless changed with `Device::set_default()`.

I searched the inference crates for explicit CPU selection and did not find any use of:

- `Device::set_default(...)`
- `Device::cpu()`
- `StreamOrDevice::cpu()`
- `with_new_default_stream(...)`

in the actual inference code paths.

## Scope Reviewed

- `rust/bee-qwen3-asr`
- `rust/bee-zipa-mlx`
- `rust/bee-vad`
- `rust/bee-g2p-charsiu-mlx`
- `rust/bee-transcribe` as the orchestrator that loads and calls the above

## Evidence

### MLX default device behavior

[`mlx-rs/mlx-rs/src/stream.rs`](/Users/amos/bearcove/bee/mlx-rs/mlx-rs/src/stream.rs:41) states that omitted stream/device parameters use the default device, and [`mlx-rs/mlx-rs/src/stream.rs`](/Users/amos/bearcove/bee/mlx-rs/mlx-rs/src/stream.rs:79) states that this is `Device::gpu()` unless overridden.

### Qwen3 ASR

[`rust/bee-qwen3-asr/src/model.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/model.rs:67) runs `prefill` and `step` through MLX module forwards and ops without supplying a CPU stream/device.

[`rust/bee-qwen3-asr/src/generate.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs:97) performs autoregressive decoding with MLX arrays and `model.step(...)`, again without a CPU override.

[`rust/bee-qwen3-asr/src/forced_aligner.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/forced_aligner.rs:108) encodes audio, runs the decoder, and applies the LM head entirely through MLX operations with no CPU device selection.

[`rust/bee-qwen3-asr/src/decoder.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/decoder.rs:39) updates the KV cache with MLX arrays and evaluates those writes in place, still without changing the device.

### ZIPA

[`rust/bee-zipa-mlx/src/infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:175) runs the frontend and encoder stages through MLX forwards and tensor ops.

[`rust/bee-zipa-mlx/src/infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:190) applies the CTC head with MLX and only reads token ids back to the host after inference.

### VAD

[`rust/bee-vad/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:113) builds the VAD inference path out of MLX `conv1d`, `matmul`, elementwise ops, and `sigmoid`, with no explicit CPU stream/device.

[`rust/bee-vad/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:43) also describes the arrays as heap-allocated Metal buffers.

### Charsiu G2P

[`rust/bee-g2p-charsiu-mlx/src/model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:682) generates from encoder output and evaluates token logits through MLX without CPU device selection.

## Important caveats

### Weight loading is CPU-oriented

Model loading is not the same thing as inference dispatch.

[`mlx-rs/mlx-rs/src/ops/io.rs`](/Users/amos/bearcove/bee/mlx-rs/mlx-rs/src/ops/io.rs:50) marks `load_safetensors_device` with `#[default_device(device = "cpu")]`, so loading safetensors defaults to CPU. That means weights are loaded CPU-side unless a caller explicitly says otherwise.

This affects:

- [`rust/bee-qwen3-asr/src/load.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/load.rs:31)
- [`rust/bee-qwen3-asr/src/weights.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/weights.rs:41)
- [`rust/bee-zipa-mlx/src/infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:147)
- [`rust/bee-vad/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:50)
- [`rust/bee-g2p-charsiu-mlx/src/load.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/load.rs:41)
- [`rust/bee-transcribe/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-transcribe/src/lib.rs:127)

That does not imply the forward pass runs on CPU.

### Some values are read back to CPU

Several code paths call `.item()`, `.as_slice()`, or `.eval()` to materialize outputs for control flow or result extraction. Examples:

- [`rust/bee-vad/src/lib.rs`](/Users/amos/bearcove/bee/rust/bee-vad/src/lib.rs:187)
- [`rust/bee-qwen3-asr/src/forced_aligner.rs`](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/forced_aligner.rs:166)
- [`rust/bee-zipa-mlx/src/infer.rs`](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs:223)
- [`rust/bee-g2p-charsiu-mlx/src/model.rs`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu-mlx/src/model.rs:684)

That is result materialization on the host. It is not evidence that the model forward pass is CPU-only.

### Transcribe code assumes Metal-backed MLX runtime

[`rust/bee-transcribe/src/mlx_stuff.rs`](/Users/amos/bearcove/bee/rust/bee-transcribe/src/mlx_stuff.rs:16) manages the MLX Metal buffer cache directly. This strongly suggests the intended runtime is GPU-backed MLX on Apple silicon.

## Bottom line

Based on static code review, these MLX inference crates are intended to run inference on the GPU by default, with CPU involvement mainly in:

- safetensors loading
- mel or feature preprocessing done outside MLX
- scalar or vector readback for decoding decisions and final outputs

I did not find code in the reviewed inference crates that forces the model forward pass onto CPU.
