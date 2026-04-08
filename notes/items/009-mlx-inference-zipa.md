# 009: MLX inference for ZIPA model

## Goal

Run the ZIPA speech-to-IPA model via MLX on Apple Silicon, so it can
run on-device without a Python runtime or server roundtrip.

## Model

- [anyspeech/zipa-small-crctc-ns-no-diacritics-700k](https://huggingface.co/anyspeech/zipa-small-crctc-ns-no-diacritics-700k)
- Architecture: CTC-based (CRCTC variant), small size
- Output: IPA tokens without diacritics

## Research needed

- What is the model architecture exactly? (encoder type, attention, etc.)
- Is there an existing MLX conversion for this model or this architecture?
- What audio preprocessing does it expect? (sample rate, mel specs, etc.)
- What is the CTC decoding vocabulary? (IPA phone inventory)
- Can we reuse any of the existing mlx-rs infrastructure from bee-qwen3-asr?

## Approach options

1. **Convert weights to MLX format** — if the architecture maps to
   existing MLX ops, convert the PyTorch checkpoint and write a thin
   Rust inference wrapper using mlx-rs
2. **Port the model** — if it's a standard encoder (e.g., Wav2Vec2-style
   or Conformer), implement it in mlx-rs directly
3. **Python bridge** — last resort, run the HuggingFace model via
   a Python subprocess. Undesirable for shipping but fine for
   offline batch processing initially.

## Validation

- Feed known audio clips through both the original Python model and
  the MLX port, compare IPA output
- Verify IPA tokens are in the same inventory as bee-phonetic expects

## Depends on

- mlx-rs (already in tree)
- Understanding of ZIPA model architecture (research step)
