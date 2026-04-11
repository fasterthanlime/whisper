# bee-g2p-charsiu scripts

This directory is for Python probes and sidecars while the strategy is still
being worked out.

The near-term plan is:

- inspect Charsiu's byte-level encoder/decoder behavior
- probe decoder cross-attention on real words
- see whether Qwen token pieces can be segmented against generated IPA
- only then decide what belongs in Rust/MLX

## Current Scripts

### `charsiu_g2p_sidecar.py`

Minimal JSON sidecar around the Charsiu checkpoint.

Example:

```bash
uv run rust/bee-g2p-charsiu/scripts/charsiu_g2p_sidecar.py --lang-code eng-us Facet Wednesday
```

### `charsiu_g2p_compare.py`

Migration/evaluation helper that compares Charsiu against the current eSpeak
baseline.

Example:

```bash
uv run rust/bee-g2p-charsiu/scripts/charsiu_g2p_compare.py --lang-code eng-us Facet SQLite serde
```

This script is for comparison work only. It still shells out to `espeak-ng`,
so it is not itself the replacement path.

### `charsiu_cross_attention_probe.py`

Model-inspection probe for the actual strategic question: whether Charsiu
decoder cross-attention can help split a word's pronunciation across Qwen
token-piece boundaries.

Examples:

```bash
eval "$(direnv export bash)"
uv run rust/bee-g2p-charsiu/scripts/charsiu_cross_attention_probe.py --word Facet --lang-code eng-us
uv run rust/bee-g2p-charsiu/scripts/charsiu_cross_attention_probe.py --word Wednesday --lang-code eng-us
```

This probe expects `BEE_ASR_MODEL_DIR` to be present so it can load the local
Qwen tokenizer and report Qwen token-piece spans for the test word.
