#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path


CONFIG = {
    "format": "zipa-mlx-hf-bundle-v1",
    "variant": "small-crctc-ns-no-diacritics-700k",
    "feature_dim": 80,
    "subsampling_factor": 4,
    "vocab_size": 127,
    "num_encoder_layers": [2, 2, 3, 4, 3, 2],
    "downsampling_factor": [1, 2, 4, 8, 4, 2],
    "feedforward_dim": [512, 768, 1024, 1536, 1024, 768],
    "num_heads": [4, 4, 4, 8, 4, 4],
    "encoder_dim": [192, 256, 384, 512, 384, 256],
    "encoder_unmasked_dim": [192, 192, 256, 256, 256, 192],
    "cnn_module_kernel": [31, 31, 15, 15, 15, 31],
    "query_head_dim": 32,
    "value_head_dim": 12,
    "pos_head_dim": 4,
    "pos_dim": 48,
    "causal": False,
    "use_ctc": True,
    "use_cr_ctc": True,
    "use_transducer": False,
}


README = """# ZIPA Small CR-CTC NS No-Diacritics 700k (MLX Q8)

This directory contains an MLX-native Q8 checkpoint bundle for the ZIPA small CR-CTC non-streaming no-diacritics model.

## Contents

- `model-q8.safetensors`: quantized MLX checkpoint in the `zipa-mlx-quantized-v1` format
- `tokens.txt`: CTC vocabulary
- `config.json`: model architecture and quantization metadata

## Source model

- Base model family: `anyspeech/zipa-small-crctc-ns-no-diacritics-700k`
- Dense source artifacts were exported locally from the ZIPA reference checkout and then quantized with `bee-zipa-mlx`

## Quantization notes

- Linear layers are quantized to 8-bit where MLX supports the target group size
- Small incompatible projections remain dense inside the checkpoint
- Non-linear modules such as norms, bypass scales, conv weights, and downsample weights remain dense

## Intended consumer

This bundle is intended to be used from the Bee repository.

The reference loader and inference implementation currently live in-tree in:

- `rust/bee-zipa-mlx`

Example from a Bee checkout:

```bash
cargo run -q -p bee-zipa-mlx --bin zipa-infer -- \\
  --quantized-checkpoint model-q8.safetensors \\
  /path/to/audio.wav
```

This artifact format is currently project-specific and should be treated as a Bee model bundle, not a general-purpose standalone MLX package.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "bearcove" / "zipa-mlx-hf"),
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--tokens",
        default=str(
            Path.home()
            / "bearcove"
            / "zipa"
            / "checkpoints"
            / "zipa-cr-ns-small-nodiacritics-700k"
            / "exp"
            / "tokens.txt"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model-q8.safetensors"
    tokens_path = Path(args.tokens).expanduser().resolve()

    subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "bee-zipa-mlx",
            "--bin",
            "zipa-export-quantized",
            "--",
            "--bits",
            str(args.bits),
            "--group-size",
            str(args.group_size),
            str(model_path),
        ],
        check=True,
        cwd=root,
    )

    shutil.copy2(tokens_path, output_dir / "tokens.txt")

    config = dict(CONFIG)
    config["quantization"] = {
        "format": "zipa-mlx-quantized-v1",
        "bits": args.bits,
        "group_size": args.group_size,
        "checkpoint": "model-q8.safetensors",
    }
    config["source_model_id"] = "anyspeech/zipa-small-crctc-ns-no-diacritics-700k"
    config["consumer_repo"] = "fasterthanlime/bee"
    config["consumer_crate"] = "bee-zipa-mlx"

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "README.md").write_text(README)

    print(f"bundle: {output_dir}")
    print(f"model: {model_path}")
    print(f"tokens: {output_dir / 'tokens.txt'}")
    print(f"config: {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
