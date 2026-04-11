#!/usr/bin/env python3
"""Convert a pytorch_model.bin checkpoint to model.safetensors."""
# /// script
# dependencies = ["torch", "safetensors", "packaging", "numpy"]
# ///

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pytorch_model.bin to safetensors")
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing pytorch_model.bin (and config.json)",
    )
    args = parser.parse_args()

    bin_path = args.model_dir / "pytorch_model.bin"
    out_path = args.model_dir / "model.safetensors"

    if not bin_path.exists():
        raise FileNotFoundError(f"No pytorch_model.bin in {args.model_dir}")

    print(f"Loading {bin_path} ...")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # safetensors requires contiguous tensors and no shared memory.
    # T5 ties encoder.embed_tokens, decoder.embed_tokens, and shared —
    # keep shared.weight as the canonical copy, clone the others.
    deduped = {}
    for k, v in state_dict.items():
        deduped[k] = v.clone().contiguous()

    print(f"Saving {out_path} ({len(deduped)} tensors) ...")
    save_file(deduped, str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
