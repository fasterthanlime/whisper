#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a saved ByT5 checkpoint on a single prompt.")
    parser.add_argument("--model-dir", default="training/byt5-small-run")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
