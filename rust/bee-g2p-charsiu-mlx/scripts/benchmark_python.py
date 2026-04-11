#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "sentencepiece", "packaging", "numpy"]
# ///
"""Benchmark the Python/PyTorch sidecar G2P inference."""

from __future__ import annotations

import time
import sys

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers"
TOKENIZER_ID = "google/byt5-small"
LANG_CODE = "eng-us"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    words = sys.argv[1:]
    if not words:
        print("Usage: benchmark_python.py word1 word2 ...", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    print(f"Device: {device}", file=sys.stderr)
    print(f"Loading model...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Warmup
    prompts = [f"<{LANG_CODE}>: hello"]
    encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**encoded, num_beams=1, max_length=64, do_sample=False)

    # --- Per-word benchmark (sequential, one word at a time) ---
    print(f"\n=== Sequential (1 word at a time) ===", file=sys.stderr)
    total_us = 0
    for word in words:
        prompts = [f"<{LANG_CODE}>: {word}"]
        encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**encoded, num_beams=1, max_length=64, do_sample=False)
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000
        total_us += elapsed_ms * 1000
        decoded = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
        print(f"{word} -> {decoded[0]}  ({elapsed_ms:.1f}ms)")

    n = len(words)
    print(f"\n{n} words in {total_us/1000:.1f}ms total, {total_us/1000/n:.1f}ms/word average (sequential)", file=sys.stderr)

    # --- Batched benchmark (all words at once) ---
    print(f"\n=== Batched (all {n} words at once) ===", file=sys.stderr)
    prompts = [f"<{LANG_CODE}>: {word}" for word in words]
    encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**encoded, num_beams=1, max_length=64, do_sample=False)
    elapsed = time.perf_counter() - start
    elapsed_ms = elapsed * 1000
    decoded = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)

    for word, ipa in zip(words, decoded):
        print(f"{word} -> {ipa}")
    print(f"\n{n} words in {elapsed_ms:.1f}ms total, {elapsed_ms/n:.1f}ms/word average (batched)", file=sys.stderr)


if __name__ == "__main__":
    main()
