#!/usr/bin/env python3
"""
Test the trained ASR correction model.

Loads the base model + LoRA adapters and runs correction on test examples.
"""

import json
import sys
from pathlib import Path

from mlx_lm import load, generate


def make_prompt(parakeet: str, qwen: str, vocab: list[str]) -> str:
    vocab_str = " ".join(vocab) if vocab else ""
    return f"<vocab> {vocab_str} <parakeet> {parakeet} <qwen> {qwen} <correct>"


def main():
    model_path = "Qwen/Qwen2.5-0.5B"
    adapter_path = "training/adapters"

    print(f"Loading model {model_path} + adapters from {adapter_path}...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Test cases: some from training data, some novel
    test_cases = [
        # From training data (should work well)
        {
            "parakeet": "The Certicrate handles serialization and the serialization in Rust.",
            "qwen": "The cerate handles serialization and deserialization in Rust.",
            "vocab": ["serde"],
            "expected": "The serde crate handles serialization and deserialization in Rust.",
        },
        {
            "parakeet": "We replaced Craig Graph with on time finish last week.",
            "qwen": "We replaced Quick Graph with OnTime Finish last week.",
            "vocab": ["CrateGraph", "on_time_finish"],
            "expected": "We replaced CrateGraph with on_time_finish last week.",
        },
        # Novel (not in training data)
        {
            "parakeet": "The sir day crate is really useful for Jason.",
            "qwen": "The third day crate is really useful for Jason.",
            "vocab": ["serde", "JSON"],
            "expected": "The serde crate is really useful for JSON.",
        },
        {
            "parakeet": "I'm using Tokyo for a sync runtime.",
            "qwen": "I'm using Tokyo for async runtime.",
            "vocab": ["tokio", "async"],
            "expected": "I'm using tokio for async runtime.",
        },
    ]

    print(f"\n{'='*70}")
    for i, tc in enumerate(test_cases):
        prompt = make_prompt(tc["parakeet"], tc["qwen"], tc["vocab"])
        result = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=100,
        )

        print(f"\nTest {i+1}:")
        print(f"  Parakeet:  {tc['parakeet']}")
        print(f"  Qwen3:     {tc['qwen']}")
        print(f"  Vocab:     {tc['vocab']}")
        print(f"  Expected:  {tc['expected']}")
        print(f"  Model:     {result.strip()}")
        match = result.strip() == tc["expected"]
        print(f"  {'✓ MATCH' if match else '✗ MISMATCH'}")

    # Also test on the held-out test set
    test_file = Path("training/data/test.jsonl")
    if test_file.exists():
        print(f"\n{'='*70}")
        print("Test set examples:")
        with open(test_file) as f:
            for line in f:
                item = json.loads(line)
                result = generate(
                    model, tokenizer,
                    prompt=item["prompt"],
                    max_tokens=100,
                )
                expected = item["completion"].strip()
                got = result.strip()
                match = got == expected
                print(f"\n  Prompt:    {item['prompt'][:80]}...")
                print(f"  Expected:  {expected}")
                print(f"  Model:     {got}")
                print(f"  {'✓' if match else '✗'}")


if __name__ == "__main__":
    main()
