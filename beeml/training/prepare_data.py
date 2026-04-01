#!/usr/bin/env python3
"""
Convert pipeline JSONL output into MLX-LM fine-tuning format.

Input format (from synth-pipeline):
  {"original_text":"...","parakeet_output":"...","qwen_output":"...","vocab":["serde"],"voice_id":"amos"}

Output format (MLX-LM completions):
  {"prompt":"<vocab> serde <parakeet> The Certicrate... <qwen> The cerate... <correct>","completion":" The serde crate..."}

Also generates unchanged examples (where prompt ≈ completion) to teach
the model not to over-correct.
"""

import json
import random
import sys
from pathlib import Path


def make_prompt(parakeet: str, qwen: str, vocab: list[str]) -> str:
    vocab_str = " ".join(vocab) if vocab else ""
    return f"<vocab> {vocab_str} <parakeet> {parakeet} <qwen> {qwen} <correct>"


def convert(input_path: str, output_dir: str, unchanged_ratio: float = 0.3):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    examples = []

    for pair in pairs:
        original = pair["original_text"]
        parakeet = pair["parakeet_output"]
        qwen = pair["qwen_output"]
        vocab = pair["vocab"]

        # Correction example: noisy → clean
        # Include EOS token so the model learns to stop
        prompt = make_prompt(parakeet, qwen, vocab)
        examples.append({
            "prompt": prompt,
            "completion": f" {original}<|endoftext|>",
        })

    # Add unchanged examples (teach the model to leave correct text alone)
    n_unchanged = int(len(examples) * unchanged_ratio)
    for pair in random.sample(pairs, min(n_unchanged, len(pairs))):
        original = pair["original_text"]
        vocab = pair["vocab"]
        # When both ASR outputs are already correct, output should be unchanged
        prompt = make_prompt(original, original, vocab)
        examples.append({
            "prompt": prompt,
            "completion": f" {original}<|endoftext|>",
        })

    random.shuffle(examples)

    # Split: 80% train, 10% valid, 10% test
    n = len(examples)
    n_test = max(1, n // 10)
    n_valid = max(1, n // 10)
    n_train = n - n_test - n_valid

    train = examples[:n_train]
    valid = examples[n_train:n_train + n_valid]
    test = examples[n_train + n_valid:]

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    print(f"\nTotal: {len(examples)} examples ({n_train} train, {n_valid} valid, {n_test} test)")
    print(f"\nSample prompt:\n{examples[0]['prompt']}")
    print(f"Sample completion:\n{examples[0]['completion']}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "training/data"
    convert(input_path, output_dir)
