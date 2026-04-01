#!/usr/bin/env python3
"""
Run the full eval pipeline on real recorded speech:
  WAV → Parakeet + Qwen3 (server mode) → correction model → compare to ground truth

All models stay warm for the entire run.
"""

import json
import subprocess
import sys
from pathlib import Path


class AsrServer:
    """Wraps an ASR binary in server mode (reads paths from stdin, writes JSON to stdout)."""

    def __init__(self, cmd: list[str]):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for "ready" message on stderr
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            sys.stderr.write(line)
            if "ready" in line.lower():
                break

    def transcribe(self, wav_path: str) -> str:
        self.proc.stdin.write(wav_path + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            return ""
        result = json.loads(line)
        if "error" in result:
            sys.stderr.write(f"ASR error: {result['error']}\n")
            return ""
        return result.get("text", "")

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/eval_recordings/desk-sm7b/manifest.jsonl"
    recordings_dir = Path(manifest_path).parent

    # Load manifest (deduplicate by sentence_index, keeping last = latest recording)
    entries = {}
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["sentence_index"]] = entry

    print(f"Loaded {len(entries)} unique recordings")

    # Build binaries first
    print("Building ASR binaries...")
    subprocess.run(
        ["cargo", "build", "-p", "synth-parakeet", "-p", "synth-qwen", "--release"],
        check=True, capture_output=True,
    )

    # Start ASR servers (models stay warm)
    print("Starting Parakeet server...")
    parakeet = AsrServer(["cargo", "run", "-p", "synth-parakeet", "--release"])

    print("Starting Qwen3 server...")
    qwen = AsrServer(["cargo", "run", "-p", "synth-qwen", "--release"])

    # Load correction model
    print("Loading correction model...")
    from mlx_lm import load, generate
    model, tokenizer = load("Qwen/Qwen2.5-0.5B", adapter_path="training/adapters")

    results = []
    for idx in sorted(entries.keys()):
        entry = entries[idx]
        wav_path = str(recordings_dir / entry["wav_path"])
        ground_truth = entry["text"]
        vocab = entry["vocab_terms"]

        # Run both ASR engines
        parakeet_text = parakeet.transcribe(wav_path)
        qwen_text = qwen.transcribe(wav_path)

        # Run corrector
        prompt = f"<parakeet> {parakeet_text} <qwen> {qwen_text} <correct>"
        corrected = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        corrected = corrected.split("<|endoftext|>")[0].strip()

        print(f"\n[{idx}] {ground_truth}")
        print(f"  P: {parakeet_text}")
        print(f"  Q: {qwen_text}")
        print(f"  C: {corrected}")

        results.append({
            "index": idx,
            "ground_truth": ground_truth,
            "parakeet": parakeet_text,
            "qwen": qwen_text,
            "corrected": corrected,
            "vocab": vocab,
        })

    parakeet.close()
    qwen.close()

    # Write results
    out_path = recordings_dir / "eval_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"Evaluated {len(results)} sentences\n")

    for r in results:
        gt = r["ground_truth"]
        p = r["parakeet"]
        q = r["qwen"]
        c = r["corrected"]

        # Simple: is corrected closer to ground truth than the best raw?
        gt_lower = gt.lower()
        gt_words = set(gt_lower.split())
        p_overlap = len(gt_words & set(p.lower().split()))
        q_overlap = len(gt_words & set(q.lower().split()))
        c_overlap = len(gt_words & set(c.lower().split()))
        best_raw = max(p_overlap, q_overlap)

        if c_overlap > best_raw:
            tag = "✓ improved"
        elif c_overlap < best_raw:
            tag = "✗ worsened"
        elif c.lower() == gt_lower:
            tag = "✓ exact"
        else:
            tag = "- unchanged"

        print(f"  [{r['index']:3d}] {tag}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
