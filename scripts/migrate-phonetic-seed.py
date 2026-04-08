#!/usr/bin/env python3
"""Migrate phonetic-seed JSONL files from flat confidence fields to nested.

Old format per word: {"word": "...", "start": ..., "end": ..., "mean_logprob": ..., "min_logprob": ..., "mean_margin": ..., "min_margin": ...}
New format per word: {"word": "...", "start": ..., "end": ..., "confidence": {"mean_lp": ..., "min_lp": ..., "mean_m": ..., "min_m": ...}}

Idempotent: skips words that already have a "confidence" key.
Processes all .jsonl files in the given directory.
"""

import json
import sys
from pathlib import Path

FIELD_MAP = {
    "mean_logprob": "mean_lp",
    "min_logprob": "min_lp",
    "mean_margin": "mean_m",
    "min_margin": "min_m",
}


def migrate_word(word: dict) -> dict:
    if "confidence" in word:
        return word  # already migrated
    if "mean_logprob" not in word:
        return word  # no confidence fields at all

    confidence = {}
    out = {}
    for k, v in word.items():
        if k in FIELD_MAP:
            confidence[FIELD_MAP[k]] = v
        else:
            out[k] = v
    if confidence:
        out["confidence"] = confidence
    return out


def migrate_row(row: dict) -> dict:
    if "words" in row and isinstance(row["words"], list):
        row["words"] = [migrate_word(w) for w in row["words"]]
    return row


def migrate_file(path: Path) -> int:
    lines = path.read_text().splitlines()
    changed = 0
    out_lines = []
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        old = line
        new_row = migrate_row(row)
        new_line = json.dumps(new_row, ensure_ascii=False)
        if new_line != old:
            changed += 1
        out_lines.append(new_line)
    if changed > 0:
        path.write_text("\n".join(out_lines) + "\n")
    return changed


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <phonetic-seed-dir>", file=sys.stderr)
        sys.exit(1)

    seed_dir = Path(sys.argv[1])
    if not seed_dir.is_dir():
        print(f"Not a directory: {seed_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for jsonl in sorted(seed_dir.glob("*.jsonl")):
        n = migrate_file(jsonl)
        if n > 0:
            print(f"  {jsonl.name}: migrated {n} rows")
            total += n

    if total == 0:
        print("All files already migrated (or no confidence fields found).")
    else:
        print(f"Done: {total} rows migrated.")


if __name__ == "__main__":
    main()
