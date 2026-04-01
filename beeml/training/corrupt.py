#!/usr/bin/env python3
"""
Phoneme-based ASR corruption engine.

Given a clean text with vocab terms, introduces realistic ASR-like errors
by finding phonetically similar words/word-sequences in CMUdict.

Usage:
    python3 training/corrupt.py "The serde crate handles serialization"
    python3 training/corrupt.py --vocab serde,tokio --batch data/sentences.jsonl
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_cmudict(path: str = "data/cmudict.txt") -> dict[str, list[str]]:
    """Load CMUdict: word → list of phonemes (ARPAbet, no stress markers)."""
    entries = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            word = parts[0].upper()
            # Skip alternate pronunciations (WORD(2), WORD(3))
            if "(" in word:
                continue
            # Strip stress markers (0,1,2) from vowels
            phonemes = [re.sub(r"\d", "", p) for p in parts[1:]]
            entries[word] = phonemes
    return entries


def build_phoneme_index(cmudict: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build reverse index: phoneme bigram → list of words containing it."""
    index = defaultdict(set)
    for word, phonemes in cmudict.items():
        for i in range(len(phonemes)):
            # Unigram
            index[phonemes[i]].add(word)
            # Bigram
            if i + 1 < len(phonemes):
                key = f"{phonemes[i]}_{phonemes[i+1]}"
                index[key].add(word)
    return {k: list(v) for k, v in index.items()}


def phoneme_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance on phoneme sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


def g2p_simple(word: str, cmudict: dict[str, list[str]]) -> list[str]:
    """Simple G2P: lookup in CMUdict, fall back to letter-by-letter."""
    upper = word.upper().strip(".,!?;:'\"")
    if upper in cmudict:
        return cmudict[upper]
    # Fallback: crude letter-to-phoneme mapping
    LETTER_PHONEMES = {
        'A': ['AE'], 'B': ['B'], 'C': ['K'], 'D': ['D'], 'E': ['EH'],
        'F': ['F'], 'G': ['G'], 'H': ['HH'], 'I': ['IH'], 'J': ['JH'],
        'K': ['K'], 'L': ['L'], 'M': ['M'], 'N': ['N'], 'O': ['AA'],
        'P': ['P'], 'Q': ['K', 'W'], 'R': ['R'], 'S': ['S'], 'T': ['T'],
        'U': ['AH'], 'V': ['V'], 'W': ['W'], 'X': ['K', 'S'],
        'Y': ['Y'], 'Z': ['Z'],
    }
    phonemes = []
    for ch in upper:
        if ch in LETTER_PHONEMES:
            phonemes.extend(LETTER_PHONEMES[ch])
    return phonemes


def find_single_word_confusions(
    target_phonemes: list[str],
    cmudict: dict[str, list[str]],
    max_distance: int = 3,
    max_results: int = 10,
) -> list[tuple[str, int]]:
    """Find single words whose phonemes are close to target."""
    candidates = []
    target_len = len(target_phonemes)

    for word, phonemes in cmudict.items():
        # Quick filter: skip if length difference is too large
        if abs(len(phonemes) - target_len) > max_distance:
            continue
        dist = phoneme_edit_distance(target_phonemes, phonemes)
        if dist <= max_distance and dist > 0:  # exclude exact matches
            candidates.append((word.lower(), dist))

    candidates.sort(key=lambda x: (x[1], x[0]))
    return candidates[:max_results]


def find_two_word_confusions(
    target_phonemes: list[str],
    cmudict: dict[str, list[str]],
    max_distance: int = 2,
    max_results: int = 10,
) -> list[tuple[str, int]]:
    """Find two-word phrases whose combined phonemes are close to target."""
    candidates = []
    target_len = len(target_phonemes)

    # Pre-filter: only consider short words (≤ target_len phonemes)
    short_words = {w: p for w, p in cmudict.items() if len(p) <= target_len and len(p) >= 1}

    # Try all split points
    for split in range(1, target_len):
        left_target = target_phonemes[:split]
        right_target = target_phonemes[split:]

        # Find best left word
        best_left = []
        for word, phonemes in short_words.items():
            if abs(len(phonemes) - len(left_target)) > 1:
                continue
            dist = phoneme_edit_distance(left_target, phonemes)
            if dist <= max_distance:
                best_left.append((word, phonemes, dist))

        best_left.sort(key=lambda x: x[2])
        best_left = best_left[:20]

        for lw, lp, ld in best_left:
            for word, phonemes in short_words.items():
                if abs(len(phonemes) - len(right_target)) > 1:
                    continue
                rd = phoneme_edit_distance(right_target, phonemes)
                total = ld + rd
                if total <= max_distance and total > 0:
                    phrase = f"{lw.lower()} {word.lower()}"
                    candidates.append((phrase, total))

    # Deduplicate
    seen = set()
    unique = []
    for phrase, dist in candidates:
        if phrase not in seen:
            seen.add(phrase)
            unique.append((phrase, dist))

    unique.sort(key=lambda x: (x[1], x[0]))
    return unique[:max_results]


def corrupt_term(
    term: str,
    cmudict: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Find a plausible ASR confusion for a technical term."""
    phonemes = g2p_simple(term, cmudict)
    if not phonemes:
        return term

    # Collect candidates from single and multi-word
    singles = find_single_word_confusions(phonemes, cmudict, max_distance=3, max_results=20)
    doubles = find_two_word_confusions(phonemes, cmudict, max_distance=2, max_results=10)

    all_candidates = singles + doubles
    if not all_candidates:
        return term

    # Weight by inverse distance (closer = more likely)
    weights = [1.0 / (dist + 0.5) for _, dist in all_candidates]
    total = sum(weights)
    weights = [w / total for w in weights]

    chosen = rng.choices([c[0] for c in all_candidates], weights=weights, k=1)[0]
    return chosen


def corrupt_sentence(
    text: str,
    vocab_terms: list[str],
    cmudict: dict[str, list[str]],
    rng: random.Random,
    corruption_prob: float = 0.7,
) -> str:
    """Corrupt a sentence by replacing vocab terms with phonetic confusions."""
    result = text
    for term in vocab_terms:
        if rng.random() > corruption_prob:
            continue
        confused = corrupt_term(term, cmudict, rng)
        if confused != term.lower():
            # Replace the term in the sentence (case-insensitive)
            result = re.sub(re.escape(term), confused, result, flags=re.IGNORECASE)
    return result


def main():
    cmudict_path = "data/cmudict.txt"
    print(f"Loading CMUdict from {cmudict_path}...", file=sys.stderr)
    cmudict = load_cmudict(cmudict_path)
    print(f"Loaded {len(cmudict)} entries", file=sys.stderr)

    rng = random.Random(42)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Single term mode
        term = sys.argv[1]
        phonemes = g2p_simple(term, cmudict)
        print(f"Term: {term}")
        print(f"Phonemes: {' '.join(phonemes)}")
        print(f"\nSingle-word confusions:")
        for word, dist in find_single_word_confusions(phonemes, cmudict):
            wp = cmudict.get(word.upper(), [])
            print(f"  {word:20s} (dist={dist})  {' '.join(wp)}")
        print(f"\nTwo-word confusions:")
        for phrase, dist in find_two_word_confusions(phonemes, cmudict):
            print(f"  {phrase:30s} (dist={dist})")
    else:
        # Interactive demo
        terms = ["serde", "tokio", "axum", "ratatui", "kajit", "GGUF", "Cargo.toml", "reqwest"]
        for term in terms:
            phonemes = g2p_simple(term, cmudict)
            print(f"\n{'='*50}")
            print(f"Term: {term}  →  phonemes: {' '.join(phonemes)}")

            singles = find_single_word_confusions(phonemes, cmudict, max_distance=3, max_results=5)
            doubles = find_two_word_confusions(phonemes, cmudict, max_distance=2, max_results=5)

            if singles:
                print(f"  Single-word:")
                for word, dist in singles:
                    print(f"    {word:20s} (dist={dist})")
            if doubles:
                print(f"  Two-word:")
                for phrase, dist in doubles:
                    print(f"    {phrase:30s} (dist={dist})")

            # Show a random corruption
            confused = corrupt_term(term, cmudict, rng)
            print(f"  → corrupted: \"{confused}\"")


if __name__ == "__main__":
    main()
