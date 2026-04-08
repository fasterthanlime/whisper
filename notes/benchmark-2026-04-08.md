# Benchmark snapshot — 2026-04-08

## Three scoreboards (GT=0.4 RT=0.1, 5-fold CV, 4 epochs)

### 1. End-to-end (all cases)

| Metric | Value |
|--------|-------|
| Canonical corrected | 89/106 (84.0%) |
| Counterex abstained | 113/113 (100.0%) |
| False positive rate | 0/113 (0.0%) |

### 2. Judge-stage (reachable only)

| Metric | Value |
|--------|-------|
| Gate balanced accuracy | 96.5% |
| Ranker top-1 accuracy | 95/99 (96.0%) |
| Composed balanced | 94.4% |

### 3. Upstream opportunity set

| Metric | Value |
|--------|-------|
| Gold retrieved | 105/106 (99.1%) |
| Gold verified | 99/106 (93.4%) |

## What produced the jump (from 74.5% → 84.0%)

### 1. espeak-ng reference IPA in vocab.jsonl (+5 cases)

Replaced hand-written IPA with espeak-ng (en-us) output for all 26 terms.
The hand-written IPA used non-standard symbols and spacing that didn't
match espeak's output for span text, causing systematic phonetic distance
inflation in the verifier.

Commit: 96c4e80

### 2. Confusion aliases from observed ASR transcripts (+3 cases)

Created `data/phonetic-seed/confusion_forms.jsonl` with aliases derived
from actual ASR transcripts in recording_examples.jsonl:

- bearcove: Berko, Berghof, Barricove
- serde: Sirdal, sturdy
- miri: Miriam
- DWARF: dwarf, door
- QEMU: kemoo (how espeak reads "QEMU" as text)
- repr: rapper, wrapper

Each alias has espeak-generated IPA so it matches the span IPA
comparison convention.

Commit: 239ceff, a9696c6

### 3. Term-based scoring fix (+2 cases)

Changed eval scoring to match by term name instead of alias_id.
When a term has multiple aliases (canonical, spoken, confusion),
the ranker picking any alias of the correct term should count as
correct. Previously it only counted exact alias_id match.

Commit: 239ceff

## Remaining 6 verification misses

These are genuinely far-off ASR cases where the transcript doesn't
resemble the spoken form:

- fasterthanlime: "fascinating article for everything"
- serde (×2): "surely"/"tract", "third-party JSON"
- QEMU: phon=0.17, very distant
- SQLite: "SQL database" (not "sequel light")
- serde: "surely" with short/low-content guards

These need either onboarding examples, ZIPA-captured pronunciation
(see item 008), or ASR-side alternative-hypothesis signals.

## Operating point

Do not touch thresholds casually. Zero false positives at 84%
canonical is an excellent product operating point.
