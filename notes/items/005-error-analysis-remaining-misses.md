# 005: Error analysis on remaining misses

## Goal

Inspect the three failure buckets to guide next improvements.
This is the highest-leverage analysis task once the two-stage judge
is in the real path.

## Priority

Move this earlier in practice than numbering suggests. Once 002 lands,
this becomes the most valuable engineering work.

## Failure decomposition (at GT=0.3, RT=0.1)

| Bucket | Count | Description |
|--------|-------|-------------|
| Not retrieved | 2 | Gold term not in shortlist |
| Not verified | 18 | Gold retrieved but verification fails |
| Gate misses | 2 | Gate prob too low, span not opened |
| Ranker misses | 8 | Gate opens but gold is not top-1 |

## Not retrieved (2 cases) — upstream

- Is the term in the index?
- Is the span too different phonetically?
- Is shortlist_limit too small?

## Not verified (18 cases) — upstream, biggest opportunity

This is the largest single bucket. Fixing verification would expand
the judge's opportunity set from 86 → up to 104 cases.

- What does verification check? G2P? Phonetic distance?
- Are these terms with unusual phonetic patterns?
- Are there systematic patterns (all identifiers? all multi-word?)

## Gate misses (2 cases)

- What are the terms/transcripts?
- Is ASR uncertainty signal missing or wrong?
- Is the context unusual?

## Ranker misses (8 cases)

- What candidate beats gold?
- Are candidate-relative features working?
- Is verified=true on the wrong candidate?
- Would candidate x context crosses help?

## Output

Per-case failure report with:
- Term, transcript, span text
- Failure stage and sub-stage
- Root cause hypothesis
- Suggested fix (if any)
- Estimated impact (how many cases would be fixed)

## Depends on

- No code dependencies — uses existing offline eval infrastructure
