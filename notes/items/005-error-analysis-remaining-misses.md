# 005: Error analysis on remaining misses

## Goal

Inspect the three failure buckets to guide next improvements.

## Failure decomposition (at GT=0.3, RT=0.1)

| Bucket | Count | Description |
|--------|-------|-------------|
| Gate misses | 2 | Gate prob too low, span not opened |
| Ranker misses | 8 | Gate opens but gold is not top-1 |
| Never reaches judge | 20 | Gold not retrieved (2) or not verified (18) |

## Gate misses (2 cases)

Investigate:
- What are the terms/transcripts?
- Is ASR uncertainty signal missing or wrong?
- Is the context unusual (odd app, rare sentence structure)?
- Could additional gate features help?

## Ranker misses (8 cases)

Investigate:
- What candidate beats gold? Is it a near-synonym or totally wrong?
- Are candidate-relative features working for these cases?
- Is verified=true on the wrong candidate?
- Would more candidate features help (e.g., candidate x context crosses)?

## Never reaches judge (20 cases)

Sub-buckets:
- **Not retrieved (2):** Is the term in the index? Is the span too
  different phonetically? Is shortlist_limit too small?
- **Not verified (18):** What does verification check? Are these
  terms that fail G2P or have unusual phonetic patterns?

This is the biggest non-judge bottleneck. Fixing verification would
expand the judge's opportunity set from 86 → up to 104 cases.

## Depends on

- No code dependencies — this is analysis work
- Uses the existing offline eval infrastructure

## Output

Per-case failure report with:
- Term, transcript, span text
- Failure stage
- Root cause hypothesis
- Suggested fix (if any)
