# Remaining misses — 2026-04-08

## Gate misses (5 cases)

All reachable + verified, but gate probability below threshold.

| Case | Term | Gate prob | ASR transcript | Notes |
|------|------|-----------|---------------|-------|
| 95 | Vec | 0.244 | "put it on in Vec" | Term in transcript, gate should open |
| 74 | serde_json | 0.262 | "Let's use serde_json to serialize" | Term in transcript, gate should open |
| 100 | x86_64 | 0.226 | "X eighty-six sixty-four" | Term in transcript |
| 11 | bearcove | 0.250 | "Bear Cove org" | Term in transcript |
| 14 | fasterthanlime | 0.298 | "Fast and the Name" | Garbled but close |

3 of 5 have the term correctly transcribed. The gate is failing to
open on easy cases → seed weights are insufficient. This is the
strongest signal for 007 (ship trained weights).

## Ranker misses (4 cases)

Gate opens, but ranker picks wrong term.

| Case | Term | Ranker picked | Prob | ASR transcript | Notes |
|------|------|--------------|------|---------------|-------|
| 34 | MIR | miri | 0.820 | "Mir" | Near-neighbor, very close IPA |
| 35 | MIR | miri | 0.820 | "Mir" | Same |
| 69 | serde | miri | 0.815 | "Siri" | Cross-term: "Siri" sounds like miri |
| 46 | repr | ripgrep | 0.841 | "regular two" | Both "r" terms |

MIR/miri is a genuine near-neighbor problem. serde→miri is
cross-term phonetic confusion. repr→ripgrep is similar.

## Verification misses (6 cases)

All genuinely far-off ASR. No alias fix possible.

Not documented here — see failure report output. These are the
ZIPA/onboarding class (item 008).
