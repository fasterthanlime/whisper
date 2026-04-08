# 008: ZIPA audio-to-IPA for alias capture

## Goal

Let users teach the system how they say a term by speaking it.
Audio goes through ZIPA (speech-to-IPA model), producing IPA directly
from the waveform — no text roundtrip, no espeak.

That IPA becomes the alias entry in the phonetic database, usable
for all downstream matching and verification.

## Why this matters

The current alias IPA is hand-written in vocab.jsonl or reused from the
canonical form. This causes verification failures because:

- Hand-written IPA doesn't match espeak's output conventions
- Identifier aliases ("u eight") reuse the canonical IPA instead of
  getting their own
- espeak-based IPA for alias text is a guess at pronunciation, not
  how anyone actually says it

ZIPA solves this by capturing real pronunciation. Both sides of the
phonetic comparison become grounded in actual speech:

- Span IPA: ASR audio → ZIPA (or espeak on ASR text, for now)
- Alias IPA: user audio → ZIPA → stored

## Pipeline

```
User says "sirday" into mic
    → audio waveform
    → ZIPA model
    → IPA tokens (e.g., ["s", "ɜː", "d", "eɪ"])
    → stored as alias for term "serde", source = spoken_observed
```

## Model

- [anyspeech/zipa-small-crctc-ns-no-diacritics-700k](https://huggingface.co/anyspeech/zipa-small-crctc-ns-no-diacritics-700k)
- Small CTC-based model, no diacritics variant
- Needs MLX inference (see 009)

## What this enables

- User onboarding: "say each term in your vocabulary"
- Per-user pronunciation aliases (accent-aware)
- Confusion alias mining from real ASR sessions (feed ASR audio
  segments through ZIPA to get ground-truth IPA for what was said)
- Offline-ish: doesn't need to run in real-time, can process
  a recording batch

## Integration points

- New alias source: `AliasSource::SpokenObserved` (or similar)
- Storage: alias DB gains IPA entries keyed by (term, user, source)
- Verification: these aliases participate in retrieval + verification
  like any other alias — but with IPA that actually matches speech

## Depends on

- 009 (MLX inference for ZIPA)
