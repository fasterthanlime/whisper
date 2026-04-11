# bee-roll

`bee-roll` is the next rollback-oriented streaming ASR core for Bee.

Bee is a consumer dictation app for macOS. It captures live audio, shows
evolving text quickly, inserts text through a custom IME, and later uses
the richer token/phone output for correction and debugging.

This crate is where the rollback model gets expressed directly, instead of
being spread across ad hoc decode code.

## What We Are Building

The product goal is not "wait for the whole utterance, then run one perfect
transcription pass."

The goal is:

- low-latency visible text while the user is still speaking
- acceptable live rewrites near the right edge
- bounded work per update
- one evolving transcript, not a pile of unrelated window-local transcripts
- token-aligned side data that later stages can actually use

That last point matters. The output here is not just text. The correction
pipeline and the debugging visualization need more structure than that.

## The Core Model

Conceptually, an utterance has two regions:

- `committed`: the stable prefix
- `fresh`: everything after that

`fresh` is the live tail. It is expected to change.

`feed()` does not just append more text forever. It keeps rewriting the
current live tail as more audio arrives.

That means:

1. keep the committed prefix
2. keep enough prefix/context for the next decode step
3. truncate the revisable tail in token space and KV space
4. decode again over the current fresh region
5. rebuild the current output tape

The canonical rollback unit is still the token boundary. Phones and timings
help choose coherent cut points, but they do not replace tokens as the
coordinate system used for truncation, commitment, and KV synchronization.

## The Three-Way Split

The right mental model for the live decode step is a three-way split:

- a kept prefix that stays alive in KV
- a replayed bridge/prefix that is fed into the next decode step
- a reopened tail that gets re-decoded

This is not full-prefix rerun.

It is also not append-only streaming.

It is a bounded-cost rollback strategy that preserves older context while
reopening the recent tail so the model can revise seam-local mistakes.

## How `feed()` Is Supposed To Work

`feed()` is the only public ingress for audio.

As audio arrives, `fresh` grows.

While `fresh` is still small, for example under a threshold like `2s`,
calling `feed()` means:

- extend the utterance audio buffer
- truncate the live tail back to the kept/replay point
- re-run ASR over the current fresh region
- return the whole current token-aligned tape

At that stage, there is no point searching for a new cut yet. The system is
still just repeatedly rewriting the live tail.

Once `fresh` is large enough, `feed()` also asks the `Cutter` to choose a
good boundary. If a cut is accepted:

- `committed` advances
- the fresh region becomes smaller again
- the next rounds go back to repeated live-tail rewrites until the fresh
  region grows enough to justify another cut search

So the loop is:

1. keep rewriting `fresh`
2. once big enough, search for a cut
3. extend `committed`
4. repeat

## Why We Return More Than Text

Text alone is not enough.

The current whole tape should be able to carry token-aligned side data such
as:

- ASR token confidence and alternatives
- detected language
- G2P-derived IPA
- ZIPA-derived phone spans

Those outputs matter for two separate reasons:

1. correction
   Later stages need token-aligned structure, not just a flattened string.
2. debugging
   The HTML/debug visualization needs to show what the model thought, how
   confident it was, and how transcript-side and audio-side phonetic views
   line up.

## Canonical Boundaries

The important internal boundary is still token space.

This crate uses token indices for:

- rollback
- commitment
- KV truncation
- stage-to-stage references

Timings, word structure, and phones are useful for deciding where a cut
should land, but the system should not become phone-indexed or word-indexed
internally just because those views are useful.

## What This Crate Owns

At a high level, `Utterance` owns:

- append-only utterance audio
- the committed boundary
- the current token-aligned output tape
- synchronized rollback state for ASR
- the cut policy hook
- the listener/debug hook

The current public surface is intentionally small:

- `Utterance::{new, feed}`
- associated token/output types
- `Cutter`
- `Listener`

## Current Status

This crate is still a scaffold.

The types are being shaped so the real state machine can land cleanly.
There is already a strong bias in the design:

- one canonical token-aligned tape
- synchronized token/KV rollback
- repeated re-decode of the fresh tail
- cut search only when the fresh region is large enough
- richer per-token output because later stages need it

What is not implemented yet is the actual concrete decode loop that wires in
Qwen3 ASR, G2P, ZIPA, and the cut decision process.
