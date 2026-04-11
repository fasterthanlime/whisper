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

The whole utterance should be thought of as three adjacent regions:

- `stable`: old text that is settled enough to keep
- `carry`: recent kept text that is replayed into the next decode step
- `preview`: the live right edge that gets reopened and re-decoded

For a longer utterance, the tape can look like this:

```text
This is a long utterance that warrants several rounds
SSSSSSSSSSSSSS CCCCCCCCCCCCCC PPPPPPPPPPPPPPPPPPPPPPP
```

That is the vocabulary this crate should use.

`stable`, `carry`, and `preview` are all part of the current tape. The
difference is how the next decode step treats them:

- `stable` remains alive in retained KV state
- `carry` is replayed as bridging text into the next decode step
- `preview` is the live tail that gets rewritten

The canonical rollback unit is still the token boundary. Phones and timings
help choose coherent cut points, but they do not replace tokens as the
coordinate system used for truncation, promotion, and KV synchronization.

## The Three-Way Split

The three-way split is not just a visualization trick. It is the actual
operational model:

- keep `stable`
- replay `carry`
- reopen `preview`

This is not full-prefix rerun.

It is also not append-only streaming.

It is a bounded-cost rollback strategy that preserves older context while
reopening the recent tail so the model can revise seam-local mistakes.

## How `feed()` Is Supposed To Work

`feed()` is the only public ingress for audio.

As audio arrives, `preview` grows.

While `preview` is still small, for example under a threshold like `2s`,
calling `feed()` means:

- extend the utterance audio buffer
- keep the existing `stable` prefix
- keep the existing `carry` bridge
- truncate the live decode state back to the kept point
- re-run ASR for the current live tail
- return the whole current token-aligned tape

At that stage, there is no point searching for a new cut yet. The system is
still just repeatedly rewriting `preview`.

Once `preview` is large enough, `feed()` also asks the `Cutter` to choose a
good boundary. If a cut is accepted:

- more material moves into `stable`
- the right edge is repartitioned into a new `carry` and `preview`
- the next rounds go back to repeated preview rewrites until `preview`
  grows enough to justify another cut search

So the loop is:

1. keep rewriting `preview`
2. once big enough, search for a cut
3. extend `stable`
4. repartition the right edge into `carry` + `preview`
5. repeat

## Worked Timeline: 200 ms Feeds

Here is the same model spelled out on a concrete timeline.

Assumptions for this example:

- the app calls `feed()` every `200 ms`
- `preview` becomes eligible for a cut search at `2.0 s`
- token text below is schematic; the point is the state transition, not the
  exact words
- on the very first decode, ASR uses its initial prompt form
- on later decodes, ASR uses its follow-up prompt form
- after the first cut in this example:
  - `stable = 0.0 .. 1.2 s`
  - `carry = 1.2 .. 1.6 s`
  - `preview = 1.6 .. 2.0 s`

### Phase 1: bootstrapping before the first cut

Before the first cut exists, there is no stable prefix yet and there is no
carry to replay. So each `feed()` call rewrites the whole current tape.

| feed | new samples | utterance audio after append | KV kept before decode | replayed carry tokens | audio fed to ASR | tape after decode | cutter? |
|---|---:|---|---|---|---|---|---|
| 1 | `+0.2s` | `0.0 .. 0.2` | none | none | `0.0 .. 0.2` | all `preview` | no |
| 2 | `+0.2s` | `0.0 .. 0.4` | none, truncate back to `0` | none | `0.0 .. 0.4` | all `preview` | no |
| 3 | `+0.2s` | `0.0 .. 0.6` | none, truncate back to `0` | none | `0.0 .. 0.6` | all `preview` | no |
| 4 | `+0.2s` | `0.0 .. 0.8` | none, truncate back to `0` | none | `0.0 .. 0.8` | all `preview` | no |
| 5 | `+0.2s` | `0.0 .. 1.0` | none, truncate back to `0` | none | `0.0 .. 1.0` | all `preview` | no |
| 6 | `+0.2s` | `0.0 .. 1.2` | none, truncate back to `0` | none | `0.0 .. 1.2` | all `preview` | no |
| 7 | `+0.2s` | `0.0 .. 1.4` | none, truncate back to `0` | none | `0.0 .. 1.4` | all `preview` | no |
| 8 | `+0.2s` | `0.0 .. 1.6` | none, truncate back to `0` | none | `0.0 .. 1.6` | all `preview` | no |
| 9 | `+0.2s` | `0.0 .. 1.8` | none, truncate back to `0` | none | `0.0 .. 1.8` | all `preview` | no |
| 10 | `+0.2s` | `0.0 .. 2.0` | none, truncate back to `0` | none | `0.0 .. 2.0` | decode full tape, then partition into `stable` / `carry` / `preview` | yes |

At `feed(10)`, the cutter runs for the first time because `preview` has
grown to the search threshold.

For this example, suppose it chooses:

```text
audio time:  0.0                1.2      1.6        2.0
             |------------------|--------|----------|
tape:        S S S S S S        C C      P P P P
```

That means:

- everything before `1.2 s` becomes `stable`
- `1.2 .. 1.6 s` becomes `carry`
- `1.6 .. 2.0 s` remains `preview`

The returned tape now has all three regions.

### Phase 2: steady-state after the first cut

After that first cut, `feed()` stops rewriting from `0`.

Instead, each later call:

1. keeps KV only for `stable`
2. drops the old `carry + preview` suffix from live decode state
3. converts the previous `carry` tokens into prompt tokens
4. decodes audio starting at the beginning of `carry`
5. rebuilds `carry + preview` from the new decode

So `feed(11)` looks like this:

```text
before feed(11):
stable = 0.0 .. 1.2
carry  = 1.2 .. 1.6
preview= 1.6 .. 2.0

new audio arrives:
append 2.0 .. 2.2

ASR input for feed(11):
- retained KV for tokens covering 0.0 .. 1.2
- follow-up prompt
- replayed carry tokens from 1.2 .. 1.6
- audio chunk 1.2 .. 2.2

result:
- stable stays 0.0 .. 1.2
- carry is regenerated
- preview is regenerated and now extends through 2.2
```

The same pattern repeats:

| feed | utterance audio after append | KV kept before decode | replayed carry tokens | audio fed to ASR | preview length after decode | cutter? |
|---|---|---|---|---|---:|---|
| 11 | `0.0 .. 2.2` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 2.2` | `0.6s` | no |
| 12 | `0.0 .. 2.4` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 2.4` | `0.8s` | no |
| 13 | `0.0 .. 2.6` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 2.6` | `1.0s` | no |
| 14 | `0.0 .. 2.8` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 2.8` | `1.2s` | no |
| 15 | `0.0 .. 3.0` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 3.0` | `1.4s` | no |
| 16 | `0.0 .. 3.2` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 3.2` | `1.6s` | no |
| 17 | `0.0 .. 3.4` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 3.4` | `1.8s` | no |
| 18 | `0.0 .. 3.6` | `stable = 0.0 .. 1.2` | tokens for `1.2 .. 1.6` | `1.2 .. 3.6` | `2.0s` | yes |

At `feed(18)`, `preview` is large enough again, so the cutter runs again.

If it accepts a later boundary, more material moves from the right edge into
`stable`, a new `carry` is chosen behind the seam, and the remaining right
edge becomes the new `preview`.

In other words, the cycle is:

```text
rewrite preview
rewrite preview
rewrite preview
rewrite preview
cut
rewrite preview
rewrite preview
...
```

The important part is that the tape returned from `feed()` is always the
whole current tape, but the next decode step only keeps KV for `stable`,
replays `carry` as prompt tokens, and reopens `preview`.

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
- stable/carry/preview partitioning
- KV truncation
- stage-to-stage references

Timings, word structure, and phones are useful for deciding where a cut
should land, but the system should not become phone-indexed or word-indexed
internally just because those views are useful.

## What This Crate Owns

At a high level, `Utterance` owns:

- append-only utterance audio
- the stable/carry/preview partition over the current tape
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
- repeated re-decode of `preview`
- `stable` / `carry` / `preview` as the primary mental model
- cut search only when `preview` is large enough
- richer per-token output because later stages need it

What is not implemented yet is the actual concrete decode loop that wires in
Qwen3 ASR, G2P, ZIPA, and the cut decision process.
