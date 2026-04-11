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

One column below is one `200 ms` slice of utterance time.

```text
Legend

A = audio currently held by the utterance
S = stable tokens; KV for these stays alive
C = carry tokens; these are replayed into the next ASR step
P = preview tokens; these are the live tail
^ = audio passed to the ASR encoder on this feed
```

In this example, the cutter wakes up whenever `preview` reaches `10`
columns, that is `2.0 s`.

### Before the first cut

Before the first cut exists:

- there is no `stable`
- there is no `carry`
- there is no KV worth keeping
- every `feed()` re-decodes from the beginning

```text
feed(1)   append one 200 ms buffer
audio held : A
kv kept    :
carry txt  :
ASR audio  : ^
tape out   : P

feed(2)
audio held : AA
kv kept    :
carry txt  :
ASR audio  : ^^
tape out   : PP

feed(3)
audio held : AAA
kv kept    :
carry txt  :
ASR audio  : ^^^
tape out   : PPP

feed(4)
audio held : AAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^
tape out   : PPPP

feed(5)
audio held : AAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^
tape out   : PPPPP

feed(6)
audio held : AAAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^^
tape out   : PPPPPP

feed(7)
audio held : AAAAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^^^
tape out   : PPPPPPP

feed(8)
audio held : AAAAAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^^^^
tape out   : PPPPPPPP

feed(9)
audio held : AAAAAAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^^^^^
tape out   : PPPPPPPPP

feed(10)  preview just reached the cut-search threshold
audio held : AAAAAAAAAA
kv kept    :
carry txt  :
ASR audio  : ^^^^^^^^^^
raw out    : PPPPPPPPPP
cut        : SSSSSSCCPP
```

So after `feed(10)` the tape has been repartitioned like this:

```text
SSSSSSCCPP
```

That means, in this worked example:

- `stable` is `6` columns
- `carry` is `2` columns
- `preview` is `2` columns

### After the first cut

Now each `feed()` does something more interesting:

- keep KV for `stable`
- replay `carry` into the prompt
- re-decode audio starting at the beginning of `carry`
- rebuild `carry + preview`

```text
feed(11)
audio held : AAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^
tape out   : SSSSSSCCPPP

feed(12)
audio held : AAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^
tape out   : SSSSSSCCPPPP

feed(13)
audio held : AAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^
tape out   : SSSSSSCCPPPPP

feed(14)
audio held : AAAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^^
tape out   : SSSSSSCCPPPPPP

feed(15)
audio held : AAAAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^^^
tape out   : SSSSSSCCPPPPPPP

feed(16)
audio held : AAAAAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^^^^
tape out   : SSSSSSCCPPPPPPPP

feed(17)
audio held : AAAAAAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^^^^^
tape out   : SSSSSSCCPPPPPPPPP

feed(18)  preview just reached the cut-search threshold again
audio held : AAAAAAAAAAAAAAAAAA
kv kept    : SSSSSS
carry txt  :       CC
ASR audio  :       ^^^^^^^^^^^^
raw out    : SSSSSSCCPPPPPPPPPP
cut        : SSSSSSSSSSSSCCPPPP
```

After `feed(18)`, the seam has moved right:

```text
old : SSSSSSCCPPPPPPPPPP
new : SSSSSSSSSSSSCCPPPP
```

That is the entire loop:

```text
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
cut
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
rewrite preview
cut
...
```

One more step, just to show the new seam taking effect immediately:

```text
feed(19)
audio held : AAAAAAAAAAAAAAAAAAA
kv kept    : SSSSSSSSSSSS
carry txt  :             CC
ASR audio  :             ^^^^^^^
tape out   : SSSSSSSSSSSSCCPPPPP
```

That is the geometry `bee-roll` is trying to express:

- the whole current tape is returned every time
- only `stable` survives in KV
- `carry` is replayed as prompt text
- `preview` is reopened and regenerated
- when `preview` gets large enough, the cutter promotes more of the right
  edge into `stable` and chooses a new `carry`

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
