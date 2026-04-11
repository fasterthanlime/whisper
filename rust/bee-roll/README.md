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

This section should be generated, not hand-aligned.

The sample below was produced by:

```bash
python3 rust/bee-roll/scripts/render_timeline.py \
  rust/bee-roll/scripts/timeline_demo.json
```

The current demo uses:

- one `200 ms` feed interval
- real words on the transcript row
- token pieces on the token row
- `stable` / `carry` / `preview` on the tape rows
- separate rows for kept KV, replayed carry text, and ASR audio span

```text
feed(10)  first cut search
time       : 0.0s                1.0s
ticks      : |   |   |   |   |   |   |   |   |   |  |
audio held : ========================================
words      :  this  is  a  long   utterance  that wa
tokens     : .this..is..a..long..utter..ance.that.wa.
kv kept    :
carry txt  :
ASR audio  : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
raw tape   : PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
cut tape   : SSSSSSSSSSSSSSSSSSSSCCCCCCCCCCCCPPPPPPPP

feed(11)  first rewrite after the cut
time       : 0.0s                1.0s                2.0s
ticks      : |   |   |   |   |   |   |   |   |   |   |  |
audio held : ============================================
words      :  this  is  a  long   utterance  that warr...
tokens     : .this..is..a..long..utter..ance.that..warr..
kv kept    : ####################
carry txt  :                      utterance
ASR audio  :                     ^^^^^^^^^^^^^^^^^^^^^^^^
tape out   : SSSSSSSSSSSSSSSSSSSSCCCCCCCCCCCCPPPPPPPPPPPP

feed(18)  second cut search
time       : 0.0s                1.0s                2.0s                3.0s
ticks      : |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |  |
audio held : ========================================================================
words      :  this  is  a  long   utterance  that   warrants    several     rounds
tokens     : .this..is..a..long..utter..ance.that..war..rants..sev...eral...rounds...
kv kept    : ####################
carry txt  :                      utterance
ASR audio  :                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
raw tape   : SSSSSSSSSSSSSSSSSSSSCCCCCCCCCCCCPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
cut tape   : SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCCCCCCCCCCCCPPPPPPPPPPP

feed(19)  first rewrite after the second cut
time       : 0.0s                1.0s                2.0s                3.0s
ticks      : |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |  |
audio held : ============================================================================
words      :  this  is  a  long   utterance  that   warrants    several     rounds now
tokens     : .this..is..a..long..utter..ance.that..war..rants..sev...eral..rounds...now..
kv kept    : #################################################
carry txt  :                                                    several
ASR audio  :                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
tape out   : SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCCCCCCCCCCCCPPPPPPPPPPPPPPP
```

That is the geometry `bee-roll` is trying to express:

- the whole current tape is returned every time
- only `stable` survives in KV
- `carry` is replayed as prompt text
- `preview` is reopened and regenerated
- when `preview` gets large enough, the cutter promotes more of the right
  edge into `stable` and chooses a new `carry`

## How Qwen3 Prompting Works Right Now

This section describes the current local implementation in
`bee-qwen3-asr` and `bee-kv`. It is not a theory section.

Two details matter up front:

1. the decoder is driven with chat-style prompt tokens
2. the prompt contains audio placeholders, but the model does not consume
   raw PCM "as tokens"

The real flow is:

1. audio samples are converted into mel features
2. the audio encoder turns those into audio feature vectors
3. the prompt contains `<|audio_pad|>` placeholders
4. at prefill time, those placeholder embedding positions are replaced with
   the encoded audio features

So the prompt is text-and-special-tokens structure around an injected audio
feature sequence.

### Initial Prompt

For the first decode step, `bee-qwen3-asr` builds this shape:

```text
<|im_start|>system
{context}
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>*N<|audio_end|>
<|im_end|>
<|im_start|>assistant
[language {lang}<asr_text>]
```

Notes:

- `{context}` is optional
- the `language {lang}<asr_text>` header is only present when language is
  non-empty
- `N` is the number of audio feature positions produced by the audio encoder,
  not the number of PCM samples

This is built by:

- [generate.rs](/Users/amos/bearcove/bee/rust/bee-qwen3-asr/src/generate.rs)

### Follow-Up Prompt

For later decode steps, the base follow-up prompt is:

```text
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>*N<|audio_end|>
<|im_end|>
<|im_start|>assistant
[language {lang}<asr_text>]
```

That means the follow-up prompt explicitly closes the previous assistant
turn, opens a new user turn with the next audio chunk, then opens a new
assistant turn.

### Where `carry` Goes

The replayed `carry` text is not baked into `build_followup_prompt()`
itself.

Instead, `bee-kv` appends the carried token IDs after the follow-up prompt
has been built:

```text
followup prompt
+ replayed carry token ids
--------------------------------
final prefill prompt for this step
```

So the effective follow-up prompt shape is:

```text
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>*N<|audio_end|>
<|im_end|>
<|im_start|>assistant
[language {lang}<asr_text>]
{carry token ids}
```

That is the concrete mechanism behind the `stable / carry / preview` model:

- `stable` survives in KV and is not replayed as text
- `carry` is replayed as prompt tokens
- `preview` is generated again after that prompt

### Why This Matters For `bee-roll`

`bee-roll` should model the actual split implied by the current decoder
setup:

- preserved prefix in KV
- replayed bridge text in prompt tokens
- regenerated right edge after the prompt

That is why `carry` is a distinct region instead of just "the first part of
preview".

## How G2P And ZIPA Fit In

`bee-roll` does not care about phonetics as an abstract add-on.

It cares because phonetic structure is useful for:

- choosing coherent cut boundaries
- powering correction later
- powering the debugging visualization

There are two different phonetic views in this codebase:

1. G2P: transcript-derived IPA
2. ZIPA: audio-derived phone output with timing

They are not interchangeable.

### G2P: Text-Derived IPA

The current G2P implementation lives in:

- [g2p.rs](/Users/amos/bearcove/bee/rust/bee-correct/src/g2p.rs)

The main type is `CachedEspeakG2p`, backed by bundled `espeak-ng`.

What it can produce:

- `ipa_tokens(text) -> Option<Vec<String>>`
  IPA tokens for a text span
- `ipa_word_tokens_in_utterance(text) -> Option<Vec<Vec<String>>>`
  per-word IPA token groups for an utterance

The second form matters more here.

It does not phonemize each word in isolation. It walks utterance prefixes and
computes each word's contribution by subtracting the longest common IPA
prefix from the previous prefix result.

That means the per-word G2P output is still contextualized by the utterance,
which is useful for boundary selection and later correction work.

So the current G2P side is:

- transcript in
- IPA token strings out
- optionally grouped per word in utterance order

### ZIPA: Audio-Derived Phones

The low-level ZIPA implementation lives in:

- [infer.rs](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs)

What ZIPA inference actually exposes:

- `InferenceOutput`
  - `log_probs`
  - `log_probs_len`
  - `token_ids`
  - `tokens`
- `GreedyInferenceOutput`
  - `frame_count`
  - `token_ids`
  - `tokens`
- `PhoneSpan`
  - `token_id`
  - `token`
  - `start_frame`
  - `end_frame`
  - `start_time_secs`
  - `end_time_secs`

So ZIPA is not just "an IPA string".

It gives:

- frame-level token decisions
- decoded phone token strings
- time-bearing phone spans derived from frame runs

In other words:

- audio in
- phone-like token stream out
- timed phone spans out

That timing is exactly why ZIPA is useful for cuts and visualization.

### Current Alignment Layer

The current code that combines transcript-side G2P with audio-side ZIPA
lives in:

- [zipa_align.rs](/Users/amos/bearcove/bee/rust/bee-transcribe/src/zipa_align.rs)

Its current `build()` flow is:

1. run ZIPA greedy inference over the audio
2. derive ZIPA phone spans
3. normalize the ZIPA phone tokens for comparison
4. run utterance-level G2P over the transcript
5. normalize the G2P tokens for comparison
6. align transcript-side and ZIPA-side token sequences
7. derive per-word timing windows / span timings from the aligned phone spans

That alignment layer is useful, but it is not the whole story.

For `bee-roll`, the important thing is to remember that the richer raw data
exists underneath it:

- G2P word/token IPA
- ZIPA phone tokens
- ZIPA timed phone spans
- ZIPA frame-level evidence

That is why `bee-roll` should think in terms of storing owned per-token
phonetic payload, not just "some final aligned word timing."

### What `bee-roll` Should Care About

The rough split is:

- tokens remain the canonical rollback boundary
- G2P provides transcript-side phonetic structure
- ZIPA provides audio-side phonetic structure and timing

That richer phonetic output can then support:

- the cutter
- the debug visualization
- the correction pipeline

Without forcing the whole rollback model to become phone-indexed.

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
