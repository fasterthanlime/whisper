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

### What The Data Actually Looks Like

The current G2P implementation is
[g2p.rs](/Users/amos/bearcove/bee/rust/bee-correct/src/g2p.rs).
The low-level ZIPA implementation is
[infer.rs](/Users/amos/bearcove/bee/rust/bee-zipa-mlx/src/infer.rs).

The most useful way to understand them is to look at a real example.

This one is generated by:

```bash
python3 rust/bee-roll/scripts/render_phonetic_compare.py --term serde
```

That script calls the real `bee-zipa-mlx` comparison binary, which in turn
runs actual G2P and actual ZIPA inference on a real recording.

```text
term       : serde
text       : For Jason, this Thursday, but we could also use Facet.

raw
g2p        : f ɔː dʒ eɪ s ə n ð ɪ s θ ɜː z d eɪ b ə t w iː k ʊ d ɔː l s əʊ j uː s f a s ɪ t
zipa       : f ɹ ə d ʒ e ɪ s ə n ð ɪ s s ə d e ɪ b ə l k ʊ d ɔ l s o ʊ j u s f æ s ɪ t

normalized + aligned
g2p  : f · ɔ d ʒ ɛ ɪ s ə n ð ɪ s θ
zipa : f ɹ ə d ʒ ɛ ɪ s ə n ð ɪ s s
diff : | > x | | | | | | | | | | x

g2p  : ə z d ɛ ɪ b ə t w ɪ k ʊ d ɔ
zipa : ə · d ɛ ɪ b ə · · l k ʊ d ɔ
diff : | < | | | | | < < x | | | |

g2p  : l s ə ʊ j ʊ s f a s ɪ t
zipa : l s ə ʊ j ʊ s f ɛ s ɪ t
diff : | | | | | | | | x | | |

scores
raw        : 0.4703
normalized : 0.7897
feat norm  : 0.8754
```

That is much closer to the real problem:

- G2P is transcript-side and stressy / lexicon-like
- ZIPA is audio-side and no-diacritics
- the raw forms differ a lot
- normalization makes them comparable enough to be useful

Here is a second real example:

```bash
python3 rust/bee-roll/scripts/render_phonetic_compare.py --term SQLite
```

```text
term       : SQLite
text       : I trust a small SQL database more than the JSON L file.

raw
g2p        : aɪ t ɹ ʌ s t eɪ s m ɔː l s k d eɪ t ə b eɪ s m ɔː ð ɐ n ð ə dʒ s ɒ n ɛ l f aɪ l
zipa       : a ɪ t ɹ ə s t u s m ɔ l s i k ɹ ə t ɪ ŋ d e ɪ t ə b e ɪ s m ɔ ɹ ð æ n ð ə d ʒ e ɪ s ə n t ɛ l f a ɪ

normalized + aligned
g2p  : a ɪ t ɹ ə s t ɛ ɪ s m ɔ l s
zipa : a ɪ t ɹ ə s t · ʊ s m ɔ l s
diff : | | | | | | | < x | | | | |

g2p  : · k · · · · · d ɛ ɪ t ə b ɛ
zipa : ɪ k ɹ ə t ɪ ŋ d ɛ ɪ t ə b ɛ
diff : > | > > > > > | | | | | | |

g2p  : ɪ s m ɔ ð ə n ð ə d ʒ · · s
zipa : ɪ s m ɔ ð ɛ n ð ə d ʒ ɛ ɪ s
diff : | | | | | x | | | | | > > |

g2p  : ɑ n · ɛ l f a ɪ l
zipa : ə n t ɛ l f a ɪ ·
diff : x | > | | | | | <

scores
raw        : 0.4680
normalized : 0.7102
feat norm  : 0.7657
```

### The Different IPA Dialects

There are at least three useful phonetic "dialects" in play:

```text
reviewed compact IPA : sˈɜːdeɪ
parsed phones        : s ɜː d eɪ

espeak probe output  : sˌɜː dˈe ɪ
parsed phones        : s ɜː d e ɪ

ZIPA style output    : no stress marks, no length marks, phone tokens with timing
```

The first two examples are from the existing tokenizer tests in
`bee-phonetic`. The last line is the important practical point about ZIPA:
its output is already in a different dialect before alignment even starts.

### What Normalization Actually Does

Normalization is what makes these dialects comparable.

Some concrete examples from the existing tests:

```text
affricates and diphthongs
  tʃ      -> t ʃ
  d͡ʒ     -> d ʒ
  aɪ      -> a ɪ
  eɪ      -> ɛ ɪ
  oʊ/əʊ   -> ə ʊ

vowel-family collapsing
  ɐ, ʌ, ɜ -> ə
  ɑ, ɒ    -> ɑ
  e, ɛ    -> ɛ
  i, ɪ    -> ɪ
  u, ʊ    -> ʊ

rhoticity collapsing
  ə ɹ     -> ə
  ɚ       -> ə

centering diphthongs
  eə      -> ɛ
  ɪə      -> ɪ
  ʊə      -> ʊ
```

This is why the normalized similarity numbers in the real examples above are
much better than the raw similarity numbers.

### How Raw ZIPA Phones Map Back To Time

ZIPA starts with frame-level output and collapses repeated non-blank phones
into `PhoneSpan`s.

Then the alignment layer normalizes those phones while preserving where they
came from.

A small example from the current tests:

```text
raw ZIPA phone spans
  t   0.10 .. 0.12
  ʃ   0.12 .. 0.14
  aɪ  0.14 .. 0.15

normalized comparison tokens
  t   source 0..1
  ʃ   source 1..2
  a   source 2..3
  ɪ   source 2..3
```

So one raw phone like `aɪ` can expand into two normalized comparison tokens,
but both still point back to the same raw phone span and therefore the same
time region.

That is the bridge between:

- raw ZIPA timing
- normalized comparison tokens
- later per-word or per-span timings

### What `bee-roll` Should Care About

The short version is:

- tokens stay canonical for rollback
- G2P gives transcript-side phonetic structure
- ZIPA gives audio-side phonetic structure and timing
- normalization makes the two sides comparable
- alignment projects that comparison back onto time

That is enough to support cuts, debugging, and later correction without
making the whole rollback model phone-indexed.

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
