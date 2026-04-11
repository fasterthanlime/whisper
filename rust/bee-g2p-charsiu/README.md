# bee-g2p-charsiu

`bee-g2p-charsiu` is the forward-looking home for Bee's non-GPL G2P story.

Right now it is deliberately a skeleton:

- a small Rust client exists, but real inference still lives in Python
- Python probes and sidecars are expected here first
- the immediate job is to understand the model and settle the strategy
- the later job is to move the useful parts into Rust and eventually MLX

This crate exists so we can stop pretending the current eSpeak-based path is
good enough.

The concrete intermediate model now lives in
[MODEL.md](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/MODEL.md).

## Current Rust Surface

This crate now has a minimal usable Rust surface:

- a persistent sidecar client in [lib.rs](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/src/lib.rs)
- a tiny CLI in [main.rs](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/src/main.rs)

The current job of that Rust code is simple:

- spawn the Python Charsiu sidecar
- send batches of words
- get back IPA strings
- or split a text into words, keep spans, and phonemize those words
- or run the cross-attention probe and return typed ownership data
- or collapse that ownership into model-facing token-piece IPA spans
- or collapse those spans into transcript-side comparison phones
- or flatten those comparison phones into the exact token stream shape the
  aligner wants

Example:

```bash
eval "$(direnv export bash)"
cargo run -p bee-g2p-charsiu -- Facet Wednesday
cargo run -p bee-g2p-charsiu -- --text "For Jason, this Thursday, use Facet."
cargo run -p bee-g2p-charsiu -- --probe-text "use Facet"
cargo run -p bee-g2p-charsiu -- --token-spans-text "use Facet"
cargo run -p bee-g2p-charsiu -- --token-phones-text "use Facet"
cargo run -p bee-g2p-charsiu -- --comparison-tokens-text "use Facet"
```

Current output:

```text
charsiu ready model=charsiu/g2p_multilingual_byT5_tiny_16_layers device=mps
Facet      ˈfeɪsət
Wednesday  ˈwɛdnɪsdi

charsiu ready model=charsiu/g2p_multilingual_byT5_tiny_16_layers device=mps
0..3   For       ˈfɔɹ
4..9   Jason     ˈdʒeɪsən
11..15 this      ˈtʰɪs
16..24 Thursday  ˈθɝzdi
26..29 use       ˈjuz
30..35 Facet     ˈfeɪsət

text    use Facet
decoded_ipa      ˈjuzˈfeɪsət
run     0..5     ˈjuz         word=use      qwen=use
run     5..12    ˈfeɪs        word=Facet    qwen=ĠFac
run     12..15   ət           word=Facet    qwen=et

text    use Facet
decoded_ipa      ˈjuzˈfeɪsət
span    0..5     ˈjuz         word=use      token=use   surface=use
span    5..12    ˈfeɪs        word=Facet    token=ĠFac  surface= Fac
span    12..15   ət           word=Facet    token=et    surface=et

text    use Facet
decoded_ipa      ˈjuzˈfeɪsət
phones  use      word=use      token=use   raw=j u z     norm=j ʊ z
phones   Fac     word=Facet    token=ĠFac  raw=f eɪ s    norm=f ɛ ɪ s
phones  et       word=Facet    token=et    raw=ə t       norm=ə t

text    use Facet
decoded_ipa      ˈjuzˈfeɪsət
cmp     use      word=use      token=use   phone=j  span=0..1
cmp     use      word=use      token=use   phone=ʊ  span=1..2
cmp     use      word=use      token=use   phone=z  span=2..3
cmp      Fac     word=Facet    token=ĠFac  phone=f  span=0..1
cmp      Fac     word=Facet    token=ĠFac  phone=ɛ  span=1..2
cmp      Fac     word=Facet    token=ĠFac  phone=ɪ  span=1..2
cmp      Fac     word=Facet    token=ĠFac  phone=s  span=2..3
cmp     et       word=Facet    token=et    phone=ə  span=0..1
cmp     et       word=Facet    token=et    phone=t  span=1..2
```

That is intentionally modest.

It gives us a Rust entry point now, without forcing the MLX/runtime work too
early.

## Why This Exists

The current G2P path is not where Bee should end up.

Problems with the current path:

- it depends on `espeak-ng`, which is GPL and therefore not suitable as the
  long-term production dependency for Bee
- it gives us a word-level IPA result, but not a principled token-level story
- it encouraged the rest of the alignment stack to drift toward word-level
  abstractions when Bee actually cares about token-level boundaries

Bee wants:

- non-GPL G2P
- token-aware phonetic structure
- a route to token-level timing by combining transcript-side phonetics with
  ZIPA's audio-side phone timing
- eventually, a native inference path that fits the rest of the MLX-based
  stack

## What Charsiu Changes

The imported Charsiu model family is not just "another script that returns
 IPA".

It changes the shape of the problem:

- it is a seq2seq model, not a rule system
- it runs on ByT5, so the input is byte-level
- it predicts IPA as generated output
- it is intended to run on already-tokenized words with a language prefix

That does **not** mean it directly gives us token timing.

It **does** mean it plausibly gives us a model signal for how the word maps to
 its pronunciation, which eSpeak never gave us.

The important opportunity is this:

- input side: bytes of the word
- output side: bytes of the IPA string
- bridge: encoder-decoder cross-attention

That bridge may be good enough to segment a word's phonemes across Qwen token
pieces inside a single word.

## The Actual Goal

The goal is not merely:

- `word -> ipa`

The goal is:

- `transcript text -> IPA`
- `transcript text -> token-piece phonetic segmentation`
- `token-piece phonetic segmentation + ZIPA phone timing -> token-level timing`

That is the real pipeline Bee wants.

In other words:

1. Qwen3 ASR gives us transcript tokens
2. those tokens are grouped into words
3. Charsiu gives us word-level pronunciation
4. Charsiu internals hopefully give us enough signal to split that
   pronunciation across token pieces inside each word
5. ZIPA gives us timed phones from audio
6. the two phonetic views meet in the middle
7. Bee gets token-level timing and richer phonetic evidence

## Working Vocabulary

We need to be precise about the different layers.

### Word

A word is an orthographic unit like:

- `Facet`
- `Thursday`

Charsiu expects already-tokenized words.

### Qwen token piece

A Qwen token is a rollback / KV-cache boundary unit.

Examples:

- `ĠThursday` can be one token
- `Facet` can be split into `ĠFac` + `et`

Bee cares about these boundaries because:

- rollback happens in token space
- KV truncation happens in token space
- display and debugging want token-aware structure

### Phone / phoneme sequence

This is the transcript-side or audio-side phonetic material.

Examples:

- transcript-side G2P IPA
- audio-side ZIPA raw phones
- normalized comparison phones used for alignment

### Timing

ZIPA gives us timing for phone spans, not for Qwen token pieces.

So token-level timing is **derived** timing.

That derivation must not be fake.

## Current Strategy

The strategy is staged.

### Stage 1: Replace eSpeak as the transcript-side G2P source

This is the immediate migration:

- use Charsiu instead of eSpeak for `word -> IPA`
- keep the current system working while removing the GPL dependency from the
  intended future path

At this stage, we are still mostly learning.

### Stage 2: Probe Charsiu internals

This is the critical research step.

We need to answer:

- can we extract encoder-decoder attention or another useful latent signal?
- can that signal be used to map output IPA bytes back to input word bytes?
- can we then project those byte spans onto Qwen token-piece boundaries inside
  each word?

If the answer is yes, then Charsiu is not only a G2P replacement.
It becomes the transcript-side segmentation engine.

### Stage 3: Build token-piece phonetic segmentation

For a word like `Facet`:

- orthographic word: `Facet`
- Qwen tokens: `ĠFac | et`
- Charsiu IPA: something like `f a s ɪ t`

The desired output is something like:

- `ĠFac -> f a s`
- `et -> ɪ t`

This segmentation may not always be perfectly deterministic.
That is acceptable, as long as it is:

- principled
- inspectable
- good enough to support timing and debugging

### Stage 4: Meet ZIPA in phonetic space

Once transcript-side token-piece phonetics exist:

- normalize transcript-side phones
- normalize ZIPA phones
- align them
- project ZIPA timing back onto token pieces

That is the route to token-level timing without inventing boundaries out of
 thin air.

### Stage 5: Move useful inference into Rust and MLX

Only after the above is understood should we move to:

- Rust wrappers
- Rust-native interfaces
- MLX-native inference

The current crate is intentionally earlier than that.

## Current Python Probes

The practical contents of this crate right now are the Python probes in
[scripts](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/scripts).

Current inventory:

- `charsiu_g2p_sidecar.py`
  Minimal JSON sidecar around the Charsiu ByT5 checkpoint.
- `charsiu_g2p_compare.py`
  Migration/evaluation helper that compares Charsiu output against the current
  eSpeak baseline.
- `charsiu_cross_attention_probe.py`
  Model-inspection probe for decoder cross-attention over a single word.

That third script is the strategically important one.

It is the first direct probe for this question:

- can Charsiu help us segment a word's pronunciation across Qwen token pieces?

## First Concrete Observation

The first useful sanity check is `Facet`, because Qwen splits it:

```text
Facet -> Fac | et
```

The current cross-attention probe reports exactly that token split on the Qwen
side, together with the Charsiu output:

```text
word         : Facet
decoded ipa  : ˈfeɪsət
qwen pieces  :
  [0] 'Fac' 0..3 bytes 0..3 -> 'Fac'
  [1] 'et' 3..5 bytes 3..5 -> 'et'
```

And the Charsiu decoder cross-attention is not random mush.

At the raw byte level, the strongest decoder steps attach to the actual word
bytes:

```text
out[ 2] 'f' -> in[10] 'F' score=0.6479
out[ 3] 'e' -> in[11] 'a' score=0.5558
out[ 6] 's' -> in[12] 'c' score=0.5368
out[ 9] 't' -> in[14] 't' score=0.5959
```

More importantly, once that attention is summed over the Qwen token-piece byte
spans, it starts to look like the split we actually want:

```text
out[ 2] 'f'
  qwen mass: 0:'Fac'=0.7068 ; 1:'et'=0.0256

out[ 3] 'e'
  qwen mass: 0:'Fac'=0.7666 ; 1:'et'=0.0366

out[ 6] 's'
  qwen mass: 0:'Fac'=0.6100 ; 1:'et'=0.1676

out[ 7] ''
  qwen mass: 0:'Fac'=0.0735 ; 1:'et'=0.6811

out[ 8] ''
  qwen mass: 0:'Fac'=0.0370 ; 1:'et'=0.7108

out[ 9] 't'
  qwen mass: 0:'Fac'=0.0350 ; 1:'et'=0.6164
```

That is still not a finished segmentation story.

It is, however, the first concrete evidence that Charsiu may let us derive a
split like:

- `Fac -> feɪs`
- `et -> ət`

That is close enough to the real target to justify continuing down this path.

It is enough to justify the next step:

- inspect Charsiu at the byte/attention level
- map those byte-level attentions back to Qwen token-piece spans
- see whether a stable token-piece phonetic segmentation falls out

The corresponding sanity check on a single-token word behaves sensibly too.

For `Wednesday`, Qwen keeps the whole word as one token:

```text
qwen pieces  :
  [0] 'Wednesday' 0..9 bytes 0..9 -> 'Wednesday'
```

And every decoder step simply places its mass onto that one piece:

```text
out[ 5] 'd'
  qwen mass: 0:'Wednesday'=0.8631

out[ 9] 's'
  qwen mass: 0:'Wednesday'=0.8332

out[11] 'i'
  qwen mass: 0:'Wednesday'=0.8315
```

That is exactly what we want from the probe:

- split words can show internal structure
- unsplit words collapse cleanly to one token piece

## Phrase-Level Prompting Is Still Research Only

The obvious objection to all of the above is that probing isolated words may
throw away useful contextual pronunciation.

So the next question is: can we feed a short phrase, keep the contextual IPA,
and still recover useful ownership by word span and Qwen token-piece span?

The current probe now supports that shape too.

For `For Jason`, Charsiu produces one phrase-level IPA string:

```text
text         : For Jason
decoded ipa  : ˈfɔɹˈdʒeɪsən
word spans   :
  [0] 'For'
  [1] 'Jason'
qwen pieces  :
  [0] 'For'
  [1] 'ĠJason'
```

And the teacher-forced cross-attention mass moves from the first word to the
second in exactly the way we would want:

```text
out[ 2] 'f'
  word mass : 0:'For'=0.6933 ; 1:'Jason'=0.0372
  qwen mass: 0:'For'=0.6933 ; 1:'ĠJason'=0.1197

out[ 9] 'd'
  word mass : 0:'For'=0.0234 ; 1:'Jason'=0.7041
  qwen mass: 0:'For'=0.0234 ; 1:'ĠJason'=0.8185

out[18] 'n'
  word mass : 0:'For'=0.0193 ; 1:'Jason'=0.6696
  qwen mass: 0:'For'=0.0193 ; 1:'ĠJason'=0.6989
```

That is the first good sign that phrase-level prompting does not immediately
destroy recoverable boundaries.

More importantly, the contextual case still works when one of the words splits
internally across Qwen token pieces.

For `use Facet`, Qwen tokenizes the second word as `ĠFac | et`:

```text
text         : use Facet
decoded ipa  : ˈjuzˈfeɪsət
word spans   :
  [0] 'use'
  [1] 'Facet'
qwen pieces  :
  [0] 'use'
  [1] 'ĠFac'
  [2] 'et'
```

The phrase-level probe shows three useful phases:

```text
out[ 2] 'j'
  word mass : 0:'use'=0.6606 ; 1:'Facet'=0.0250
  qwen mass: 0:'use'=0.6606 ; 1:'ĠFac'=0.1096 ; 2:'et'=0.0184

out[ 8] 'e'
  word mass : 0:'use'=0.0493 ; 1:'Facet'=0.7882
  qwen mass: 0:'use'=0.0493 ; 1:'ĠFac'=0.7699 ; 2:'et'=0.0335

out[14] 't'
  word mass : 0:'use'=0.0143 ; 1:'Facet'=0.6712
  qwen mass: 0:'use'=0.0143 ; 1:'ĠFac'=0.0634 ; 2:'et'=0.6294
```

So the current evidence points in the direction we actually want:

- phrase-level prompting gives contextual pronunciation
- word ownership is still recoverable
- Qwen token-piece ownership inside a word is still visible

That does not prove the whole strategy.

It does make the research path more concrete: run Charsiu on chunks, then use
teacher-forced cross-attention to project the generated IPA back onto word and
token spans.

## Longer Chunks Still Carry Recoverable Ownership

The next obvious question is whether that phrase-level story survives a more
realistic short chunk.

For:

```text
For Jason, this Thursday, use Facet.
```

the model output is already a little ugly:
the model output differs from the per-word baseline:

```text
decoded ipa : ˌfɔɹˈdʒeɪsənˈtʰɪtʰɝsdaɪˈjuzəˌfeɪst
```

So this is not evidence that chunk-level prompting is "solved".

It is evidence that ownership is still mostly recoverable even when the
phrase-level output diverges from the per-word baseline.

The compact ownership trace from the probe looks like this:

```text
out[ 2] 'f' word[0]='For'       qwen[0]='For'
out[ 9] 'd' word[1]='Jason'     qwen[1]='ĠJason'
out[21] 't' word[2]='this'      qwen[3]='Ġthis'
out[31] 's' word[3]='Thursday'  qwen[4]='ĠThursday'
out[38] 'j' word[4]='use'       qwen[6]='Ġuse'
out[45] 'f' word[5]='Facet'     qwen[7]='ĠFac'
out[50] 't' word[5]='Facet'     qwen[8]='et'
```

That is the important result.

Even on this longer chunk:

- the rough word-level handoff remains intact
- the rough Qwen token-piece handoff remains intact
- the internal `ĠFac | et` split is still visible inside the phrase

So the current evidence points to a specific tradeoff:

- chunk-level prompting improves contextuality
- longer chunks may diverge from the per-word baseline
- but cross-attention ownership can stay useful even before IPA quality is
  fully settled

That is enough to keep pursuing chunk-level prompting as a research path.

It is **not** enough to make chunk-level prompting the baseline.

The current baseline remains:

- invoke Charsiu per word
- recover token-piece ownership inside that word
- build token-level timing on top of that

The phrase/chunk work is useful because it tells us what extra contextual
signal may exist, not because it has displaced the per-word plan.

## What We Are Not Doing

We are not:

- pretending word timings are enough
- pretending token timing can be guessed visually
- hardcoding a fake split of IPA across token pieces
- writing production Rust first and discovering the model later

This crate is for understanding first.

## The Likely Shape Of The Future API

Not final, but the intended direction is roughly:

```text
word text
  -> charsiu word ipa
  -> token-piece segmentation
  -> normalized comparison phones per token-piece span
```

And then:

```text
token-piece phonetic spans
  + ZIPA normalized phones
  + ZIPA phone timings
  -> token-level timing / evidence
```

The important thing is that the primitive should become token-addressed, even
if some phonetic payload spans multiple token pieces.

## Immediate Next Experiments

1. run Charsiu on real Bee examples and compare with current eSpeak output
2. inspect decoder cross-attention for words that split across Qwen tokens
3. see whether the `Facet -> ĠFac | et` style case can be segmented
4. test normalization and alignment on those segmented outputs
5. only then decide what stable Rust types belong in the real runtime crates
