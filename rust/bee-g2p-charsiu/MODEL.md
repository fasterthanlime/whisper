# Charsiu Token-Timing Model

This file describes the forward-looking intermediate model for Bee's
Charsiu-based G2P stack.

It is not an implementation file.

It is the shape we want the experiments, diagrams, and eventual runtime code to
converge on.

## Problem

Today we have:

1. Qwen3 transcript tokens
2. ZIPA timed phones from audio
3. transcript-side G2P phones at the **word** level

That is why the current alignment API is word-based.

The transcript-side phonetic material cannot be mapped back to Qwen token
pieces, so token-level timing cannot fall out of the current pipeline.

The new goal is:

- keep ZIPA as the audio-side timed phone source
- replace eSpeak with Charsiu as the transcript-side G2P source
- recover token-piece ownership inside each word
- make token-level timing a derived result of phonetic alignment

## Baseline Invocation Boundary

Charsiu should be invoked **per word**.

That is both:

- the documented intended use in the upstream project
- the cleaner baseline from our own experiments so far

Phrase-level prompting remains interesting as a research path, but it is not
the default strategy.

So the primary input boundary is:

```text
Qwen token tape
  -> group into words
  -> run Charsiu once per word
```

## Core Idea

The missing bridge is:

```text
word text
  -> Charsiu IPA
  -> Charsiu attention-backed ownership over token pieces inside that word
```

Once we have that, the rest of the pipeline becomes straightforward:

```text
token-piece phonetic spans
  + ZIPA timed phones
  -> aligned token-level timing
```

## Coordinate Systems

We need to keep the coordinate systems distinct.

### Transcript token coordinates

This is the canonical Bee coordinate system.

- one entry per Qwen token piece
- rollback and KV truncation happen here
- output timing must eventually land here

### Word coordinates

Words are a derived grouping over transcript tokens.

- useful as the invocation boundary for Charsiu
- not the final timing boundary

### Transcript phonetic coordinates

These are phones derived from text.

- initially one IPA string per word
- then subdivided into token-piece-owned spans inside that word

### Audio phonetic coordinates

These are phones derived from audio.

- ZIPA raw phones
- ZIPA normalized comparison phones
- ZIPA timing lives here first

## Intermediate Data Model

This is the conceptual model, not final Rust syntax.

### Transcript token tape

```text
TranscriptToken {
  token_index
  qwen_token
  surface_text
  char_start
  char_end
  word_index
}
```

Meaning:

- `token_index` is the utterance-global Qwen token boundary
- `surface_text` is the text slice for this token
- `word_index` says which orthographic word owns this token

### Word view

```text
TranscriptWord {
  word_index
  text
  char_start
  char_end
  token_start
  token_end
}
```

Meaning:

- this is a view over the token tape
- `token_start..token_end` gives the Qwen token pieces inside the word

### Charsiu word result

```text
WordIpa {
  word_index
  ipa_text
  ipa_bytes
}
```

This is the direct per-word G2P output.

### Token-piece phonetic ownership inside a word

```text
TokenPieceIpaSpan {
  word_index
  token_index
  ipa_start
  ipa_end
  ipa_text
  ownership_score
}
```

Meaning:

- this says which span of a word's IPA belongs to which Qwen token piece
- it is derived from Charsiu internals, not from ZIPA
- `ownership_score` exists because this is inferred, not guaranteed

For a split word like:

```text
Facet -> ĠFac | et
```

we want something like:

```text
token ĠFac -> feɪs
token et   -> ət
```

### Transcript comparison phones

```text
TranscriptTokenPhones {
  token_index
  raw_ipa_text
  normalized_phones
}
```

This is the transcript-side payload that meets ZIPA.

Important:

- one token can still own multiple normalized phones
- punctuation tokens may own none
- this is token-addressed, not word-addressed

### ZIPA phones

```text
ZipaPhone {
  phone_index
  raw_phone
  normalized_phone
  start_time
  end_time
}
```

This is already basically what ZIPA provides today.

### Token-aligned timing result

```text
AlignedTokenTiming {
  token_index
  start_time
  end_time
  zipa_phone_start
  zipa_phone_end
  confidence
}
```

Meaning:

- the final timing boundary is back in Qwen token space
- the timing comes from ZIPA
- the ownership comes from transcript-side token-addressed phonetics

## Pipeline

The intended pipeline is:

```text
Qwen transcript tokens
  -> group into words
  -> Charsiu per word
  -> recover token-piece IPA ownership inside each word
  -> normalize transcript-side token phones
  -> normalize ZIPA phones
  -> align transcript token phones with ZIPA phones
  -> read ZIPA times back onto Qwen tokens
```

## Why This Fixes The Current Problem

The current public API is word-based because the G2P stage destroys the token
boundary.

This model preserves it again:

- Charsiu still runs at the word boundary
- but the result is projected back onto token pieces
- so the alignment layer can return token-level timing instead of only
  word-level timing

That is the whole point.

## What This Does Not Require

This model does **not** require:

- Charsiu to emit timing directly
- ZIPA to know anything about Qwen tokens
- phrase-level G2P prompting as the default path

It only requires:

- per-word IPA from Charsiu
- a usable ownership signal from Charsiu internals
- token-addressed normalization/alignment above that

## Open Questions

1. What is the right transcript-side unit for ownership?
   - raw IPA byte span
   - IPA symbol span
   - normalized phone span

2. How should uncertain ownership be represented?
   - hard assignment
   - soft scores
   - merged token-span fallback

3. How much punctuation should participate?
   - probably none on the transcript-phonetic side

4. Do we need a confidence-bearing result at the token level?
   - probably yes
   - because inferred ownership is not equally trustworthy everywhere

## Immediate Implication For Diagrams

Once this model exists, the README diagrams should be built around:

- top: Qwen token pieces
- middle: token-owned transcript phonetics
- bottom: ZIPA timed phones

And the meeting point is no longer "word alignment".

It is:

```text
token-addressed transcript phones <-> timed ZIPA phones
```
