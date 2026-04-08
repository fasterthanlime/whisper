# Pipeline Redesign: Two-Stage Architecture with Proper Types

## Problem

The current bee-transcribe decomposition (SpeechGate, Generator, Aligner, Corrector as separate structs) has fundamental issues:

- **RotateInstruction** returns values and hopes the caller applies them correctly. Finish bug: `finish_commit` returned wrong `raw_tokens_to_drop`, caller rotated wrong amount, `make_update` double-counted tokens.
- **Parallel vectors** everywhere: `raw_token_ids` + `raw_token_logprobs` in Generator, `committed_text_tokens` + `committed_logprobs` in Aligner. Easy to get out of sync.
- **Raw primitives** for unrelated quantities: `usize` for token counts, sample counts, byte offsets. Easy to swap.
- **Leaked internal state**: `metadata_token_count` threaded through from StructuredAsrOutput to Aligner to RotateInstruction. Callers must coordinate state they shouldn't know about.
- **Committed byte offset** is fragile. Corrections change text length. Token index is the only stable reference.

## Architecture

Two stages. Stage 1 is ASR (one struct, methods on `&mut self`). Stage 2 is correction (separate, async/background). They communicate through a `TextBuffer`.

### Foundation Types

#### `AudioBuffer`

Not raw `Vec<f32>`. A proper type that knows its sample rate, supports splitting by time, trimming, appending. Assertions on sample rate when combining buffers.

```
AudioBuffer {
    samples: Vec<f32>,
    sample_rate: SampleRate,
}
```

Operations: split at `Seconds`, trim before `Seconds`, append, slice a `TimeRange`, duration, len, is_empty.

#### `Seconds`

Newtype for time. Not raw `f64`. Supports arithmetic (Add, Sub), conversion to/from sample counts given a `SampleRate`.

#### `SampleRate`

Newtype wrapping `u32`. Not a raw number.

#### `TimeRange`

`{ start: Seconds, end: Seconds }`. Used for word timing, audio slicing.

#### `AsrToken`

Replaces all parallel `(Vec<TokenId>, Vec<TokenLogprob>)` pairs.

```
AsrToken {
    id: TokenId,
    logprob: f32,
    margin: f32,
}
```

#### `TokenCount`

Newtype wrapping `usize`. For commit thresholds, rollback counts. Never confused with sample counts or byte offsets.

#### `TextBuffer`

The core data structure. A flat vector of token entries with an optional word boundary marker. Text is decoded on demand from the tokenizer, never cached as a String.

```
TextBuffer {
    entries: Vec<TokenEntry>,
    committed: TokenIndex,  // entries[..committed] are stable + aligned
                            // entries[committed..] are pending, rewritten each feed()
}

TokenEntry {
    token: AsrToken,
    word: Option<WordStart>,  // Some = this token starts a new word
}

WordStart {
    alignment: Option<WordAlignment>,  // None = not yet aligned
}

WordAlignment {
    time: TimeRange,       // set together, always
    audio: AudioBuffer,    // set together, always
}
```

To see all tokens for a word: start at a `WordStart`, iterate until the next `WordStart`. Confidence computed on demand from the tokens' logprobs. Text decoded on demand.

Committed vs pending is a token index, not a byte offset. On feed(): entries before committed are untouched, entries after committed are replaced. On commit: alignment filled in, committed advances.

### Stage 1 (ASR) — One Struct

```
Session {
    filters: AudioFilterChain,  // VAD, normalization, etc.
    text: TextBuffer,
    detected_language: String,
    decoder: DecodeSession,
    engine: &Engine,
}
```

**AudioFilterChain**: Composable audio processing. Push samples in, processed audio comes out (or nothing, if gated by VAD). Session doesn't know what's in the chain. VAD runs before normalization (VAD trained on raw audio).

```
trait AudioFilter {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer>;
}
```

**DecodeSession**: Replaced on rotation. Owns the audio being decoded and the decoder state. Rotation = throw away old, make new. Can't get it wrong.

```
DecodeSession {
    audio: AudioBuffer,          // audio for this sub-session
    encoder_cache: EncoderCache,
    mel_extractor: MelExtractor,
    tokens: Vec<AsrToken>,       // current tokens (metadata + text)
    metadata_end: usize,
    start_time: Seconds,         // when this sub-session starts in timeline
}
```

The audio and tokens live together — same struct, same lifetime. `start_time` provides the offset for converting alignment timestamps to absolute. When we rotate, the new session gets `start_time = old.start_time + cut_point`.

**Feed/finish share code.** Common `decode_and_maybe_commit(final_pass: bool)` method. `final_pass` = higher token budget + commit everything. No code duplication.

**Commit is internal.** The aligner logic (deciding when to commit, running forced alignment, filling in word alignments, trimming audio, replacing decode session) is all methods on Session. No RotateInstruction returned. No external coordination.

On commit:
1. Run forced alignment on decoder.audio for N text tokens
2. Build WordAlignments (time + audio slices from decoder.audio)
3. Fill alignment into text buffer entries
4. Advance text.committed
5. decoder.audio.trim_before(last_word_end)
6. Replace decoder with fresh DecodeSession (start_time advanced)

On finish: same thing but commit ALL remaining tokens, then decoder is empty.

**make_update**: decode committed tokens for text, decode pending tokens for text, concatenate. Committed length = byte length of committed text. Alignments = words that have alignment filled in.

### Stage 2 (Correction) — Separate, Async

Receives the TextBuffer from stage 1. Each aligned word carries its own AudioBuffer. Stage 2 never reaches back into stage 1.

Stage 2 tracks a token index into stage 1's TextBuffer: "I've corrected up to token N." It produces its own corrected text for that region.

Correction runs in the background:
- On each feed(): check if background work finished (merge results), check if enough new committed words (kick off new batch)
- Needs right context: can't correct the rightmost committed word (waits for next commit)
- IPA needs audio context wider than a single word (use neighboring words' audio)
- On finish: wait for in-flight work, flush remaining corrections (best effort)

Output stitching:
```
correction_text(0..corrected_up_to) + decode(corrected_up_to..committed) + decode(committed..end)
```

All references to stage 1 are token indices. Corrections produce separate text. The two never interfere.

## Key Invariant

**Stage 2 correction decisions act only on the committed token prefix.** The live tail (pending tokens beyond the committed boundary) is for display/stitching only — it may change on every feed() call and must never be used for correction decisions. This invariant prevents stale-reference and double-counting bugs. It must be stated explicitly in code and docs.

Presentation combines: corrected committed text + uncorrected committed text (awaiting right context) + live uncorrected tail.

## Boundary Rules

- **Internal source of truth**: token indices. All stage-to-stage references, committed boundaries, correction progress tracking use token indices.
- **External/UI source of truth**: character spans. The mapping from token indices to character ranges happens at the presentation boundary (when building output for Swift), not in the core logic.
- **Audio copies per word**: acceptable. Don't complicate the design to avoid small copies unless profiling proves they matter.
- **Decode-on-demand**: acceptable. Tokens are canonical. No premature text caching on the Rust side.
- **Audio filter chain**: separate signal hygiene (clipping, DC cleanup) from heavier processing (VAD, normalization) conceptually, so they're not mixed.

## Implementation Order

1. **AudioBuffer, Seconds, SampleRate, TimeRange** — pure types, unit tested in isolation
2. **AsrToken, TokenCount, TokenIndex** — trivial newtypes
3. **TextBuffer** (TokenEntry, WordStart, WordAlignment) — unit tested with synthetic tokens
4. **DecodeSession** — wraps audio + decoder state
5. **AudioFilter trait + VAD filter + normalization filter** — tested independently
6. **Session (stage 1)** — methods on one struct, uses foundation types
7. **CorrectionStage (stage 2)** — async background, reads TextBuffer
8. **Integration tests** — feed real audio through both stages, assert no duplication, assert alignments carry audio

## Current Bugs to Fix Along the Way

- **Duplication bug in finish()**: gone by construction (decoder replaced, no tokens to double-count)
- **Mid-speech silence passes through VAD**: speech gate should handle end-of-speech detection properly
- **No audio normalization**: add as a filter in the chain
- **`committed_len` as byte offset**: replaced with token index
