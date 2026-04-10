# Progressive ASR with Rollback and Persistent KV Cache

## Purpose

This note records the discussion so far about progressive ASR decoding, why naive full reruns are wasteful, why pure append-only KV reuse is not enough, and why the more interesting direction is rollback with selective cache reuse.

This document is intentionally written from first principles and from the architecture discussion only. It does not inspect or rely on implementation details from `bee-transcribe`.

## Basic Model of ASR

At the highest level, an ASR system consumes audio and produces text tokens. That framing is directionally correct, but it hides a crucial architectural split.

The audio is usually not consumed by the language model as raw PCM in the same sense that text is consumed as prompt tokens. Instead, the system first transforms audio into acoustic features or acoustic latent representations. Those audio-derived states then condition text generation.

So the real shape is:

- audio waveform in
- feature extraction or acoustic encoding
- latent acoustic sequence
- autoregressive text decoding conditioned on those acoustic states
- text tokens out

That distinction matters because when talking about KV caching, there is not just one state to care about. There is text-side autoregressive state, and there is also audio-side encoded state.

## Qwen3 ASR Architecture in This Discussion

The model discussed here is not a single plain LM that directly accepts audio samples as if they were text tokens. It is a speech stack with an acoustic front-end and a text-generating decoder/LM.

From the diagram discussed:

- audio waveform is converted into filterbank-style features at roughly `100 Hz`
- an `AuT Encoder` processes those features and compresses them into a hidden sequence at roughly `12.5 Hz`
- an autoregressive decoder attends to those audio states and emits transcript text
- in the broader setup, a `Pretrained AuT Encoder` feeds audio-derived embeddings into a `Qwen3 LM`
- the LM then generates transcript tokens conditioned on both text context and those audio-derived representations

So the architecture has a clear separation:

- acoustic side: derive a usable representation from audio
- autoregressive text side: predict transcript tokens while attending to that representation

That is the backdrop for every reuse question. It is possible that different parts of the pipeline admit different reuse strategies.

## Live Dictation Baseline: Full Prefix Reruns

The baseline scenario discussed was live dictation over six seconds of speech, with visible progress updates every two seconds.

The naive implementation would do three separate inferences:

1. decode audio from `0s` to `2s`
2. decode audio from `0s` to `4s`
3. decode audio from `0s` to `6s`

Each of those runs starts from fresh state. That means recomputing everything for the repeated prefix:

- repeated feature extraction for earlier audio
- repeated acoustic encoding for earlier audio
- repeated autoregressive decoding from a cold cache
- repeated regeneration of transcript hypotheses for the same early segment

This is correct in the sense that each run can reconsider earlier text in light of later audio. It is also expensive because the same early audio is processed multiple times.

Conceptually:

- `0..2` is processed three times
- `0..4` is processed twice
- only `4..6` is processed once

This is the quality-oriented baseline, not the latency-oriented one.

## Earlier Session Rotation Strategy

Before talking about persistent KV reuse, the discussion introduced an older strategy for making progressive decoding workable.

That strategy was not to fully freeze the already-produced text. Instead, it kept some recent text revisable, while also carrying a small amount of fixed earlier text as explicit context into the next decode.

In the simplified example:

- the latest transcript already contains some decoded text
- the system decides that the last `12` text tokens are unstable and should be reconsidered
- those last `12` tokens are rolled back
- another `12` tokens before that are kept as a fixed prefix and explicitly fed again as context
- decoding resumes from there

The purpose of that fixed prefix was to help the model stay grounded in what came before the rollback boundary, without asking it to regenerate the entire transcript from scratch.

That strategy already assumes a split between:

- a stable prefix that is treated as settled enough to keep
- a recent suffix that is allowed to change

What persistent KV reuse changes is not the conceptual split. It changes how much explicit replay is needed.

## Why Pure Append-Only KV Reuse Is Not the Right Comparison Target

A tempting optimization is:

1. decode `0..2`
2. keep the resulting KV cache
3. append audio `2..4` and continue decoding from the existing cache
4. append audio `4..6` and continue again

That is cheaper, but it is not the same computation as full-prefix reruns.

A pure append-only continuation tends to lock in earlier textual decisions. In ASR, that is often undesirable because later audio frequently disambiguates earlier audio. If the model has already committed to one textual interpretation, continuing from that exact autoregressive state can make it difficult or impossible to revise the earlier choice.

So append-only reuse is not just an optimization. It changes the behavior:

- lower latency
- less recomputation
- less revision power
- more dependence on earlier mistakes

That is why the more interesting direction is not "reuse everything and only append" but rather "reuse what should remain fixed, invalidate what should be reconsidered, and regenerate from there."

## The New Direction: Rollback with Persistent KV Cache

The new direction discussed is more precise than a general sliding-window stitcher.

It is not:

- decode overlapping windows
- get several competing local transcripts
- stitch arbitrary token spans together afterward

Instead, it is:

- maintain a committed prefix
- maintain a revisable tail
- periodically roll back a suffix of recent transcript state
- keep older, more stable state intact
- regenerate from the rollback point forward using newly available audio

That means the unit of editing is not an arbitrary sliding-window merge. It is a deliberate truncation of the recent decode state followed by fresh generation from that cut point.

This matters because it preserves a single canonical transcript state with controlled local revisions, rather than introducing a general-purpose text-stitching problem across overlapping hypotheses.

A representative flow looks like this:

1. decode some audio and obtain transcript text
2. decide that the last region is still unstable
3. roll back far enough to cover that unstable region
4. preserve all state before that rollback boundary
5. continue decoding with new audio from the rollback boundary onward

This gives the model room to revise recent decisions while still avoiding full-prefix recomputation.

## Why This Direction Is Attractive

Several things already discussed make this direction stronger than naive window stitching.

### 1. It matches the actual product goal

The goal here is low-latency, near-real-time visible text. Final accuracy is less important than responsiveness. Visible rewrites are acceptable. A suffix-revision strategy is a direct fit for that requirement.

### 2. It preserves a single evolving transcript

Instead of treating each window as a separate hypothesis that must later be reconciled, this strategy maintains one transcript with a stable prefix and a mutable suffix.

That keeps the problem local. The system only needs to answer:

- how far back should I cut?
- what state do I keep?
- what state do I recompute?

### 3. It localizes revision cost

Earlier text can remain intact while only the recent tail is regenerated. That is exactly the tradeoff that can buy lower latency without fully sacrificing the ability to revise.

### 4. Persistent KV can replace explicit prefix replay

The older strategy needed a fixed text prefix to be replayed in the next prompt so the model still had context from before the rollback region.

If the stable prefix KV cache can be preserved directly, that replay may no longer be necessary, or may be reduced substantially. The model would already "remember" the preserved prefix through cached autoregressive state.

## Role of the Existing Phoneme Infrastructure

A major part of the viability argument comes from the phoneme tooling already present in the project.

The setup described is:

- the Qwen3 ASR produces a high-quality text transcript
- a G2P pipeline converts that transcript into phonemes
- a separate model, ZIPA, processes the same audio segment and produces phonemes with timings

This provides two parallel phoneme views:

- text-derived phonemes from the transcript
- audio-derived timed phonemes from ZIPA

That matters because rollback boundaries do not have to be chosen in raw token-count space alone.

Instead, the system can use phonetic structure and timing to identify a meaningful boundary. In particular, if the transcript can be mapped to phoneme spans and those phonemes can be aligned against ZIPA's timed phoneme sequence, then the system can choose rollback points that correspond to actual spoken units rather than arbitrary text-token boundaries.

The practical payoff is that rollback can happen on word or phonetic boundaries with confidence that the conditioning still matches the cut. That removes one of the nastier problems that would otherwise appear in a suffix-regeneration design.

In the conversation, this was treated as essentially solved enough for current purposes.

## What This Design Does Not Need to Optimize For

Several possible concerns were explicitly deprioritized.

### UI stability is not the bottleneck

Visible rewrites are acceptable. The product is allowed to show evolving text, and there is already an animation strategy for that. So the system does not need to aggressively suppress changes purely for presentation reasons.

### There is no requirement for a separate high-accuracy final pass

This is important. The design goal is not "progressive low-latency preview plus a slower final correction pass." The goal is the progressive decode itself. Latency matters more than maximal final accuracy.

That removes a whole class of design pressure around end-of-utterance redecoding.

### The hard problem is not generic text stitching

Because the intended direction is rollback inside one evolving transcript rather than arbitrary overlap-merging between window outputs, the problem should not be framed primarily as a text stitching problem.

That reframes the architecture discussion toward state invalidation and state reuse rather than post-hoc merging.

## What Seems Largely Resolved in This Discussion

Within the scope of this conversation, several things appear to have a working conceptual answer.

### Rollback boundary choice

This is not being treated as "drop the last N tokens because N is convenient." The presence of transcript text, G2P phonemes, and ZIPA timed phonemes means rollback can be performed at a meaningful boundary, potentially at the word level.

That sharply reduces the fear that rollback will cut the transcript at an incoherent point.

### Conditioning consistency across the cut

The stated assumption is that the alignment is accurate enough that cutting and resuming from that point is not a speculative gamble. The conversation treated this as sufficiently trustworthy for the current design direction.

### Incorrect committed text is handled by leaving a buffer

The answer to "what if the committed prefix is still wrong?" is not "never commit." The answer is to keep a revisable buffer behind the live edge and avoid committing too aggressively.

In other words, the design already includes a mechanism for protecting against premature commitment.

### Explicit prefix replay is probably no longer fundamental

The old strategy of replaying a small fixed text prefix may become unnecessary if KV for the stable prefix can be preserved directly.

That does not mean it can never help as a fine tuning measure. It does mean it is no longer the central mechanism.

## What Still Appears Open

At this stage, the main open question is not whether rollback is a sensible idea. It is what state can actually be reused safely and efficiently.

### 1. Text-side KV reuse

If the model has already consumed a stable text prefix, can that autoregressive self-attention cache simply be retained while rolling back only the mutable suffix?

This is the most obvious and most promising reuse target. If it works cleanly, it avoids replaying older text context.

### 2. Audio-side state reuse

The more subtle question is what part of the audio conditioning can survive the rollback.

The architecture described has an acoustic front-end and an autoregressive decoder/LM. That raises several possibilities:

- reuse low-level acoustic features
- reuse encoder outputs for already-seen audio
- rebuild only the audio states corresponding to the rollback tail and newly arrived audio
- or recompute audio-side state more aggressively than text-side state if the model's coupling makes partial reuse unsafe

This is the core technical unknown that the next discussion is supposed to investigate.

### 3. Relationship between rollback and cross-attention conditioning

Even if text-side KV can be truncated cleanly, the decoder's conditioning on audio may depend on a set of audio-side keys and values that do not have the same truncation semantics.

The question is not just whether the system has a cache. The question is whether the cache is structurally separable into:

- preserved prefix text state
- discarded suffix text state
- preserved audio state
- refreshed audio state

That boundary may or may not line up cleanly with the rollback policy.

### 4. Positional semantics after truncation

Any reuse design has to be compatible with however positions are represented internally. If preserved states are tied to specific token positions or specific audio-frame positions, the truncation strategy has to respect those assumptions.

This is not yet analyzed in the discussion, but it remains an obvious technical point to validate when looking at actual model usage.

### 5. Whether text replay remains useful as a fallback

The current view is that explicit replay of a small fixed prefix is probably not fundamental once KV reuse exists. That still leaves an implementation question:

- is direct KV preservation robust enough on its own?
- or does replaying a very short text anchor near the rollback boundary still improve continuity in practice?

This feels like a tuning question rather than a first-principles blocker, but it is still open.

## Overall Assessment So Far

From the discussion alone, the direction looks technically interesting and internally coherent.

The strongest case in its favor is that it lines up with the actual product goal:

- low latency
- near-real-time updates
- visible rewrites are acceptable
- no separate full finalization pass required

The second strongest point in its favor is that the project already has unusually relevant structure for choosing rollback boundaries:

- high-quality transcript text
- G2P conversion to phonemes
- independent audio-derived phonemes with timings from ZIPA

That means rollback can be grounded in phonetic and temporal structure instead of raw token-count heuristics.

The main thing still unanswered is not whether rollback is a valid product strategy. It is how far persistent cache reuse can be pushed in the actual model pipeline without violating the model's conditioning semantics.

Put differently:

- the transcript-side policy is mostly taking shape
- the open systems question is cache topology

## Next Question

The next step should be to reason carefully about the exact kinds of state involved during decoding and which of them are candidates for reuse under a rollback regime.

The key split to analyze is:

- text-side autoregressive KV
- audio-side encoded or attended state
- how those interact at the rollback boundary

That is the technical center of gravity for the next phase of the discussion.
