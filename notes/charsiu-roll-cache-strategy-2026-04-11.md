# Charsiu Cache Strategy For `stable/carry/preview`

Date: 2026-04-11

## Purpose

This note proposes how Charsiu should fit into Bee's rollback-oriented
`stable/carry/preview` model without blowing the `~200 ms` update budget.

The key distinction is:

- plain Charsiu G2P is cheap enough to be part of the hot path
- teacher-forced cross-attention ownership is a second, richer pass and should
  not run on every live update

This follows the intended split already described in
[`rust/bee-g2p-charsiu/README.md`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/README.md:20).

## Background

`bee-roll` models the live transcript as three adjacent regions:

- `stable`
- `carry`
- `preview`

See [`rust/bee-roll/README.md`](/Users/amos/bearcove/bee/rust/bee-roll/README.md:21).

Operationally:

- `stable` is kept
- `carry` is replayed
- `preview` is reopened and rewritten

Charsiu must respect that shape. Re-running expensive alignment-grade work over
the entire visible transcript every `200 ms` is the wrong fit for this model.

## Observations

### Plain G2P and forced attention are different jobs

The current `bee-g2p-charsiu` crate already treats them as separate things:

- persistent sidecar / IPA generation
- cross-attention probe
- ownership collapse into token-piece spans
- comparison-token and aligner-ready views

See:

- [`rust/bee-g2p-charsiu/README.md`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/README.md:20)
- [`rust/bee-g2p-charsiu/scripts/charsiu_cross_attention_probe.py`](/Users/amos/bearcove/bee/rust/bee-g2p-charsiu/scripts/charsiu_cross_attention_probe.py:98)

The probe path is inherently heavier because it runs a full teacher-forced model
forward with cross-attention output enabled:

- `output_attentions=True`
- `use_cache=False`

That makes it the wrong thing to put on the tight live-update loop.

### Plain Charsiu batching is already in range

Recent local numbers show that batched Charsiu G2P is in the right latency
range for the product path.

That means the budget pressure is mostly about:

- avoiding redundant recomputation
- keeping the hot path limited to the cheap pass
- reserving the richer pass for cut/alignment moments

## Proposed split

### Hot path: every `feed()` update

Run only:

- word segmentation for the current mutable right edge
- batched plain Charsiu G2P
- normalization into comparison phones if needed

Do not run:

- teacher-forced cross-attention ownership
- token-piece projection
- aligner-ready phrase probe over the whole visible transcript

Hot-path scope should usually be:

- changed words in `preview`
- changed words in `carry`

Never recompute plain G2P for `stable`.

### Alignment path: only when needed

Run the richer cross-attention ownership pass only when:

- a cut candidate is accepted
- a word or token-piece timing projection is actually needed
- a seam-local region changed enough that token-piece ownership must be refreshed

Alignment-path scope should usually be:

- `carry + preview`
- or an even smaller seam-focused slice

Never rerun forced attention for `stable`.

## Cache layers

There should be two independent caches.

### 1. Plain IPA cache

Purpose:

- serve the hot path cheaply
- avoid repeating `word -> ipa`

Suggested key:

- `(lang_code, normalized_word_text)`

Suggested value:

- raw IPA string
- normalized phone sequence
- optional metadata like model id / normalization version

Normalization for the key should be conservative and explicit. For example:

- Unicode NFC
- preserve case unless lowercasing is proven safe for the language/model setup
- preserve apostrophes / hyphens if they affect pronunciation

This cache should be long-lived and process-wide.

### 2. Ownership / forced-attention cache

Purpose:

- serve the alignment path
- avoid repeating teacher-forced cross-attention analysis

Suggested key:

- `(lang_code, probe_text, qwen_piece_boundaries_version, charsiu_probe_version)`

Where:

- `probe_text` is the exact text slice fed to the probe
- `qwen_piece_boundaries_version` guards the mapping logic from text to Qwen token pieces
- `charsiu_probe_version` guards ownership collapse logic and normalization rules

Suggested value:

- decoded IPA
- word spans
- Qwen token-piece spans
- per-output-step ownership summaries
- collapsed token-piece IPA spans
- comparison-token sequence
- aligner-ready adapter view if already derived

This cache should be smaller and more local than the plain IPA cache, because
the key space is phrase/slice-based rather than single-word-based.

## Invalidation policy by tape region

### `stable`

`stable` is frozen.

Policy:

- never rerun plain G2P
- never rerun forced attention
- keep previously derived comparison-token and provenance data

If a later design discovers a bug in normalization or ownership collapse,
invalidate by version stamp, not by live rollback logic.

### `carry`

`carry` is replayed and may survive across multiple updates unchanged.

Policy:

- if the exact carried text is unchanged, reuse both caches
- if `carry` text changes, invalidate only the changed suffix/range
- do not invalidate `stable`

The important point is that unchanged carried text must not trigger repeated
teacher-forced probe work.

### `preview`

`preview` is always mutable.

Policy:

- plain G2P may run every update, but only for changed words
- forced attention should usually wait until commit/cut/alignment time

If some live UI path wants ownership before commit, it should request it only
for the narrow seam region and only on demand.

## Change detection

The key implementation detail is to detect deltas instead of treating the whole
visible transcript as new.

Suggested approach:

1. token-address the tape as usual
2. derive the current word segmentation for `carry + preview`
3. diff against the previous `carry + preview` word list
4. identify:
   - unchanged words
   - inserted words
   - deleted words
   - rewritten span
5. only enqueue the changed words/span for Charsiu work

For plain IPA this can happen at word granularity.

For forced attention it is better to operate on a contiguous changed text span
with a small amount of surrounding context when needed.

## Suggested operating policy

### Plain G2P microbatch policy

For the hot path:

- keep one hot sidecar/model instance
- collect changed-word requests up to a small deadline such as `150-180 ms`
- flush earlier if queue size reaches something like `8-16`
- group by language
- optionally bucket by rough length to reduce padding waste

This is the path that should support the `~12 words every 200 ms` product goal.

### Forced-attention policy

For the alignment path:

- trigger on commit/cut, not every feed
- probe only the seam-local text slice
- cache the result by exact probe text
- reuse until that slice changes

This treats ownership as alignment-grade data, not UI-preview data.

## Recommended seam-local probe scope

The default probe scope should be:

- all words in `carry`
- all words in `preview`

with optional trimming if needed:

- if `carry + preview` is large, probe only the words near the seam plus enough
  local context to preserve pronunciation quality

The exact amount of context should be chosen empirically, but the principle is:

- phrase context is useful
- full visible transcript context is usually unnecessary

## Data products to freeze at commit

When text moves from mutable to `stable`, freeze all of:

- plain IPA
- normalized comparison phones
- ownership-derived token-piece IPA spans
- transcript comparison-token provenance
- aligner-ready comparison-token ranges

After commit, this data should be treated as immutable synchronized state owned
by the utterance/tape, not as something to be rederived opportunistically later.

## What not to do

Do not:

- rerun teacher-forced attention on the whole visible transcript every `200 ms`
- rerun plain G2P for `stable`
- invalidate the whole Charsiu state when only the right edge changed
- optimize for giant-batch throughput when the product constraint is small,
  regular microbatches

## Concrete recommendation

Use Charsiu in two tiers.

Tier 1: hot path

- changed words only
- batched plain G2P
- aggressive word-level cache

Tier 2: alignment path

- seam-local `carry + preview` phrase probe
- teacher-forced cross-attention ownership
- cache by exact probe text and versioned collapse rules

That preserves the token-level alignment strategy described in
`bee-g2p-charsiu` while keeping the `feed()` loop compatible with Bee's live
latency budget.
