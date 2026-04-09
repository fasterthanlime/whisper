# ZIPA Candidate Scoring Notes

## Current State

The main alignment problem has changed.

Before:
- the utterance-level phone alignment was assigning bad ownership near word boundaries
- word cards were inheriting the wrong phones from neighboring words

Now:
- the main word-card path uses a segmental, non-overlapping word-slice selection
- ownership is much better
- the next frontier is not "which word owns these phones?"
- the next frontier is "given the chosen slice, does ZIPA provide useful evidence among plausible candidates?"

That means alignment is no longer the main blocker. ZIPA scoring inside the chosen slice is.

## Architecture Shift

Old path:
1. utterance-level phone DP
2. project word ranges from that path
3. repair ownership downstream

New path:
1. generate candidate ZIPA slices per word
2. score each word against each slice locally
3. run a global monotonic DP over word candidates
4. choose one non-overlapping slice per word

This segmental path is now the right abstraction. We should not go back to utterance-DP-derived word lanes as the main path.

## What Improved

### `zipa-targeted-v1-023`
Prompt:
- `I used serde in the parser.`

Observed ASR:
- `I used Thursday in the parser.`

Current backend word lanes:
- `I` -> `a ɪ`
- `used` -> `j ʊ z d`
- `Thursday` -> `s ə t ɪ`
- `in` -> `ɪ n`
- `the` -> `ð ə`
- `parser` -> `p ɑ z ə`

This is a real win for the new path. The `j` onset now belongs to `used`, which was exactly the kind of boundary-ownership failure we were trying to eliminate.

### `zipa-targeted-v1-041`
Prompt:
- `Actually Miri caught it.`

Observed ASR:
- `Actually, Mary caught it.`

Before the segmental scoring improvements:
- `Actually` -> `ɛ k t ʃ ə w ə`
- `Mary` -> `l ɪ m ɪ ɪ`

After the segmental scoring improvements:
- `Actually` -> `ɛ k t ʃ ə w ə l ɪ`
- `Mary` -> `m ɪ ɪ`

This shows that ownership is much better:
- `Actually` now keeps its tail
- the second word now starts with `m`

The remaining weakness is not ownership. It is that ZIPA still does not separate the internal phone content strongly enough to make `Miri` obvious.

## What We Learned From Manual Transcribe Contrasts

Several contrast recordings were inspected in the browser with IPA visible.

Observed patterns:

`Mary.`
- transcript: `m ɛ ɹ ɪ`
- ZIPA variants seen:
  - `m ɪ l ɪ`
  - `m ɛ ∅ ɪ`

`Mary caught it.`
- transcript: `m ɛ ɹ ɪ`
- ZIPA:
  - `m ɛ ∅ ɪ`

`Merry Cottage.`
- `Merry`
- transcript: `m ɛ ɪ`
- ZIPA:
  - `m ɪ ɪ`

`Actually, marry.`
- transcript around the second word:
  - `m a ɪ`
- ZIPA around the second word:
  - `... l ɪ m ɛ ɪ`

`Actually, marrying.`
- transcript around the second word:
  - `m a ɪ ɪ ŋ`
- ZIPA around the second word:
  - `... l ɪ m ɪ ɪ`

Conclusion:
- ZIPA is not robustly producing a distinct `Miri` pattern
- but it is also not simply outputting the same exact `Mary` sequence every time
- `Mary / Merry / marry / Miri` are collapsing into a broad `m-?-ɪ` basin
- longer contexts still show some boundary residue before the `m`

## Core Product Question

The right question is no longer:
- can ZIPA discover the word from scratch?

The right question is:
- given the chosen word slice, does ZIPA prefer candidate A or candidate B?

In production, ZIPA should be a verifier / reranker over a bounded candidate set, not a free search over the universe.

## Candidate Scoring Plan

### Diagnostics / eval

In eval we *do* know the gold target.

That means it is valid to ask questions like:
- what score does ZIPA assign to `m ɪ ɹ ɪ`-like sequences?
- what score does it assign to `m ɛ ɹ ɪ`-like sequences?

This helps answer:
- was the correct form absent from the ZIPA lattice?
- or was it present but lost by greedy decoding?

### Production

In production we do *not* know the answer in advance.

So ZIPA should score only:
- the transcript surface pronunciation
- the retrieved candidate pronunciations for that span
- maybe a few near-verified alternatives

This keeps the problem bounded and aligned with the rest of the correction pipeline.

The production question becomes:
- if retrieval already thinks `miri` and `mary` are both plausible, does the ZIPA evidence prefer one over the other?

## What Data ZIPA Already Exposes

`bee-zipa-mlx` inference already returns frame-level log probabilities:
- `log_probs`
- `log_probs_len`
- greedy `token_ids`
- greedy `tokens`

Current decoding is only:
- per-frame argmax
- then greedy CTC collapse

That means there are at least three useful next layers:

1. per-frame top-k token alternatives
2. slice-local CTC beam / n-best decode
3. constrained candidate scoring for a small set of phone strings

## Recommended Order

1. Keep the segmental word-slice path as the main path.
2. Add cheap per-frame top-k ZIPA diagnostics.
3. Add slice-local CTC beam or constrained candidate scoring.
4. Use fixed prompt-id harness cases throughout.

## Fixed Harness Cases

Keep iterating on:
- `zipa-targeted-v1-023`
- `zipa-targeted-v1-041`

And then broaden to a few more borderline terms:
- `serde`
- `Mach-O`
- `ripgrep`
- `QEMU`

These together should tell us whether the next layer is:
- a general ZIPA decoding problem
- or a one-off lexical weirdness around the `Mary / Merry / marry / Miri` cluster

## Current Conclusion

The segmental recommendation has been validated.

The remaining work is no longer mainly alignment ownership. It is:
- whether the correct phone sequence has posterior mass inside the chosen slice
- and whether ZIPA can usefully rescore the candidate set that correction/retrieval already provides
