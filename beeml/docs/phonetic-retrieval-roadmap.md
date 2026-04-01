# Phonetic Retrieval Roadmap

## Goal

Replace the current brute-force span proposal path with an indexed phonetic retrieval pipeline that:

- starts from a strong ASR transcript (`Qwen/MLX`)
- uses forced-aligner timings for locality
- uses `eSpeak` IPA as the main searchable representation
- retrieves plausible correction candidates quickly enough for interactive use
- defers context-dependent choice to a later reranker stage

This roadmap is specifically for the non-ZIPA path.

## Current Takeaway

What appears to be true now:

- `Qwen/MLX` transcript quality is already strong enough to be the main text source.
- forced aligner timings are already available and good enough for region locality.
- `eSpeak` IPA has produced the best candidate pairs so far.
- the real unsolved problem is efficient candidate retrieval over phonetic forms.
- the current expensive step is effectively `many transcript spans x many lexicon entries`.

## Problem Statement

Given:

- a transcript with word timings
- optional confidence per token/word later
- `eSpeak` IPA for transcript words/spans
- a lexicon containing canonical terms, spoken variants, and confusion pairs

we need:

- a fast way to retrieve plausible correction candidates for contiguous transcript spans
- without brute-forcing every span against every lexicon entry
- while preserving enough provenance to debug retrieval failures and tune ranking

## Non-Goals

For this roadmap, do not optimize for:

- audio-based retrieval
- ZIPA-first alignment
- end-to-end neural spoken term detection
- whole-sentence correction as the primary retrieval mechanism

ZIPA can be reevaluated later as an auxiliary signal. It is not the center of this design.

## Target Architecture

The pipeline should become:

1. `Qwen/MLX` transcript
2. forced-aligner word timings
3. `eSpeak` IPA projection for words and spans
4. phonetic retrieval index
5. shortlist verification with feature-aware phonetic distance
6. context-dependent reranking over a small number of local candidates

The key separation is:

- **retrieval** decides which spans and term candidates are plausible
- **reranking** decides which candidate fits the sentence context

## Phase 1: Lexicon Expansion

Build a normalized lexicon representation for retrieval.

Each lexicon entry should include:

- `term`
- `alias_text`
- `alias_source`
  - `canonical`
  - `spoken`
  - `confusion`
- `ipa`
- `ipa_reduced`
- `token_count`
- `phone_count`
- `identifier_flags`
  - acronym-like
  - contains digits
  - snake/camel/symbol-derived

Do not add G2P N-best yet.

Reason:

- human-entered spoken variants and confusion pairs are already high-signal
- G2P expansions are likely to increase candidate noise early

## Phase 2: Primary Retrieval Index

Build one main index first:

- boundary-aware IPA 2-gram postings
- boundary-aware IPA 3-gram postings
- length bucket
- token-count bucket

This should be implemented as an inverted index, not brute-force scan.

Suggested stored retrieval features per alias:

- raw IPA q-grams
- reduced IPA q-grams
- start/end boundary grams
- token-count bucket
- phone-length bucket

The index should return a shortlist with provenance, not just term ids.

Each retrieved hit should retain:

- `term`
- `alias_source`
- `matched_alias`
- `which_index_view_matched`
- `qgram_overlap_count`
- `length_bucket_match`
- `token_count_match`

## Phase 3: Span Enumeration

We still need a span proposal strategy, but it should be cheap and explicit.

Initial version:

- enumerate contiguous spans up to 4 or 5 words
- derive span IPA from `eSpeak`
- query the phonetic index for each span

Keep span metadata:

- token range
- char range
- time range from forced aligner
- original text
- IPA
- word count

This is still potentially expensive, so it must be paired with retrieval filters:

- length bucket match
- token-count compatibility
- q-gram overlap threshold

## Phase 4: Verification

Only verify the top shortlist per span.

Recommended verifier:

- feature-aware phonetic distance
- use `rspanphon` as the current base

Verifier output should include:

- normalized phonetic similarity
- raw feature distance
- candidate/source metadata
- exact/compact/prefix indicators if useful

This verifier should be authoritative for shortlist refinement, but not used as the full search algorithm.

## Phase 5: Contextual Reranking

Reranking should be local and comparative.

It should not score random full-sentence mutations independently.

For each proposed region:

- original sentence
- left context
- right context
- original span
- candidate replacements
- keep-original option

Ask the reranker to choose among local alternatives for that region.

This stage should consume a small, already filtered set of candidates.

## Phase 6: Short-Query Lane

Short phonetic strings behave differently and will likely need special handling.

Do this only after the primary index works.

Options:

- stricter thresholds for very short IPA queries
- explicit acronym/identifier rules
- later: trie/TST + Levenshtein automaton or deletion index for `k=1`-style short queries

Do not build the full short-query sidecar before the main q-gram path is stable.

## Debuggability Requirements

Every retrieval result must preserve provenance.

This is mandatory.

For each shortlisted candidate, we need to know:

- which span generated it
- which alias source produced it
- which index view retrieved it
- why it survived filtering
- verifier score
- whether the reranker accepted it

Without this, tuning will be guesswork.

## Recommended Module Boundaries

Add new backend modules roughly along these lines:

- `phonetic_lexicon.rs`
  - lexicon expansion
  - alias normalization
  - IPA storage

- `phonetic_index.rs`
  - q-gram postings
  - retrieval query path
  - shortlist generation

- `phonetic_verify.rs`
  - feature-aware verification
  - score explanation/debug output

- `region_proposal.rs`
  - span enumeration
  - early filters
  - non-overlapping proposal selection later

Existing reranker logic can initially stay where it is, but it should consume the new retrieval outputs.

## Evaluation Plan

We need two evaluation loops.

### 1. Retrieval Benchmark

Measure retrieval independent of reranking.

For a fixed set of human examples:

- was the target term retrieved at all?
- was it in top 1 / top 3 / top 10?
- how many spans were queried?
- how many candidates were verified?
- retrieval latency

### 2. End-to-End Correction Eval

Measure:

- exact sentence recovery
- target term recovery
- target proposed
- target accepted
- latency breakdown

Failure buckets should remain explicit:

- no proposal
- target proposed but not selected
- wrong proposal selected
- target-only partial fix

## Implementation Order

Recommended execution order:

1. lexicon normalization for phonetic retrieval
2. q-gram inverted index over IPA
3. span query -> shortlist path
4. feature-aware verifier on shortlist
5. retrieval benchmark
6. reranker integration on top of new shortlist
7. short-query special handling
8. optional later work:
   - feature q-gram view
   - G2P expansions
   - automaton/trie short-query sidecar
   - WFST-based alias/verbalization path

## Success Criteria

The first prototype is successful if:

- it eliminates brute-force span-vs-lexicon search
- target retrieval recall is materially better than current brute-force heuristics
- it is fast enough for interactive correction
- it is diagnosable when it fails

The first prototype does **not** need:

- perfect context handling
- G2P expansions
- every possible phonetic view
- ZIPA integration

It needs to produce a trustworthy, inspectable shortlist.

## Open Questions

- how aggressively should spans be enumerated before retrieval cost dominates?
- should reduced IPA be in the first index or added only after baseline results?
- what is the best verifier threshold policy for acronym-like terms?
- can `Qwen/MLX` token confidence be surfaced soon enough to guide span proposal early?

## Immediate Next Step

Implement the lexicon record type and the first q-gram inverted index over boundary-aware IPA.
