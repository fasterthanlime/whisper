# beeml Correction System

## Purpose

This document is the system-level description for the correction product that
`beeml` and `beeml-web` are supposed to become.

It sits above:

- [phonetic-retrieval-roadmap.md](/Users/amos/bearcove/bee/docs/phonetic-retrieval-roadmap.md)
- [phonetic-retrieval-implementation-checklist.md](/Users/amos/bearcove/bee/docs/phonetic-retrieval-implementation-checklist.md)
- [eval-frontend-handoff.md](/Users/amos/bearcove/bee/docs/eval-frontend-handoff.md)

Those documents cover pieces of the system. This one defines the whole shape:

- what is being trained
- what runs in production
- what assets are canonical
- what `beeml` should expose
- what `beeml-web` must make debuggable

## Current Reality

Today:

- `beeml` is still a minimal transcription RPC server
- `beeml-web` is still a minimal transcription demo with some richer inspector
  scaffolding
- `bee-phonetic` now has the first file-first seed dataset and a first indexed
  retrieval baseline

What is already effectively solved enough for this design:

- ASR
- forced alignment

What is not solved:

- retrieving good correction candidates fast enough
- deciding which local correction to apply using sentence context
- making the whole decision path inspectable in the frontend

That means the product is not "transcription". The product is:

- transcript correction for technical vocabulary
- phonetic retrieval and verification
- contextual local reranking
- interactive inspection and evaluation

## Final Objective

The end state should be:

- `beeml` is the production and evaluation backend for correction
- `beeml-web` is the main frontend for live debugging, evaluation, and review
- the correction system is retrieval-first, not generation-first
- every accepted or rejected correction is explainable in the UI without reading
  logs

The system should accept an already-good transcript plus timings, retrieve a
small set of plausible span-local corrections, verify them phonetically, choose
among them with a contextual reranker, and return both the corrected text and
the reasoning trace.

## System Split

The architecture should be treated as two different systems sharing assets:

1. training and evaluation
2. production inference

They use the same lexicon and retrieval concepts, but they have different
responsibilities.

## Training and Evaluation

Training exists to build and validate correction artifacts.

It should answer:

- which aliases and pronunciations belong in the lexicon
- which retrieval views actually recover the target term
- whether the reranker can choose correctly when the target is present
- which artifact bundle should be promoted into production

### Training Inputs

The durable source data should be file-first and reviewable.

Primary sources today:

- [data/phonetic-seed/vocab.jsonl](/Users/amos/bearcove/bee/data/phonetic-seed/vocab.jsonl)
- [data/phonetic-seed/sentence_examples.jsonl](/Users/amos/bearcove/bee/data/phonetic-seed/sentence_examples.jsonl)
- [data/phonetic-seed/recording_examples.jsonl](/Users/amos/bearcove/bee/data/phonetic-seed/recording_examples.jsonl)
- [data/phonetic-seed/audio](/Users/amos/bearcove/bee/data/phonetic-seed/audio)

Confusion surfaces should not be part of the default canonical seed until they
prove useful enough to justify the noise.

### Training Products

Training should produce explicit, versioned artifacts, not opaque database
state.

Expected artifact families:

- normalized lexicon snapshot
- derived alias snapshot
- phonetic retrieval indexes
- retrieval evaluation fixtures and metrics
- reranker training examples
- reranker weights or adapters
- bundle metadata and thresholds

### Training Stages

The training/eval loop should be separated into two questions.

#### 1. Retrieval

For a transcript span, does the target term enter the shortlist at all?

This is where we measure:

- top-1 / top-3 / top-10 target recall
- miss buckets by term, span shape, token count, identifier class
- which retrieval view produced the candidate
- where short queries or technical terms fail

If the target never enters the shortlist, the reranker is not at fault.

#### 2. Reranking

Given a small local candidate set that already contains the target, can the
model choose the right edit while accounting for sentence context?

This is a selection problem, not a free-form generation problem.

The reranker training examples should look like:

- left context
- original span
- right context
- keep-original option
- candidate replacements
- optional retrieval features and priors
- gold choice

That is the regime where a small model is plausible.

### Model Scope

The learned part of the system should be constrained.

Do not treat the reranker as a general sentence rewrite model.

The intended model behavior is:

- compare a small number of local alternatives
- use sentence context to choose
- preserve the original when no candidate is better

That means the central ML question is:

- can a small reranker, around the 0.5B to 0.6B range, reliably choose among
  local correction alternatives?

This is tractable if:

- the target appears in the candidate set
- negatives are hard rather than random
- "keep original" is always represented
- context windows are local and consistent

## Production Inference

Production inference should load a fixed bundle and serve correction RPCs.

It should not depend on query-heavy database logic.

### Production Inputs

The production path should assume ASR and alignment are already available.

The core request should be:

- transcript
- word timings
- optional confidence later
- optional raw audio when needed for debug or fallback features

### Production Pipeline

The correction path should be:

1. receive transcript and timings
2. enumerate plausible contiguous spans
3. derive searchable phonetic views for each span
4. retrieve candidates through indexed lanes
5. verify only the small shortlist with a stronger phonetic scorer
6. rerank local alternatives with sentence context
7. assemble the corrected sentence
8. return both result and trace

### Retrieval Stage

The intended first serious retrieval prototype is:

- boundary-aware IPA 2-gram postings
- boundary-aware IPA 3-gram postings
- articulatory-feature n-gram postings
- length and token-count filters
- a short-query fallback lane for very short phonetic strings

This is still an inverted-index architecture, but not the weak version where
only raw q-gram overlap is available.

The current low-recall baseline is useful because it confirms the failure mode:

- the target usually does not enter the candidate pool at all

That means the shortlist generator needs more structure, not just a different
final verifier.

### Lexicon and Alias Policy

The retrieval system should index aliases, not just canonical spellings.

Alias families should eventually include:

- canonical term
- human-entered spoken variants
- identifier verbalizations
- carefully accepted confusion-derived variants
- later, optional G2P N-best variants with priors

For technical vocabulary, identifier verbalization is important enough to treat
as first-class:

- camelCase
- snake_case
- digit expansions
- acronym and spelled-letter forms
- symbol verbalizations

This should be represented explicitly rather than hidden in ad hoc aliases.

### Verification Stage

The verifier should be stronger than the retriever and should only run on a
small shortlist.

Its job is:

- reject cheap retrieval noise
- score phonetic plausibility more faithfully
- preserve enough structured evidence for the reranker and UI

The verifier should move toward feature-aware or learned weighted alignment.

### Reranking Stage

The reranker should operate region-by-region.

Its choice set should contain:

- keep original
- candidate sentence with candidate edit A
- candidate sentence with candidate edit B
- candidate sentence with candidate edit C

It should not score arbitrary full-sentence rewrites independently.

The reranker output should include:

- chosen candidate index
- chosen text
- candidate-level scores or probabilities
- confidence or margin

## Canonical Assets

The system should distinguish between source-of-truth assets and derived
artifacts.

### Source-of-Truth Assets

These should stay reviewable and file-first:

- vocabulary terms
- reviewed IPA
- spoken variants
- authored sentences
- recording manifests
- audio references

### Derived Artifacts

These should be rebuilt, not hand-edited:

- normalized alias rows
- identifier verbalizations
- reduced IPA views
- articulatory feature views
- phonetic indexes
- retrieval eval fixtures
- reranker training examples
- reranker weights

## `beeml` Backend Role

`beeml` should become the RPC boundary for correction.

It should own:

- artifact loading
- retrieval
- verification
- reranking
- correction assembly
- debug trace production
- evaluation-oriented RPCs

It should not push core correction logic into the frontend.

### RPC Design Principle

Every production operation should have a debuggable representation.

That does not require separate code paths. It does require responses that can
include a trace payload when requested.

The system should prefer:

- one canonical correction pipeline
- one fast response shape
- one richer debug shape built from the same internals

### Expected RPC Families

The exact method names can change, but `beeml` should grow toward families like:

#### Production

- `correct_transcript(...)`
- `stream_correct(...)`
- `transcribe_and_correct(...)` only if still operationally useful

#### Debug and Inspection

- `debug_retrieval(...)`
- `debug_correction(...)`
- `inspect_term(...)`
- `explain_candidate(...)`

#### Evaluation

- `run_retrieval_eval(...)`
- `run_correction_eval(...)`
- `list_eval_cases(...)`
- `get_eval_case(...)`

The important boundary is not HTTP routes. It is typed RPC methods and payloads.

## `beeml-web` Frontend Role

`beeml-web` should be the primary debug and evaluation surface for the
correction system.

That means the frontend is not just a result viewer. It is the inspection tool
for the entire pipeline.

### Hard Requirement

Every correction decision should be explainable in `beeml-web`.

No stage should require reading backend logs to understand:

- why a target was missed
- why a candidate survived
- why the reranker chose or rejected an edit

### Required Debug Surface

The frontend should expose the pipeline as stages.

#### 1. Input and Span View

Show:

- transcript
- word timings
- selected span
- span text
- span IPA
- reduced IPA
- feature view if available

#### 2. Retrieval View

Show:

- which retrieval lanes fired
- top candidates per lane
- alias source
- matched alias
- q-gram overlap counts
- feature overlap if present
- token-count and length compatibility
- candidates removed by filtering

#### 3. Verification View

Show:

- verifier score
- phonetic alignment or compact comparison
- source metadata
- why the candidate stayed or fell out

#### 4. Reranking View

Show:

- sentence context
- keep-original option
- sentence alternatives side by side
- chosen candidate
- scores or probabilities
- confidence or margin

#### 5. Eval and Batch Analysis

Show:

- retrieval recall summaries
- correction success summaries
- miss buckets
- per-term failure clustering
- side-by-side comparison of bundle versions

This should support questions like:

- show all misses for `AArch64`
- show all cases where retrieval succeeded but reranking failed
- compare retrieval-only versus retrieval-plus-reranker

## Debug Data Requirements

The system must preserve provenance through all stages.

For each returned candidate, the trace should be able to answer:

- which transcript span produced it
- which alias source produced it
- which index lanes matched it
- which filters it survived
- what verification score it received
- whether it reached the reranker
- whether it was accepted

Without this, tuning becomes guesswork.

## Suggested Backend Trace Shape

The exact Rust types can evolve, but the logical structure should be:

- request metadata
- transcript and timings
- span list
- retrieval per span
- verification per span
- reranker candidates per chosen region
- accepted edits
- final corrected text
- timing breakdown

That trace should be serializable and stable enough for `beeml-web` to render
without backend-specific ad hoc transformations.

## Non-Goals

For this system definition, do not optimize for:

- whole-sentence generative rewriting as the main correction mechanism
- database-centric runtime query logic
- hiding uncertainty behind one opaque confidence number
- a frontend that only renders final text

## Implementation Order

The practical order still looks like this:

1. canonical file-first datasets
2. normalized lexicon and alias build path
3. stronger indexed retrieval:
   - IPA 2-gram and 3-gram lanes
   - articulatory feature lanes
   - identifier verbalization aliases
4. short-query fallback lane
5. stronger verification
6. constrained contextual reranker
7. debug-first `beeml` RPCs
8. `beeml-web` inspector and eval flows built directly on those RPCs

## Acceptance Criteria

This system is in the right shape when all of the following are true:

- the production correction path runs entirely from versioned artifacts
- retrieval and reranking are measurable separately
- a bad correction can be explained end-to-end in `beeml-web`
- a missed correction can be localized to:
  - alias coverage
  - retrieval
  - verification
  - reranking
- `beeml-web` is useful as both a product surface and a development debugger

That is the target to build toward.
