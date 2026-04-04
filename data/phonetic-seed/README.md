# Phonetic Seed Dataset

This is the canonical file-first seed dataset for the current phonetic
retrieval rebuild.

It is derived from the recovered SQLite snapshot, but it intentionally avoids
reproducing the old database schema.

## Included Files

- `vocab.jsonl`
  - one row per canonical term
  - fields:
    - `term`
    - `spoken`
    - `ipa`
    - `description`
- `sentence_examples.jsonl`
  - human-authored sentence examples
  - fields:
    - `term`
    - `text`
    - `kind`
    - `surface_form`
- `recording_examples.jsonl`
  - authored recordings paired with transcript text
  - fields:
    - `term`
    - `text`
    - `take`
    - `audio_path`
    - `transcript`
- `audio/`
  - the `.ogg` files referenced by `recording_examples.jsonl`

## Excluded From Canonical Seed

The first production lexicon build should not depend on the recovered confusion
surface tables.

Those rows still exist in `data/recovered-seed/` for inspection, but they are
not part of this canonical dataset by default.

## Intended Use

- build the reviewed phonetic lexicon from `vocab.jsonl`
- use `sentence_examples.jsonl` for sentence corpus seeding
- use `recording_examples.jsonl` plus `audio/` for eval and reranker training

This keeps the product/runtime path file-first and explicit.


Auxiliary files not loaded by default:
- `counterexample_sentences.jsonl`: sentence rows intentionally authored as non-target confusions / distractors
- `counterexample_recordings.jsonl`: recording rows corresponding to those counterexamples

The canonical retrieval/eval path should use `sentence_examples.jsonl` and `recording_examples.jsonl` only.
