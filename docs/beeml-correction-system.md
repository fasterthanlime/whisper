# beeml Correction Plan

## Goal

Ship a dictation correction layer on top of a strong fixed ASR model.

The app starts with:
- the base ASR model
- zero custom vocabulary

Everything else is learned from user corrections.

## Product Shape

When a user notices a mistake, they select the bad span and provide the correct text.
That correction should immediately:
- create or update a local vocabulary entry
- add spoken / phonetic aliases if we can derive them
- update the local phonetic retrieval index
- store the correction event for future training

The base ASR model does not change.

## Runtime System

Runtime should be simple:
1. ASR produces the transcript
2. local correction system proposes spans
3. phonetic retrieval searches only the user's local vocabulary
4. verifier scores phonetic plausibility
5. judge decides:
   - keep original
   - or replace with one candidate

## What The Judge Is

The judge is not a text-generation model.
It is a small classifier over candidate features.

First learned judge:
- logistic regression

Input:
- one candidate feature row
- plus a synthetic `keep_original` candidate

Output:
- score for whether this is the right action for the span

At inference:
- score all candidates plus `keep_original`
- choose the best one only if it beats `keep_original` by enough margin

## Candidate Features

The judge uses features we already compute:
- q-gram retrieval scores
- token-distance score
- feature-distance score
- boundary penalties
- guards and acceptance-floor booleans
- alias source
- identifier flags
- token-count / phone-count compatibility

Later we can add:
- left/right context features
- user priors
- session recency priors

Streaming-specific feature to explore:
- keep some record of early streaming ASR hypotheses, not just the final stabilized text
- sometimes the model hears the term correctly first and then "fixes" it into a worse context-driven word later
- the judge may need features like first-hit token, revision count, or earliest stable hypothesis for the span

## What "Online Learning" Means Here

For v1, online learning means:
- user adds a correction
- local vocabulary updates immediately
- retrieval updates immediately
- local priors / memory update immediately

That is the core learning loop.

It does not require updating model weights on every correction.

## Model Training

First model-training step:
1. export candidate feature rows from eval
2. label them as:
   - correct replacement
   - or `keep_original`
3. train logistic regression offline
4. save weight vector
5. run that weight vector in Rust at inference time

This is enough to prove whether a tiny learned judge beats the current deterministic scorer.

## If We Want True Online Weight Updates Later

Then we need:
- a stable numeric feature vector
- a local event log of user corrections
- a label policy for `keep_original` vs replacement
- an incremental optimizer
- persisted user-local weights
- rollback if updates go bad

`linfa-logistic` is good for the first offline baseline.
It is not the full online-update story by itself.

## Immediate Next Steps

1. export feature rows from eval
2. build training-prep for span/candidate labels
3. train logistic-regression baseline
4. compare it to the deterministic scorer
5. if it wins, wire it into the app

## Explicit Non-Goals Right Now

Not now:
- retraining the ASR model
- generic bundle/versioning machinery
- large reranker models as the default path
- complicated rollout policy docs

The only thing we need to prove next is:
can a tiny judge over our existing candidate features beat the hand-tuned scorer?
