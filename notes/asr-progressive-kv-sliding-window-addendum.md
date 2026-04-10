# Sliding-Window KV Rollback Addendum

This note is an addendum to `notes/asr-progressive-kv-cache-report.md`.

It does not replace that note. Its purpose is to state the actual experiment in a much more concrete and unambiguous way.

## The Experiment

The experiment is not full-prefix rerun.

It is not:

- decode `0..2`
- decode `0..4`
- decode `0..6`

It is also not pure append-only streaming.

It is a fixed-cost sliding-window decode with rollback and KV retention.

The pattern is:

1. decode a fixed-size audio window
2. roll back a fixed-size overlap in timeline space
3. truncate text state and KV state to that rollback point
4. preserve everything before that cut
5. decode the next fixed-size window starting at the rollback point

Example:

- decode `0..2`
- roll back to `1`
- decode `1..3`
- roll back to `2`
- decode `2..4`

The specific durations are tunable. The important property is the shape:

- fixed decode window size
- fixed rollback overlap
- bounded work per step
- preserved prefix state through retained KV
- revisable suffix through rollback

## Why Roll Back At All

Rollback exists because append-only continuation locks in mistakes near the live edge.

Later audio can disambiguate the recent past. If decoding only appends, then the model is forced to continue from a state that may already contain a bad local decision. That is cheap, but it weakens revision power exactly where streaming ASR needs it most.

Rollback reopens a recent overlap region so that the model can revise unstable text while keeping older, more stable state intact.

So the design is:

- keep the stable prefix cheap by retaining KV
- keep the recent tail revisable by rolling it back

## Why Keep KV

Keeping KV is not a side optimization. It is the mechanism that makes the experiment worthwhile.

After the rollback cut, the prefix before that cut should remain alive in model state. That means the next step does not need to replay or recompute the preserved prefix from scratch.

The intended effect is:

- rollback removes only the unstable suffix
- KV truncation preserves the kept prefix
- the next decode continues from the preserved prefix state rather than rebuilding the whole history

Without retained KV, the design degenerates toward replaying prompt history over and over.

## The Role of Timing

The rollback cut is defined in timeline space first, not in raw token-count space.

That means:

- choose the rollback point by time
- map that time to the corresponding prefix/suffix split in transcript state
- truncate KV at the position corresponding to that kept prefix

The critical idea is that rollback is governed by audio time, because the overlap is meant to provide acoustic context around the seam.

This is why the availability of word timings matters. It gives a way to relate:

- audio time
- transcript span
- KV truncation point

The important point is not "correct word boundaries" as a product requirement. The important point is that the rollback cut should be derived from timing information rather than from an arbitrary "drop the last N tokens" heuristic.

## Prompting Requirement

The prompt scheme has to cooperate with this design.

Each step should semantically mean:

- preserve the transcript state before the rollback cut
- discard the reopened suffix
- provide the next fixed-size audio window
- continue the same evolving transcript

So the prompting question is not merely "can the model accept more audio?"

The real question is:

- does the prompt framing support continuation of one evolving transcript after truncation
- or does it bias the model toward treating each window as a separate chunk-local answer

That question matters because bad prompt semantics could fight rollback even if the cache mechanics are correct.

## What This Experiment Is Trying to Buy

The target tradeoff is:

- bounded compute per update
- low latency
- visible real-time text
- enough overlap to repair seam-local mistakes
- no growing-prefix reruns

In one sentence:

keep the old prefix alive, reopen only the recent overlap, and advance through audio at constant cost.

## What The First Bridge Run Showed

The first useful bridge experiment was a three-way split inside each fixed-size window:

- earliest third: kept in KV
- middle third: replayed as prompt text in the next window
- latest third: reopened and re-decoded

For a `3s` window that means:

- `0..1` committed
- `1..2` replayed as prompt text
- `2..3` reopened
- next window is still a full `3s` window, shifted by `1s`

So the sequence is:

- decode `0..3`
- carry `1..2`
- decode `1..4`
- carry `2..3`
- decode `2..5`

The crucial finding was that the replayed text cannot be computed from the newly generated suffix alone.

That is wrong because once a chunk has non-empty kept generated text, the "middle third" of the effective row can span both:

- generated words that will be kept
- generated words that would otherwise be called "bridge"

So the carry for the next row must be derived from the effective row transcript:

- `replayed_prefix + newly_generated_text`

and then aligned against that row's audio as one unit.

If the split is computed too late, on the generated suffix only, the middle-third carry becomes too small and shifts left relative to what the model is actually doing.

## Visualization Lessons

The row-by-row view is still useful, but only if it respects the same transcript semantics as the experiment.

The important rendering rules are:

- carried prompt text must be shown separately from newly generated words
- the row alignment must be computed on the effective transcript, not the generated suffix alone
- a stitched transcript view should use the timings already assigned when segments survived, not re-align them a second time

That stitched view turned out to be important because a run can look odd row-by-row while still producing a correct composed transcript.

## Practical Conclusion

The bridge experiment is more promising than the early broken visualizations suggested.

The key requirements are now clearer:

- fixed-size windows
- fixed-size three-way split
- replay prefix derived from the effective row transcript
- full-width final window pinned to EOF rather than a short tail window
- a stitched committed timeline that shows what transcript the strategy actually produces
