# Eval Frontend Handoff

This is the minimum contract for rebuilding the eval UI outside the monolithic
`index.html`, ideally as a small Vite/TypeScript app.

Scope intentionally limited:

1. `#/eval`
2. live correction inspector
3. human audio eval list + per-case inspector
4. shared playback/timeline controls

Do not try to rebuild the whole dashboard first.

## Why This Exists

The current eval UI in
[`/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/static/index.html`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/static/index.html)
has too much accumulated state and partial duplication between:

- `Live Reranker Eval`
- `Human Audio Eval`

The backend is good enough to support an eval-only rewrite now. The new app
should target the canonical fields below and treat legacy aliases as temporary.

## Backend Routes Needed

### 1. `POST /api/asr/dual`

Use for live microphone recordings before correction.

Implementation:
- [`api_asr_dual()`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs)

Input:
- raw `audio/wav` request body

Current response shape:

```json
{
  "qwen": "",
  "parakeet": "text",
  "cohere": "",
  "parakeet_alignment": [
    {"w":"word","s":0.12,"e":0.42,"c":0.98}
  ],
  "qwen_error": null,
  "parakeet_error": null,
  "cohere_error": null,
  "correction_input": "parakeet"
}
```

Important:
- current live eval is effectively Parakeet-only
- `qwen` and `cohere` are compatibility placeholders right now
- `parakeet_alignment[*].c` is word confidence

### 2. `POST /api/correct-prototype`

Use for live correction after ASR.

Implementation:
- [`api_correct_prototype()`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs)

Request body:

```json
{
  "transcript": "string",
  "audio_wav_base64": "optional base64 wav",
  "max_span_tokens": 6,
  "max_span_proposals": 12,
  "max_candidates_per_span": 4,
  "use_model_reranker": true,
  "use_prototype_adapters": true,
  "reranker_mode": "trained",
  "prototype_reranker_train_id": 262,

  "qwen": "legacy alias for transcript"
}
```

Canonical request field:
- `transcript`

Legacy request field:
- `qwen`

Response shape:

```json
{
  "original": "input transcript",
  "corrected": "corrected sentence",
  "accepted": [...],
  "proposals": [...],
  "sentence_candidates": [...],
  "alignments": {
    "timing_source": "espeak_zipa_dp | parakeet_words | forced_aligner_fallback | no_acoustic_context | no_transcript",
    "espeak": [...],
    "qwen": [...],
    "zipa": [...],
    "zipa_espeak": [...],
    "zipa_qwen": [...]
  },
  "zipa_trace": {...},
  "reranker": {...}
}
```

Important:
- `alignments.espeak` is the canonical eSpeak-aligned word lane
- `alignments.qwen` currently aliases the same data for compatibility
- `alignments.zipa_espeak` is the canonical grouped phone lane
- `alignments.zipa_qwen` currently aliases the same grouped lane for compatibility

### 3. `POST /api/correct-prototype/bakeoff`

Use for batched human-audio eval.

Implementation:
- [`api_correct_prototype_bakeoff()`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs)

Request body:

```json
{
  "source": "human",
  "limit": 5,
  "randomize": true,
  "sample_seed": 12345,
  "use_model_reranker": true,
  "use_prototype_adapters": true,
  "reranker_mode": "trained",
  "prototype_reranker_train_id": 262
}
```

For `source: "human"` the endpoint is async:
- response usually returns `{ "job_id": N }`
- client must poll `/api/jobs/{id}`

Completed result shape:

```json
{
  "source": "human",
  "limit": 5,
  "randomize": true,
  "sample_seed": 12345,
  "prototype_only_eval": true,
  "processed": 5,
  "summary": {
    "n": 5,
    "baseline": 0,
    "current": 0,
    "prototype": 2,
    "prototype_only": 2,
    "current_only": 0,
    "both_wrong": 3,
    "prototype_wrong": 3
  },
  "failure_buckets": {...},
  "target_summary": {...},
  "entries": [...]
}
```

Each `entries[]` item currently includes:

```json
{
  "term": "ripgrep",
  "case_id": "hum-123",
  "source": "human",
  "expected": "expected clean sentence",
  "qwen": "legacy transcript field",
  "recording_id": 123,
  "template_sentence": "...",
  "expected_fragment": "ripgrep",
  "expected_fragment_preview": "...",
  "expected_fragment_phonemes": "...",
  "qwen_fragment": "...",
  "qwen_fragment_phonemes": "...",
  "hit_count": 1,
  "baseline_ok": false,
  "baseline_target_ok": false,
  "current": "legacy current field",
  "prototype": "reranker output",
  "prototype_ok": false,
  "prototype_target_ok": true,
  "analysis": {
    "failure_reason": "no_proposal | proposal_found_no_sentence_edit | reranker_missed_target_candidate | target_only | exact_ok",
    "target_proposed": true,
    "target_sentence_candidate": false,
    "target_accepted_edit": false,
    "exact_ok": false,
    "target_ok": true
  },
  "prototype_accepted": [...],
  "prototype_trace_excerpt": {
    "proposal_count": 7,
    "sentence_candidate_count": 2,
    "accepted_count": 0
  }
}
```

Important:
- these row objects still expose legacy names like `qwen`
- the new frontend should normalize them immediately into a canonical view model

### 4. `GET /api/jobs/{id}`

Use to poll async human eval jobs.

Implementation:
- [`api_job_detail()`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs)
- response type:
  [`Job`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/db.rs)

Response shape:

```json
{
  "id": 42,
  "job_type": "prototype_bakeoff",
  "status": "running | completed | failed",
  "config": "...json string or null",
  "log": "string log",
  "result": "json string or null",
  "created_at": "...",
  "finished_at": "..."
}
```

Notes:
- `result` is a JSON string, not structured JSON
- while running, `result` may already contain a partial snapshot
- client should `JSON.parse(job.result)` when present

### 5. `POST /api/correct-prototype/bakeoff/detail`

Use to lazy-load one human eval row’s full inspector.

Implementation:
- [`api_correct_prototype_bakeoff_detail()`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs)

Request body:

```json
{
  "source": "human",
  "recording_id": 123,
  "transcript": "canonical transcript",
  "expected": "expected clean sentence",
  "current": "current transcript",
  "prototype": "prototype output",
  "use_model_reranker": true,
  "use_prototype_adapters": true,
  "reranker_mode": "trained",
  "prototype_reranker_train_id": 262,

  "qwen": "legacy alias for transcript"
}
```

Canonical request field:
- `transcript`

Response shape:

```json
{
  "ok": true,
  "recording_id": 123,
  "transcript_label": "Parakeet",
  "transcript": "Parakeet transcript",
  "transcript_error": null,
  "current": "legacy alias",
  "qwen": "legacy alias",
  "parakeet": "legacy alias",
  "cohere": "",
  "parakeet_alignment": [
    {"w":"word","s":0.12,"e":0.42,"c":0.98}
  ],
  "qwen_error": null,
  "parakeet_error": null,
  "cohere_error": null,
  "correction_input": "parakeet",
  "elapsed_ms": 4200,
  "alignments": {
    "timing_source": "espeak_zipa_dp",
    "expected": [...],
    "espeak": [...],
    "qwen": [...],
    "current": [...],
    "prototype": [...],
    "zipa": [...],
    "zipa_espeak": [...],
    "zipa_qwen": [...]
  },
  "zipa_trace": {...},
  "prototype_trace": {
    "corrected": "...",
    "accepted": [...],
    "proposals": [...],
    "sentence_candidates": [...],
    "reranker": {...}
  }
}
```

Important:
- this is the canonical data source for the row inspector
- the current HTML lazily loads this only when the user clicks `Details`

### 6. `GET /api/author/recordings/{id}/audio`

Use for playback in human eval.

Implementation route:
- [`main.rs`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs)

Notes:
- binary audio endpoint
- current UI feeds it directly to `new Audio(...)`

## Canonical Frontend Model

The new app should normalize backend data immediately into this shape:

```ts
type EvalTranscriptSource = "parakeet" | "transcript" | "unknown";

type TimedToken = {
  w: string;
  s: number;
  e: number;
  c?: number | null;
};

type PrototypeAlignments = {
  timingSource: string;
  expected?: TimedToken[];
  espeak?: TimedToken[];
  current?: TimedToken[];
  prototype?: TimedToken[];
  zipa?: TimedToken[];
  zipaEspeak?: TimedToken[];
};

type EvalInspectorData = {
  transcript: string;
  transcriptLabel: string;
  transcriptError?: string | null;
  transcriptSource: EvalTranscriptSource;
  parakeetAlignment: TimedToken[];
  correctionInput: string;
  elapsedMs?: number | null;
  alignments: PrototypeAlignments;
  zipaTrace?: unknown;
  prototype: {
    corrected: string;
    accepted: unknown[];
    proposals: unknown[];
    sentenceCandidates: unknown[];
    reranker?: unknown;
  };
};
```

Normalization rules:

- `transcript = payload.transcript ?? payload.current ?? payload.parakeet ?? payload.qwen ?? ""`
- `transcriptLabel = payload.transcript_label ?? "Transcript"`
- `alignments.espeak = payload.alignments?.espeak ?? payload.alignments?.qwen ?? []`
- `alignments.zipaEspeak = payload.alignments?.zipa_espeak ?? payload.alignments?.zipa_qwen ?? []`

The new frontend should stop treating `qwen` as canonical.

## What To Rebuild

Minimum components:

1. `EvalPage`
2. `LiveEvalPanel`
3. `HumanEvalPanel`
4. `EvalCaseCard`
5. `EvalInspector`
6. `EvalPlaybackBar`
7. `EvalTimeline`
8. `ProposalInspector`

The important architectural rule:

- one shared inspector
- one shared playback controller
- one shared timeline renderer

Do not keep separate “live” and “human” timeline implementations.

## What To Ignore

Do not port these first:

- the rest of the dashboard
- `#/vocab`
- `#/corpus`
- old applied-eval UI
- deprecated Qwen/Cohere eval rows

This rewrite should be Parakeet-first for eval.

## Current UX Problems To Fix

These are known issues in the current `index.html` implementation:

1. too much duplicated shell code around the shared inspector
2. no proper scrubber
3. playback controls not consistently visible where the lanes are
4. summary rows still expose too much noise
5. lane sets have drifted over time
6. legacy field names (`qwen`, `current`) leak into the UI contract

## Recommended Rewrite Policy

Use these lanes in the new inspector:

1. `Parakeet`
2. `eSpeak`
3. `ZIPA@eSpeak`
4. `ZIPA`

Optional debug-only lanes later:

- `expected`
- `prototype`
- `current`

Default eval row count:
- use `5` in the new UI, not `150`

Human eval interaction model:

1. run batch
2. show compact list of cases
3. click row to open one shared inspector
4. lazy-load detail
5. playback + scrubber available inside the inspector, not somewhere else

## Concrete Backend Debt Still Present

These are backend cleanup items, not blockers for the rewrite:

1. remove legacy aliases:
   - `qwen`
   - `current`
   - `zipa_qwen`
2. rename `qwen_alignment` / `zipa_by_qwen` in Rust internals where they now really mean eSpeak-aligned transcript timing
3. emit structured job results instead of JSON strings in `/api/jobs/{id}`

Do not wait for those before starting the TS app.

## Files To Read

Backend:
- [`/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs)
- [`/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs)
- [`/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/db.rs`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/db.rs)

Current frontend reference only:
- [`/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/static/index.html`](/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/static/index.html)

## Suggested First Deliverable

Only rebuild this flow:

1. record live audio
2. run `/api/asr/dual`
3. run `/api/correct-prototype`
4. render shared inspector with:
   - transcript
   - accepted edits
   - proposal selector
   - timeline
   - playback bar

Then add:

5. human eval list
6. async bakeoff job polling
7. lazy-loaded case detail

That sequence keeps the rewrite bounded and useful quickly.

## Recommended Dev Topology

During development, do not keep redeploying the frontend to `souffle`.

Preferred setup:

1. Vite app runs locally on this laptop
2. Rust backend keeps running remotely on `souffle`
3. frontend talks to the backend over Tailscale

Practical recommendation:

- use a Vite dev proxy
- point it at the remote backend, for example:
  - `http://souffle.dropbear-piranha.ts.net:3456`
  - or the Tailscale hostname/IP directly

That gives:

- local hot reload
- no frontend redeploy loop
- same backend/data/model environment as production-ish eval
- no browser CORS pain if the proxy owns `/api/*`

So the frontend should be built assuming:

- app origin: local Vite server
- API origin in dev: proxied through Vite
- API origin in production later: configurable

If someone insists on direct browser-to-remote API calls instead of proxying, then the Rust backend will need explicit CORS. That is not the preferred first step.
