# Cut-Trace Viewer

Structured per-feed trace of every cut decision in bee-roll, with a live
React viewer built into beeml-web.

## Enable the trace

Set `BEE_ROLL_CUT_TRACE` to an absolute path before running bee-roll:

```sh
export BEE_ROLL_CUT_TRACE=/tmp/cuts.jsonl
```

The file is created (or truncated) when bee-roll starts the first utterance,
and one JSON object per event is appended per feed.

## Event types recorded

| event | when |
|---|---|
| `feed_start` | top of each feed, with current audio end in seconds |
| `plan_preview_decode` | after planning rollback boundaries |
| `rewrite_preview` | before truncating the tape |
| `update_preview_from` | after the preview-from boundary is updated |
| `cut_candidate` | result of `find_auto_cut_boundary` (Auto mode only) |
| `cut_applied` | cut outcome — `applied: true/false`, `cut_sample_secs` |
| `feed_end` | end of feed: full transcript, word spans with ZIPA timing and region tags, repeated-bigram flag |

Word spans carry `start_secs`/`end_secs` from ZIPA alignment and a `region`
field (`"stable"`, `"carry"`, or `"preview"`).

## Open the viewer

In two terminals:

```sh
# Terminal 1 — SSE server (reads the trace file)
BEE_ROLL_CUT_TRACE=/tmp/cuts.jsonl node debug/cut-trace-server.js

# Terminal 2 — beeml-web dev server
pnpm --dir beeml-web dev
```

Then open: **http://localhost:5173/cut-trace**

The viewer connects over SSE and updates the moment the trace file changes —
no reload or manual refresh needed. Run bee-roll in a third terminal and watch
the feed list populate in real time.

## What you can see

- **Feed list** (left): every feed indexed in order. Green `CUT` badge when
  stable was promoted, orange `LOOP` badge when repeated bigrams were
  detected.
- **Timeline** (right top): word boxes positioned at their ZIPA-aligned times,
  coloured by region (green = stable, amber = carry, red = preview). A red
  vertical line marks the audio cut point when a cut was applied.
- **Events table** (right bottom): all events for the selected feed with all
  structured fields inline.
- **context_debug** collapsible: the raw token-window debug string from Rust.
