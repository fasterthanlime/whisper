// Types for bee-roll cut-trace JSONL events.
// Keep in sync with rust/bee-roll/src/cut_trace.rs

export interface WordSpan {
  start: number; // start token index (utterance-global, inclusive)
  end: number; // end token index (utterance-global, exclusive)
  text: string;
  start_secs: number | null; // ZIPA-aligned start time
  end_secs: number | null; // ZIPA-aligned end time
  region: "stable" | "carry" | "preview";
}

export interface FeedStartEvent {
  event: "feed_start";
  feed_index: number;
  audio_end_secs: number;
}

export interface PlanPreviewDecodeEvent {
  event: "plan_preview_decode";
  feed_index: number;
  stable_through: number;
  preview_from: number;
  tape_end: number;
  retained_decoder_position: number;
  current_kv_position: number;
}

export interface RewritePreviewEvent {
  event: "rewrite_preview";
  feed_index: number;
  rollback_to: number;
  decoder_position: number;
  current_kv_position: number;
}

export interface UpdatePreviewFromEvent {
  event: "update_preview_from";
  feed_index: number;
  stable_through: number;
  tape_end: number;
  target: number;
  preview_from: number;
}

export interface CutCandidateEvent {
  event: "cut_candidate";
  feed_index: number;
  stable_through: number;
  preview_from: number;
  latest_legal_boundary: number;
  chosen_boundary: number | null;
}

export interface CutAppliedEvent {
  event: "cut_applied";
  feed_index: number;
  prev_stable_through: number;
  prev_preview_from: number;
  new_stable: number;
  cut_sample: number | null;
  cut_sample_secs: number | null;
  applied: boolean;
  context_debug: string;
}

export interface FeedEndEvent {
  event: "feed_end";
  feed_index: number;
  stable_through: number;
  preview_from: number;
  tape_end: number;
  transcript: string;
  word_spans: WordSpan[];
  has_repeated_ngrams: boolean;
}

export type TraceEvent =
  | FeedStartEvent
  | PlanPreviewDecodeEvent
  | RewritePreviewEvent
  | UpdatePreviewFromEvent
  | CutCandidateEvent
  | CutAppliedEvent
  | FeedEndEvent;

export interface FeedGroup {
  feedIndex: number;
  events: TraceEvent[];
  feedStart: FeedStartEvent | null;
  feedEnd: FeedEndEvent | null;
  cutApplied: CutAppliedEvent | null;
  cutCandidate: CutCandidateEvent | null;
}

export function groupByFeed(events: TraceEvent[]): Map<number, FeedGroup> {
  const map = new Map<number, FeedGroup>();
  for (const ev of events) {
    const fi = ev.feed_index;
    if (!map.has(fi)) {
      map.set(fi, {
        feedIndex: fi,
        events: [],
        feedStart: null,
        feedEnd: null,
        cutApplied: null,
        cutCandidate: null,
      });
    }
    const g = map.get(fi)!;
    g.events.push(ev);
    if (ev.event === "feed_start") g.feedStart = ev;
    if (ev.event === "feed_end") g.feedEnd = ev;
    if (ev.event === "cut_applied") g.cutApplied = ev;
    if (ev.event === "cut_candidate") g.cutCandidate = ev;
  }
  return map;
}
