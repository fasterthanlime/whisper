// Types for bee-roll cut-trace JSONL events.
// Keep in sync with rust/bee-roll/src/cut_trace.rs

export interface ZipaTimingTrace {
  kind: "aligned" | "deleted" | "projected" | "invalid";
  start_secs: number | null;
  end_secs: number | null;
  projected_at: number | null;
  normalized_start: number | null;
  normalized_end: number | null;
}

export interface AsrAlternativeTrace {
  token_id: number;
  token_text: string;
  logit: number;
}

export interface ZipaPhoneSpanTrace {
  phone: string;
  start_secs: number;
  end_secs: number;
}

export interface IndexRangeTrace {
  start: number;
  end: number;
}

export interface WordTokenTrace {
  token_index: number;
  token_text: string;
  token_surface: string;
  token_start_secs: number;
  token_end_secs: number;
  asr_margin: number | null;
  asr_concentration: number | null;
  asr_alternatives: AsrAlternativeTrace[];
  g2p_ipa: string | null;
  transcript_phones: string[];
  zipa_phone_spans: ZipaPhoneSpanTrace[];
  zipa_timing: ZipaTimingTrace;
}

export interface WordSpan {
  start: number;
  end: number;
  text: string;
  start_secs: number | null;
  end_secs: number | null;
  region: "stable" | "carry" | "preview";
  tokens: WordTokenTrace[];
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

export interface PreviewApplyTokenChange {
  token_index: number;
  token_text: string;
  old_timing: ZipaTimingTrace;
  new_timing: ZipaTimingTrace;
}

export interface PreviewApplyEvent {
  event: "preview_apply";
  feed_index: number;
  stable_through: number;
  preview_from: number;
  changed_tokens: PreviewApplyTokenChange[];
}

export interface PreviewAlignmentTokenTrace {
  token_index: number;
  token_text: string;
  token_surface: string;
  word_index: number | null;
  word_surface: string | null;
  projected_range: IndexRangeTrace | null;
  raw_phone_range: IndexRangeTrace | null;
  zipa_phone_spans: ZipaPhoneSpanTrace[];
  zipa_timing: ZipaTimingTrace;
}

export interface PreviewAlignmentEvent {
  event: "preview_alignment";
  feed_index: number;
  stable_through: number;
  preview_from: number;
  seam_start: number;
  tokens: PreviewAlignmentTokenTrace[];
}

export interface ZipaCacheTrimEvent {
  event: "zipa_cache_trim";
  feed_index: number;
  cut_sample: number;
  cut_sample_secs: number;
  before_audio_start_secs: number;
  before_audio_end_secs: number;
  after_audio_start_secs: number;
  after_audio_end_secs: number;
  before_spans: ZipaPhoneSpanTrace[];
  after_spans: ZipaPhoneSpanTrace[];
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
  | PreviewAlignmentEvent
  | PreviewApplyEvent
  | CutAppliedEvent
  | ZipaCacheTrimEvent
  | FeedEndEvent;

export interface FeedGroup {
  feedIndex: number;
  events: TraceEvent[];
  feedStart: FeedStartEvent | null;
  feedEnd: FeedEndEvent | null;
  cutApplied: CutAppliedEvent | null;
  cutCandidate: CutCandidateEvent | null;
  previewAlignment: PreviewAlignmentEvent | null;
  previewApply: PreviewApplyEvent | null;
  zipaCacheTrim: ZipaCacheTrimEvent | null;
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
        previewAlignment: null,
        previewApply: null,
        zipaCacheTrim: null,
      });
    }
    const group = map.get(fi)!;
    group.events.push(ev);
    if (ev.event === "feed_start") group.feedStart = ev;
    if (ev.event === "feed_end") group.feedEnd = ev;
    if (ev.event === "cut_applied") group.cutApplied = ev;
    if (ev.event === "cut_candidate") group.cutCandidate = ev;
    if (ev.event === "preview_alignment") group.previewAlignment = ev;
    if (ev.event === "preview_apply") group.previewApply = ev;
    if (ev.event === "zipa_cache_trim") group.zipaCacheTrim = ev;
  }
  return map;
}
