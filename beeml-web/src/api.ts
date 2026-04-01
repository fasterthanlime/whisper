import type {
  EvalInspectorData,
  BakeoffResult,
  BakeoffEntry,
  Job,
  TimedToken,
  SentenceCandidate,
  Reranker,
  RetrievalDebugResult,
} from "./types";

// --- Raw backend shapes (private, normalized immediately) ---

type RawAsrDualResponse = {
  parakeet: string;
  parakeet_alignment: TimedToken[];
};

type RawCorrectResponse = {
  original: string;
  corrected: string;
  accepted: unknown[];
  proposals: unknown[];
  sentence_candidates: unknown[];
  alignments: {
    timing_source: string;
    transcript?: TimedToken[];
    espeak?: TimedToken[];
    zipa?: TimedToken[];
    zipa_espeak?: TimedToken[];
    expected?: TimedToken[];
    prototype?: TimedToken[];
  };
  zipa_trace?: unknown;
  reranker?: unknown;
};

type RawDetailResponse = {
  ok: boolean;
  recording_id: number;
  transcript_label?: string;
  transcript?: string;
  transcript_error?: string | null;
  parakeet_alignment: TimedToken[];
  elapsed_ms?: number;
  alignments: RawCorrectResponse["alignments"];
  zipa_trace?: unknown;
  prototype_trace?: {
    corrected: string;
    accepted: unknown[];
    proposals: unknown[];
    sentence_candidates: unknown[];
    reranker?: unknown;
  };
};

type RawBakeoffEntry = {
  term: string;
  case_id: string;
  source: string;
  transcript_label?: string;
  transcript_error?: string | null;
  transcript: string;
  expected: string;
  recording_id: number;
  hit_count: number;
  prototype_ok: boolean;
  prototype_target_ok: boolean;
  prototype: string;
  analysis: {
    failure_reason: string;
    exact_ok: boolean;
    target_ok: boolean;
  };
  prototype_trace_excerpt: {
    proposal_count: number;
    sentence_candidate_count: number;
    accepted_count: number;
  };
};

type RawBakeoffResult = {
  source: string;
  limit: number;
  processed: number;
  summary: {
    n: number;
    prototype: number;
    prototype_wrong: number;
    both_wrong: number;
  };
  entries: RawBakeoffEntry[];
};

type RawRetrievalCandidate = {
  alias_id: number;
  term: string;
  alias_text: string;
  alias_source: string;
  matched_view: string;
  qgram_overlap: number;
  token_count_match?: boolean;
  phone_count_delta: number;
  coarse_score: number;
  phonetic_score?: number;
};

type RawRetrievalDebugResponse = {
  ok: boolean;
  transcript: string;
  summary: {
    alias_count: number;
    returned_spans: number;
    max_span_words: number;
    max_candidates_per_span: number;
  };
  timing: {
    db_ms: number;
    lexicon_ms: number;
    index_ms: number;
    span_enumeration_ms: number;
    shortlist_ms: number;
    verify_ms: number;
    total_ms: number;
  };
  spans: {
    token_start: number;
    token_end: number;
    char_start: number;
    char_end: number;
    start_sec?: number | null;
    end_sec?: number | null;
    text: string;
    ipa_tokens: string[];
    reduced_ipa_tokens: string[];
    shortlist: RawRetrievalCandidate[];
    verified: RawRetrievalCandidate[];
  }[];
};

// --- Normalization ---

function normalizeAlignments(raw: RawCorrectResponse["alignments"] | null | undefined) {
  if (!raw) {
    return { timingSource: "", transcript: [], expected: [], espeak: [], prototype: [], zipa: [], zipaEspeak: [] };
  }
  return {
    timingSource: raw.timing_source,
    transcript: raw.transcript,
    expected: raw.expected,
    espeak: raw.espeak,
    prototype: raw.prototype,
    zipa: raw.zipa,
    zipaEspeak: raw.zipa_espeak,
  };
}

function normalizeSentenceCandidates(raw: unknown[]): SentenceCandidate[] {
  if (!raw) return [];
  return raw.map((c: any) => ({
    label: c.label ?? "",
    text: c.text ?? "",
    edits: (c.edits ?? []).map((e: any) => ({
      tokenStart: e.token_start,
      tokenEnd: e.token_end,
      charStart: e.char_start,
      charEnd: e.char_end,
      from: e.from,
      matchedForm: e.matched_form,
      fromPhonemes: e.from_phonemes,
      to: e.to,
      toPhonemes: e.to_phonemes,
      via: e.via,
      score: e.score,
      phoneticScore: e.phonetic_score,
    })),
    score: c.score ?? 0,
  }));
}

function normalizeReranker(raw: unknown): Reranker | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as any;
  return {
    mode: r.mode ?? "",
    chosenIndex: r.chosen_index,
    chosenText: r.chosen_text,
    candidateCount: r.candidate_count ?? 0,
    candidates: r.candidates?.map((c: any) => ({
      index: c.index,
      text: c.text,
      heuristicScore: c.heuristic_score,
      yesProb: c.yes_prob,
      noProb: c.no_prob,
      answer: c.answer,
    })),
  };
}

function normalizeCorrectResponse(
  raw: RawCorrectResponse,
  transcript: string,
  transcriptLabel: string,
  parakeetAlignment: TimedToken[],
): EvalInspectorData {
  return {
    transcript,
    transcriptLabel,
    transcriptSource: "parakeet",
    parakeetAlignment,
    alignments: normalizeAlignments(raw.alignments),
    zipaTrace: raw.zipa_trace,
    prototype: {
      corrected: raw.corrected,
      accepted: raw.accepted as EvalInspectorData["prototype"]["accepted"],
      proposals: raw.proposals as EvalInspectorData["prototype"]["proposals"],
      sentenceCandidates: normalizeSentenceCandidates(raw.sentence_candidates),
      reranker: normalizeReranker(raw.reranker),
    },
  };
}

function normalizeDetailResponse(raw: RawDetailResponse): EvalInspectorData {
  const transcript = raw.transcript ?? "";
  const transcriptLabel = raw.transcript_label ?? "Parakeet";
  const pt = raw.prototype_trace;
  return {
    transcript,
    transcriptLabel,
    transcriptError: raw.transcript_error,
    transcriptSource: "parakeet",
    parakeetAlignment: raw.parakeet_alignment ?? [],
    elapsedMs: raw.elapsed_ms,
    alignments: normalizeAlignments(raw.alignments),
    zipaTrace: raw.zipa_trace,
    prototype: pt
      ? {
          corrected: pt.corrected,
          accepted: pt.accepted as EvalInspectorData["prototype"]["accepted"],
          proposals: pt.proposals as EvalInspectorData["prototype"]["proposals"],
          sentenceCandidates: normalizeSentenceCandidates(pt.sentence_candidates),
          reranker: normalizeReranker(pt.reranker),
        }
      : { corrected: "", accepted: [], proposals: [], sentenceCandidates: [] },
  };
}

function normalizeBakeoffEntry(raw: RawBakeoffEntry): BakeoffEntry {
  return {
    term: raw.term,
    caseId: raw.case_id,
    source: raw.source,
    expected: raw.expected,
    transcript: raw.transcript,
    recordingId: raw.recording_id,
    hitCount: raw.hit_count,
    prototypeOk: raw.prototype_ok,
    prototypeTargetOk: raw.prototype_target_ok,
    prototype: raw.prototype,
    analysis: {
      failureReason: raw.analysis.failure_reason,
      exactOk: raw.analysis.exact_ok,
      targetOk: raw.analysis.target_ok,
    },
    prototypeTraceExcerpt: {
      proposalCount: raw.prototype_trace_excerpt.proposal_count,
      sentenceCandidateCount:
        raw.prototype_trace_excerpt.sentence_candidate_count,
      acceptedCount: raw.prototype_trace_excerpt.accepted_count,
    },
  };
}

function normalizeBakeoffResult(raw: RawBakeoffResult): BakeoffResult {
  return {
    source: raw.source,
    limit: raw.limit,
    processed: raw.processed,
    summary: {
      n: raw.summary.n,
      prototype: raw.summary.prototype,
      prototypeWrong: raw.summary.prototype_wrong,
      bothWrong: raw.summary.both_wrong,
    },
    entries: raw.entries.map(normalizeBakeoffEntry),
  };
}

function normalizeRetrievalCandidate(raw: RawRetrievalCandidate) {
  return {
    aliasId: raw.alias_id,
    term: raw.term,
    aliasText: raw.alias_text,
    aliasSource: raw.alias_source,
    matchedView: raw.matched_view,
    qgramOverlap: raw.qgram_overlap,
    tokenCountMatch: raw.token_count_match,
    phoneCountDelta: raw.phone_count_delta,
    coarseScore: raw.coarse_score,
    phoneticScore: raw.phonetic_score,
  };
}

function normalizeRetrievalDebugResponse(
  raw: RawRetrievalDebugResponse,
): RetrievalDebugResult {
  return {
    transcript: raw.transcript,
    summary: {
      aliasCount: raw.summary.alias_count,
      returnedSpans: raw.summary.returned_spans,
      maxSpanWords: raw.summary.max_span_words,
      maxCandidatesPerSpan: raw.summary.max_candidates_per_span,
    },
    timing: {
      dbMs: raw.timing.db_ms,
      lexiconMs: raw.timing.lexicon_ms,
      indexMs: raw.timing.index_ms,
      spanEnumerationMs: raw.timing.span_enumeration_ms,
      shortlistMs: raw.timing.shortlist_ms,
      verifyMs: raw.timing.verify_ms,
      totalMs: raw.timing.total_ms,
    },
    spans: raw.spans.map((span) => ({
      tokenStart: span.token_start,
      tokenEnd: span.token_end,
      charStart: span.char_start,
      charEnd: span.char_end,
      startSec: span.start_sec,
      endSec: span.end_sec,
      text: span.text,
      ipaTokens: span.ipa_tokens,
      reducedIpaTokens: span.reduced_ipa_tokens,
      shortlist: span.shortlist.map(normalizeRetrievalCandidate),
      verified: span.verified.map(normalizeRetrievalCandidate) as RetrievalDebugResult["spans"][number]["verified"],
    })),
  };
}

// --- API calls ---

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

/** Record audio → ASR transcription */
export async function asrDual(audioWav: Blob): Promise<RawAsrDualResponse> {
  const res = await fetch("/api/asr/dual", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: audioWav,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

/** Run correction on a transcript */
export async function correctPrototype(params: {
  transcript: string;
  audioWavBase64?: string;
  trainId: number;
}): Promise<EvalInspectorData> {
  const raw = await post<RawCorrectResponse>("/api/correct-prototype", {
    transcript: params.transcript,
    audio_wav_base64: params.audioWavBase64,
    use_model_reranker: true,
    use_prototype_adapters: true,
    reranker_mode: "trained",
    prototype_reranker_train_id: params.trainId,
  });
  return normalizeCorrectResponse(raw, params.transcript, "Parakeet", []);
}

/** Start a human eval bakeoff job */
export async function startBakeoff(params: {
  limit: number;
  trainId: number;
  caseIds?: string[];
  randomize?: boolean;
  sampleSeed?: number;
}): Promise<{ jobId: number }> {
  const raw = await post<{ job_id: number }>(
    "/api/correct-prototype/bakeoff",
    {
      source: "human",
      limit: params.limit,
      case_ids: params.caseIds,
      randomize: params.randomize ?? true,
      sample_seed: params.sampleSeed,
      use_model_reranker: true,
      use_prototype_adapters: true,
      reranker_mode: "trained",
      prototype_reranker_train_id: params.trainId,
    },
  );
  return { jobId: raw.job_id };
}

/** Poll a job's status */
export async function getJob(id: number): Promise<Job> {
  const res = await fetch(`/api/jobs/${id}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const raw = await res.json();
  return {
    id: raw.id,
    jobType: raw.job_type,
    status: raw.status,
    config: raw.config,
    log: raw.log,
    result: raw.result,
    createdAt: raw.created_at,
    finishedAt: raw.finished_at,
  };
}

/** Parse a completed bakeoff job's result */
export function parseBakeoffResult(job: Job): BakeoffResult | null {
  if (!job.result) return null;
  const raw: RawBakeoffResult = JSON.parse(job.result);
  return normalizeBakeoffResult(raw);
}

/** Lazy-load full detail for one human eval case */
export async function bakeoffDetail(params: {
  recordingId: number;
  transcript: string;
  expected: string;
  prototype: string;
  trainId: number;
}): Promise<EvalInspectorData> {
  const raw = await post<RawDetailResponse>(
    "/api/correct-prototype/bakeoff/detail",
    {
      source: "human",
      recording_id: params.recordingId,
      transcript: params.transcript,
      expected: params.expected,
      prototype: params.prototype,
      use_model_reranker: true,
      use_prototype_adapters: true,
      reranker_mode: "trained",
      prototype_reranker_train_id: params.trainId,
    },
  );
  return normalizeDetailResponse(raw);
}

export async function phoneticRetrievalDebug(params: {
  transcript: string;
  maxSpanWords?: number;
  maxCandidatesPerSpan?: number;
  maxSpans?: number;
}): Promise<RetrievalDebugResult> {
  const raw = await post<RawRetrievalDebugResponse>(
    "/api/correct-prototype/retrieval-debug",
    {
      transcript: params.transcript,
      max_span_words: params.maxSpanWords,
      max_candidates_per_span: params.maxCandidatesPerSpan,
      max_spans: params.maxSpans,
    },
  );
  return normalizeRetrievalDebugResponse(raw);
}

/** Audio URL for a human eval recording */
export function recordingAudioUrl(recordingId: number): string {
  return `/api/author/recordings/${recordingId}/audio`;
}
