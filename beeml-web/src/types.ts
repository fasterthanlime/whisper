/** A timed token from alignment data */
export type TimedToken = {
  w: string;
  s: number;
  e: number;
  c?: number | null;
  /** Mean log-probability from ASR decoder (per-word aggregate) */
  meanLogprob?: number | null;
  /** Minimum log-probability from ASR decoder */
  minLogprob?: number | null;
  /** Mean top1−top2 margin from ASR decoder */
  meanMargin?: number | null;
  /** Minimum top1−top2 margin from ASR decoder */
  minMargin?: number | null;
};

export type EvalTranscriptSource = "qwen" | "transcript" | "unknown";

export type PrototypeAlignments = {
  timingSource: string;
  transcript?: TimedToken[];
  expected?: TimedToken[];
  espeak?: TimedToken[];
  prototype?: TimedToken[];
  zipa?: TimedToken[];
  zipaEspeak?: TimedToken[];
};

export type ProposalCandidate = {
  term: string;
  source: string;
  matchedForm?: string;
  rank: number;
  similarity?: number | null;
  phonemes?: string;
};

export type Proposal = {
  spanText: string;
  spanStart: number;
  spanEnd: number;
  candidates: ProposalCandidate[];
};

export type AcceptedEdit = {
  original: string;
  replacement: string;
  score: number;
  delta: number;
};

export type CandidateEdit = {
  tokenStart: number;
  tokenEnd: number;
  charStart: number;
  charEnd: number;
  from: string;
  matchedForm: string;
  fromPhonemes?: string | null;
  to: string;
  toPhonemes?: string | null;
  via: string;
  score: number;
  phoneticScore?: number | null;
};

export type SentenceCandidate = {
  label: string;
  text: string;
  edits: CandidateEdit[];
  score: number;
};

export type Reranker = {
  mode: string;
  chosenIndex?: number | null;
  chosenText?: string | null;
  candidateCount: number;
  candidates?: {
    index: number;
    text: string;
    heuristicScore: number;
    yesProb: number;
    noProb: number;
    answer: string;
  }[];
};

export type PrototypeTrace = {
  corrected: string;
  accepted: AcceptedEdit[];
  proposals: Proposal[];
  sentenceCandidates: SentenceCandidate[];
  reranker?: Reranker | null;
};

export type PhoneticRescueCandidate = {
  term: string;
  aliasText: string;
  aliasSource: string;
  candidateNormalized: string[];
  featureSimilarity?: number | null;
  similarityDelta?: number | null;
};

export type PhoneticAlignmentKind = "Match" | "Substitute" | "Insert" | "Delete";
export type PhoneticAnchorConfidence = "Low" | "Medium" | "High";
export type PhoneticSpanUsefulness = "Low" | "Medium" | "High";
export type PhoneticSpanClass =
  | "Repeat"
  | "ShortCodeTerm"
  | "VowelHeavy"
  | "ProperNoun"
  | "FunctionWord"
  | "Ordinary";

export type PhoneticAlignmentOp = {
  kind: PhoneticAlignmentKind;
  transcriptIndex?: number | null;
  zipaIndex?: number | null;
  transcriptToken?: string | null;
  zipaToken?: string | null;
  cost: number;
};

export type PhoneticRescueSpan = {
  spanText: string;
  tokenStart: number;
  tokenEnd: number;
  startSec: number;
  endSec: number;
  zipaNormStart: number;
  zipaNormEnd: number;
  zipaRaw: string[];
  zipaNormalized: string[];
  transcriptNormalized: string[];
  transcriptPhoneCount: number;
  chosenZipaPhoneCount: number;
  transcriptSimilarity?: number | null;
  transcriptFeatureSimilarity?: number | null;
  projectedAlignmentScore?: number | null;
  chosenAlignmentScore?: number | null;
  secondBestAlignmentScore?: number | null;
  alignmentScoreGap?: number | null;
  alignmentSource: string;
  anchorConfidence: PhoneticAnchorConfidence;
  spanClass: PhoneticSpanClass;
  spanUsefulness: PhoneticSpanUsefulness;
  zipaRescueEligible: boolean;
  alignment: PhoneticAlignmentOp[];
  candidates: PhoneticRescueCandidate[];
};

export type PhoneticWordAlignment = {
  wordText: string;
  tokenStart: number;
  tokenEnd: number;
  startSec: number;
  endSec: number;
  transcriptNormalized: string[];
  zipaNormStart: number;
  zipaNormEnd: number;
  zipaRaw: string[];
  zipaNormalized: string[];
  alignment: PhoneticAlignmentOp[];
};

export type PhoneticRescueTrace = {
  snapshotRevision: bigint;
  alignedTranscript: string;
  pendingText: string;
  fullTranscript: string;
  tailAmbiguity: {
    pendingTokenCount: number;
    lowConcentrationCount: number;
    lowMarginCount: number;
    volatileTokenCount: number;
    meanConcentration: number;
    meanMargin: number;
    minConcentration: number;
    minMargin: number;
  };
  worstRawSpanIndex?: number | null;
  worstContentfulSpanIndex?: number | null;
  bestRescueSpanIndex?: number | null;
  utteranceZipaRaw: string[];
  utteranceZipaNormalized: string[];
  utteranceTranscriptNormalized: string[];
  utteranceSimilarity?: number | null;
  utteranceFeatureSimilarity?: number | null;
  utteranceAlignment: PhoneticAlignmentOp[];
  wordAlignments: PhoneticWordAlignment[];
  spans: PhoneticRescueSpan[];
};

/** Canonical inspector data — normalized from any backend response */
export type EvalInspectorData = {
  transcript: string;
  transcriptLabel: string;
  transcriptError?: string | null;
  transcriptSource: EvalTranscriptSource;
  expected?: string;
  qwenAlignment: TimedToken[];
  elapsedMs?: number | null;
  alignments: PrototypeAlignments;
  zipaTrace?: unknown;
  phoneticTrace?: PhoneticRescueTrace | null;
  prototype: PrototypeTrace;
};

/** A single case from a human eval bakeoff run */
export type BakeoffEntry = {
  term: string;
  caseId: string;
  source: string;
  expected: string;
  transcript: string;
  recordingId: number;
  hitCount: number;
  prototypeOk: boolean;
  prototypeTargetOk: boolean;
  prototype: string;
  analysis: {
    failureReason: string;
    exactOk: boolean;
    targetOk: boolean;
  };
  prototypeTraceExcerpt: {
    proposalCount: number;
    sentenceCandidateCount: number;
    acceptedCount: number;
  };
};

export type BakeoffSummary = {
  n: number;
  prototype: number;
  prototypeWrong: number;
  bothWrong: number;
};

export type BakeoffResult = {
  source: string;
  limit: number;
  processed: number;
  summary: BakeoffSummary;
  entries: BakeoffEntry[];
};

export type JobStatus = "running" | "completed" | "failed";

export type Job = {
  id: number;
  jobType: string;
  status: JobStatus;
  config?: string | null;
  log: string;
  result?: string | null;
  createdAt: string;
  finishedAt?: string | null;
};

export type RetrievalCandidate = {
  aliasId: number;
  term: string;
  aliasText: string;
  aliasSource: string;
  matchedView: string;
  qgramOverlap: number;
  tokenCountMatch?: boolean;
  phoneCountDelta: number;
  coarseScore: number;
};

export type VerifiedRetrievalCandidate = RetrievalCandidate & {
  phoneticScore: number;
};

export type RetrievalDebugSpan = {
  tokenStart: number;
  tokenEnd: number;
  charStart: number;
  charEnd: number;
  startSec?: number | null;
  endSec?: number | null;
  text: string;
  ipaTokens: string[];
  reducedIpaTokens: string[];
  shortlist: RetrievalCandidate[];
  verified: VerifiedRetrievalCandidate[];
};

export type RetrievalDebugTiming = {
  dbMs: number;
  lexiconMs: number;
  indexMs: number;
  spanEnumerationMs: number;
  shortlistMs: number;
  verifyMs: number;
  totalMs: number;
};

export type RetrievalDebugResult = {
  transcript: string;
  summary: {
    aliasCount: number;
    returnedSpans: number;
    maxSpanWords: number;
    maxCandidatesPerSpan: number;
  };
  timing: RetrievalDebugTiming;
  spans: RetrievalDebugSpan[];
};
