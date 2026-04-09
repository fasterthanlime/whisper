import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  CorpusAlignmentEvalJob,
  CorpusAlignmentEvalResult,
  CorpusAlignmentEvalRow,
  TranscribePhoneticTrace as RpcTranscribePhoneticTrace,
} from "../beeml.generated";
import { PhoneticRescuePanel } from "./PhoneticRescuePanel";
import type { PhoneticRescueTrace } from "../types";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
}

function formatPercentMetric(value: number | null | undefined) {
  return value == null ? "n/a" : `${Math.round(value * 100)}%`;
}

function formatTokens(tokens: string[]) {
  return tokens.length > 0 ? tokens.join(" ") : "∅";
}

function bucketLabel(bucket: string) {
  switch (bucket) {
    case "repeat_confusion":
      return "repeated phrase";
    case "dropped_word":
      return "missing word";
    case "extra_word":
      return "extra word";
    case "proper_noun":
      return "name";
    case "homophoneish":
      return "sounds alike";
    default:
      return bucket.replaceAll("_", " ");
  }
}

function toPhoneticTrace(trace: RpcTranscribePhoneticTrace): PhoneticRescueTrace {
  return {
    snapshotRevision: trace.snapshot_revision,
    alignedTranscript: trace.aligned_transcript,
    pendingText: trace.pending_text,
    fullTranscript: trace.full_transcript,
    tailAmbiguity: {
      pendingTokenCount: trace.tail_ambiguity.pending_token_count,
      lowConcentrationCount: trace.tail_ambiguity.low_concentration_count,
      lowMarginCount: trace.tail_ambiguity.low_margin_count,
      volatileTokenCount: trace.tail_ambiguity.volatile_token_count,
      meanConcentration: trace.tail_ambiguity.mean_concentration,
      meanMargin: trace.tail_ambiguity.mean_margin,
      minConcentration: trace.tail_ambiguity.min_concentration,
      minMargin: trace.tail_ambiguity.min_margin,
    },
    worstRawSpanIndex: trace.worst_raw_span_index,
    worstContentfulSpanIndex: trace.worst_contentful_span_index,
    bestRescueSpanIndex: trace.best_rescue_span_index,
    utteranceZipaRaw: trace.utterance_zipa_raw,
    utteranceZipaNormalized: trace.utterance_zipa_normalized,
    utteranceTranscriptNormalized: trace.utterance_transcript_normalized,
    utteranceSimilarity: trace.utterance_similarity,
    utteranceFeatureSimilarity: trace.utterance_feature_similarity,
    utteranceAlignment: trace.utterance_alignment.map((op) => ({
      kind: op.kind.tag,
      transcriptIndex: op.transcript_index,
      zipaIndex: op.zipa_index,
      transcriptToken: op.transcript_token,
      zipaToken: op.zipa_token,
      cost: op.cost,
    })),
    wordAlignments: trace.word_alignments.map((word) => ({
      wordText: word.word_text,
      tokenStart: word.token_start,
      tokenEnd: word.token_end,
      startSec: word.start_sec,
      endSec: word.end_sec,
      transcriptNormalized: word.transcript_normalized,
      zipaNormStart: word.zipa_norm_start,
      zipaNormEnd: word.zipa_norm_end,
      zipaRaw: word.zipa_raw,
      zipaNormalized: word.zipa_normalized,
      alignment: word.alignment.map((op) => ({
        kind: op.kind.tag,
        transcriptIndex: op.transcript_index,
        zipaIndex: op.zipa_index,
        transcriptToken: op.transcript_token,
        zipaToken: op.zipa_token,
        cost: op.cost,
      })),
    })),
    spans: trace.spans.map((span) => ({
      spanText: span.span_text,
      tokenStart: span.token_start,
      tokenEnd: span.token_end,
      startSec: span.start_sec,
      endSec: span.end_sec,
      zipaNormStart: span.zipa_norm_start,
      zipaNormEnd: span.zipa_norm_end,
      zipaRaw: span.zipa_raw,
      zipaNormalized: span.zipa_normalized,
      transcriptNormalized: span.transcript_normalized,
      transcriptPhoneCount: span.transcript_phone_count,
      chosenZipaPhoneCount: span.chosen_zipa_phone_count,
      transcriptSimilarity: span.transcript_similarity,
      transcriptFeatureSimilarity: span.transcript_feature_similarity,
      projectedAlignmentScore: span.projected_alignment_score,
      chosenAlignmentScore: span.chosen_alignment_score,
      secondBestAlignmentScore: span.second_best_alignment_score,
      alignmentScoreGap: span.alignment_score_gap,
      alignmentSource: span.alignment_source,
      anchorConfidence: span.anchor_confidence.tag,
      spanClass: span.span_class.tag,
      spanUsefulness: span.span_usefulness.tag,
      zipaRescueEligible: span.zipa_rescue_eligible,
      alignment: span.alignment.map((op) => ({
        kind: op.kind.tag,
        transcriptIndex: op.transcript_index,
        zipaIndex: op.zipa_index,
        transcriptToken: op.transcript_token,
        zipaToken: op.zipa_token,
        cost: op.cost,
      })),
      candidates: span.candidates.map((candidate) => ({
        term: candidate.term,
        aliasText: candidate.alias_text,
        aliasSource: candidate.alias_source.tag,
        candidateNormalized: candidate.candidate_normalized,
        featureSimilarity: candidate.feature_similarity,
        similarityDelta: candidate.similarity_delta,
      })),
    })),
  };
}

function RowCard({
  row,
  selected,
  onSelect,
}: {
  row: CorpusAlignmentEvalRow;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      className={`corpus-eval-row-card${selected ? " is-selected" : ""}`}
      style={{
        cursor: "pointer",
        userSelect: "text",
      }}
    >
      <div className="corpus-eval-row-topline">
        <span className="mini-badge">{row.term}</span>
        <span className="corpus-eval-row-score">phonetic {formatPercentMetric(row.utterance_feature_similarity)}</span>
      </div>
      <div className="corpus-eval-row-title">{row.prompt_text}</div>
      <div className="corpus-eval-row-subline">
        <span>heard: {row.asr_transcript}</span>
      </div>
      {row.error ? <div className="error-pill">{row.error}</div> : null}
    </div>
  );
}

export function CorpusAlignmentEvalPanel({ wsUrl }: { wsUrl: string }) {
  const [limit, setLimit] = useState(100);
  const [bucket, setBucket] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CorpusAlignmentEvalResult | null>(null);
  const [job, setJob] = useState<CorpusAlignmentEvalJob | null>(null);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [showRunComposer, setShowRunComposer] = useState(false);
  const autoStartedRef = useRef(false);

  const runEval = useCallback(async (options?: { limit?: number; bucket?: string; randomize?: boolean }) => {
    try {
      setStatus("Starting corpus alignment eval...");
      setError(null);
      setResult(null);
      setSelectedPromptId(null);
      const client = await connectBeeMl(wsUrl);
      const nextLimit = options?.limit ?? limit;
      const nextBucket = options?.bucket ?? bucket;
      const response = await client.startCorpusAlignmentEvalJob({
        limit: nextLimit,
        bucket: nextBucket.trim() ? nextBucket.trim() : null,
        randomize: options?.randomize ?? false,
      });
      if (!response.ok) throw new Error(response.error);
      setJob(response.value);
      setStatus("Corpus alignment eval running...");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [bucket, limit, wsUrl]);

  useEffect(() => {
    if (autoStartedRef.current) return;
    autoStartedRef.current = true;
    void runEval({ limit: 5, bucket: "", randomize: true });
  }, [runEval]);

  useEffect(() => {
    if (!job || job.status.tag !== "Running") return;
    let cancelled = false;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const client = await connectBeeMl(wsUrl);
          const response = await client.getCorpusAlignmentEvalJob(job.job_id);
          if (!response.ok) throw new Error(response.error);
          if (cancelled) return;
          setJob(response.value);
          if (response.value.status.tag === "Completed") {
            setResult(response.value.result);
            setSelectedPromptId(response.value.result?.rows[0]?.prompt_id ?? null);
            setStatus(null);
          } else if (response.value.status.tag === "Failed") {
            setError(response.value.error ?? "Corpus eval failed.");
            setStatus(null);
          } else {
            setStatus(
              `Corpus alignment eval running... ${response.value.completed_rows}/${response.value.total_rows}`,
            );
          }
        } catch (e) {
          if (!cancelled) {
            setError(e instanceof Error ? e.message : String(e));
            setStatus(null);
          }
        }
      })();
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [job, wsUrl]);

  const selectedRow = useMemo(() => {
    if (!result) return null;
    return (
      result.rows.find((row) => row.prompt_id === selectedPromptId) ??
      result.rows[0] ??
      null
    );
  }, [result, selectedPromptId]);

  const overallMean = useMemo(() => {
    if (!result || result.rows.length === 0) return null;
    const values = result.rows
      .map((row) => row.utterance_feature_similarity)
      .filter((value): value is number => value != null);
    if (values.length === 0) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }, [result]);

  const startRandomFiveRun = useCallback(() => {
    setBucket("");
    setLimit(5);
    setShowRunComposer(false);
    void runEval({ limit: 5, bucket: "", randomize: true });
  }, [runEval]);

  const startConfiguredRun = useCallback(() => {
    setShowRunComposer(false);
    void runEval({ limit, bucket, randomize: false });
  }, [bucket, limit, runEval]);

  return (
    <div
      className="prototype-lab"
      style={{
        width: "min(1500px, 100%)",
        margin: "0 auto",
        gap: "0.75rem",
      }}
    >
      {result ? (
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(250px, 310px) minmax(0, 1fr)",
            gap: "0.75rem",
            alignItems: "start",
          }}
        >
          <section
            className="prototype-card"
            style={{
              position: "sticky",
              top: "0.75rem",
              maxHeight: "calc(100vh - 1.5rem)",
              overflow: "hidden",
            }}
          >
            <div className="corpus-eval-sidebar-toolbar">
              <div className="corpus-eval-sidebar-header">
                <strong>Examples</strong>
                {job ? (
                  <span className="mini-badge">
                    {job.completed_rows}/{job.total_rows}
                  </span>
                ) : null}
                <span className="mini-badge">{result.rows.length} rows</span>
              </div>
              <div className="corpus-eval-sidebar-actions">
                <button
                  type="button"
                  className="mini-badge"
                  style={{ cursor: "pointer" }}
                  onClick={startRandomFiveRun}
                >
                  ↻ Random 5-run
                </button>
                <button
                  type="button"
                  className="mini-badge"
                  style={{ cursor: "pointer" }}
                  onClick={() => setShowRunComposer((value) => !value)}
                >
                  + New run
                </button>
              </div>
            </div>

            <div className="prototype-stack corpus-eval-sidebar-stack" style={{ gap: "0.5rem" }}>
              {(status || error || job) && (
                <div className="notice-row corpus-eval-status-row" style={{ margin: 0 }}>
                  {status ? <span className="status-pill">{status}</span> : null}
                  {error ? <span className="error-pill">{error}</span> : null}
                </div>
              )}
              {showRunComposer ? (
                <div
                  className="prototype-card prototype-card-tight"
                  style={{
                    gap: "0.65rem",
                    padding: "0.75rem",
                    background: "var(--bg-elevated)",
                  }}
                >
                  <div className="eyebrow">New run</div>
                  <div className="numeric-row" style={{ gridTemplateColumns: "1fr", gap: "0.5rem" }}>
                    <label>
                      <span>limit</span>
                      <input
                        type="number"
                        min={1}
                        max={200}
                        value={limit}
                        onChange={(e) => setLimit(Number(e.target.value) || 1)}
                      />
                    </label>
                    <label>
                      <span>bucket filter</span>
                      <input
                        value={bucket}
                        onChange={(e) => setBucket(e.target.value)}
                        placeholder="optional"
                      />
                    </label>
                  </div>
                  <div className="control-actions" style={{ justifyContent: "flex-start" }}>
                    <button className="primary" onClick={startConfiguredRun}>
                      Start run
                    </button>
                    <button type="button" onClick={() => setShowRunComposer(false)}>
                      Cancel
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="corpus-eval-summary-strip">
              <div className="corpus-eval-summary-chip corpus-eval-summary-chip-primary">
                <span className="eyebrow">Overall phonetic match</span>
                <strong>{formatPercentMetric(overallMean)}</strong>
              </div>
              {result.bucket_summaries.map((summary) => (
                <div key={summary.bucket} className="corpus-eval-summary-chip">
                  <span className="eyebrow">{bucketLabel(summary.bucket)}</span>
                  <strong>{formatPercentMetric(summary.utterance_feature_similarity_mean)}</strong>
                </div>
              ))}
            </div>

            <div
              className="corpus-eval-row-list"
              style={{
                display: "grid",
                gap: "0.45rem",
                overflow: "auto",
                paddingRight: "0.1rem",
              }}
            >
              {result.rows.map((row) => (
                <RowCard
                  key={`${row.prompt_id}:${row.take}`}
                  row={row}
                  selected={row.prompt_id === selectedRow?.prompt_id}
                  onSelect={() => setSelectedPromptId(row.prompt_id)}
                />
              ))}
            </div>
          </section>

          <section
            className="prototype-card"
            style={{
              gap: "0.65rem",
              minHeight: "calc(100vh - 8rem)",
            }}
          >
            {selectedRow ? (
              <>
                <div className="corpus-eval-case-toolbar">
                  <strong className="corpus-eval-case-title">{selectedRow.prompt_text}</strong>
                  <div className="corpus-eval-case-chips">
                    <span className="badge">{selectedRow.term}</span>
                    <span className="mini-badge">{bucketLabel(selectedRow.bucket)}</span>
                    <span className="mini-badge">take {selectedRow.take}</span>
                    <span className="mini-badge">{selectedRow.prompt_id}</span>
                  </div>
                </div>

                <div className="token-row muted corpus-eval-case-meta">
                  <span>Heard: {selectedRow.asr_transcript}</span>
                  <span>{formatPercentMetric(selectedRow.utterance_feature_similarity)} match</span>
                  {selectedRow.prompt_notes ? <span>Note: {selectedRow.prompt_notes}</span> : null}
                </div>

                {selectedRow.trace ? (
                  <PhoneticRescuePanel
                    trace={toPhoneticTrace(selectedRow.trace)}
                    wsUrl={wsUrl}
                    sourceAudioPath={selectedRow.wav_path}
                  />
                ) : (
                  <div className="prototype-empty">
                    {selectedRow.error ?? "No phonetic trace available."}
                  </div>
                )}
              </>
            ) : (
              <div className="prototype-empty">Pick a row from the left.</div>
            )}
          </section>
        </section>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <header className="panel-header-row">
            <div>
              <strong>Corpus Alignment Eval</strong>
              <span>Starting a random 5-case run.</span>
            </div>
          </header>
          <div className="prototype-empty">
            {error ?? status ?? "Starting random 5-case run..."}
          </div>
        </section>
      )}
    </div>
  );
}
