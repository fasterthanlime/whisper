import { useCallback, useEffect, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  CorpusAlignmentEvalJob,
  CorpusAlignmentEvalResult,
  CorpusAlignmentEvalRow,
  TranscribePhoneticTrace as RpcTranscribePhoneticTrace,
} from "../beeml.generated";
import { AlignmentView, PhoneticRescuePanel } from "./PhoneticRescuePanel";
import type { PhoneticRescueTrace } from "../types";
import type { PhoneticAlignmentOp, PhoneticRescueSpan } from "../types";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
}

function formatTokens(tokens: string[]) {
  return tokens.length > 0 ? tokens.join(" ") : "∅";
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
      className="failure-card"
      style={{
        gap: "0.45rem",
        textAlign: "left",
        border: selected
          ? "1px solid color-mix(in srgb, var(--accent) 70%, var(--border))"
          : "1px solid var(--border)",
        background: selected
          ? "linear-gradient(180deg, color-mix(in srgb, var(--bg-subtle) 88%, transparent), var(--bg-elevated))"
          : "var(--bg-elevated)",
        cursor: "pointer",
        userSelect: "text",
        padding: "0.65rem 0.7rem",
      }}
    >
      <div className="failure-topline">
        <span className="mini-badge">{row.ordinal}</span>
        <span className="mini-badge">{row.bucket}</span>
        <span className="mini-badge">{row.term}</span>
        <span className="failure-score">uf {formatMetric(row.utterance_feature_similarity)}</span>
      </div>
      <div className="failure-transcript">{row.prompt_text}</div>
      <div className="failure-pills">
        <span className="failure-pill">take {row.take}</span>
        <span className="failure-pill">raw {formatMetric(row.utterance_similarity)}</span>
        <span className="failure-pill">contentful {row.contentful_span_count}</span>
        <span className="failure-pill">eligible {row.rescue_eligible_span_count}</span>
      </div>
      <div className="token-row muted" style={{ lineHeight: 1.45 }}>
        ASR: {row.asr_transcript}
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

  const runEval = useCallback(async () => {
    try {
      setStatus("Starting corpus alignment eval...");
      setError(null);
      setResult(null);
      const client = await connectBeeMl(wsUrl);
      const response = await client.startCorpusAlignmentEvalJob({
        limit,
        bucket: bucket.trim() ? bucket.trim() : null,
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

  return (
    <div
      className="prototype-lab"
      style={{
        width: "min(1500px, 100%)",
        margin: "0 auto",
        gap: "0.75rem",
      }}
    >
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Corpus Alignment Eval</strong>
            <span>Use the left rail to pick a case. Use the right workspace to inspect utterance alignment and per-word IPA.</span>
          </div>
          {result ? <span className="badge">{result.rows.length} rows</span> : null}
        </header>

        <div className="control-bar">
          <div className="numeric-row">
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
            <label style={{ minWidth: 180 }}>
              <span>bucket filter</span>
              <input
                value={bucket}
                onChange={(e) => setBucket(e.target.value)}
                placeholder="optional"
              />
            </label>
          </div>
          <div className="control-actions">
            <button className="primary" onClick={() => void runEval()}>
              {job?.status.tag === "Running" ? "Re-run Eval" : "Start Eval"}
            </button>
          </div>
        </div>

        {(status || error) && (
          <div className="notice-row">
            {status ? <span className="status-pill">{status}</span> : null}
            {error ? <span className="error-pill">{error}</span> : null}
          </div>
        )}
        {job ? (
          <div className="prototype-summary">
            <span>job {job.job_id}</span>
            <span>status {job.status.tag.toLowerCase()}</span>
            <span>
              progress {job.completed_rows}/{job.total_rows}
            </span>
          </div>
        ) : null}
      </section>

      {result ? (
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(280px, 360px) minmax(0, 1fr)",
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
            <header className="panel-header-row panel-header-compact">
              <div>
                <strong>Cases</strong>
                <span>Pick a recording to inspect.</span>
              </div>
              <span className="badge">{result.rows.length}</span>
            </header>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                gap: "0.45rem",
              }}
            >
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "6px",
                  padding: "0.55rem 0.6rem",
                  background: "var(--bg-elevated)",
                }}
              >
                <div className="eyebrow">Overall UF</div>
                <div style={{ fontSize: "1.9rem", fontWeight: 700, lineHeight: 1.05 }}>
                  {formatMetric(overallMean)}
                </div>
              </div>
              {result.bucket_summaries.map((summary) => (
                <div
                  key={summary.bucket}
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    padding: "0.55rem 0.6rem",
                    background: "var(--bg-elevated)",
                  }}
                >
                  <div className="eyebrow">{summary.bucket}</div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 700, lineHeight: 1.1 }}>
                    {formatMetric(summary.utterance_feature_similarity_mean)}
                  </div>
                  <div className="token-row muted">{summary.rows} rows</div>
                </div>
              ))}
            </div>

            <div
              style={{
                display: "grid",
                gap: "0.6rem",
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
              gap: "0.85rem",
              minHeight: "calc(100vh - 8rem)",
              border: "1px solid color-mix(in srgb, var(--accent) 35%, var(--border))",
            }}
          >
            {selectedRow ? (
              <>
                <header className="panel-header-row">
                  <div>
                    <strong>{selectedRow.prompt_text}</strong>
                    <span>
                      {selectedRow.prompt_id} · {selectedRow.bucket} · take {selectedRow.take}
                    </span>
                  </div>
                  <span className="badge">{selectedRow.term}</span>
                </header>

                <div className="prototype-summary">
                  <span>UF {formatMetric(selectedRow.utterance_feature_similarity)}</span>
                  <span>raw {formatMetric(selectedRow.utterance_similarity)}</span>
                  <span>ASR {selectedRow.asr_transcript}</span>
                </div>

                {selectedRow.prompt_notes ? (
                  <div
                    className="prototype-empty"
                    style={{
                      textAlign: "left",
                      padding: "0.6rem 0.75rem",
                      border: "1px solid color-mix(in srgb, var(--accent) 18%, var(--border))",
                      background: "color-mix(in srgb, var(--bg-subtle) 78%, transparent)",
                    }}
                  >
                    {selectedRow.prompt_notes}
                  </div>
                ) : null}

                {selectedRow.trace ? (
                  <PhoneticRescuePanel trace={toPhoneticTrace(selectedRow.trace)} />
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
          <div className="prototype-empty">No corpus eval run yet.</div>
        </section>
      )}
    </div>
  );
}
