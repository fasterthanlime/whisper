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
  const trace = row.trace ? toPhoneticTrace(row.trace) : null;
  const previewOps = lanePreviewOps(trace);
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
        gap: "0.55rem",
        textAlign: "left",
        border: selected ? "1px solid var(--accent)" : "1px solid var(--border)",
        background: selected ? "var(--bg-subtle)" : "var(--bg-elevated)",
        cursor: "pointer",
        userSelect: "text",
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
        <span className="failure-pill">spans {row.positive_span_count}</span>
        <span className="failure-pill">
          worst {formatMetric(row.worst_span_feature_similarity)}
        </span>
        <span className="failure-pill">best Δ {formatMetric(row.best_span_delta)}</span>
      </div>
      <div className="token-row muted">ASR: {row.asr_transcript}</div>
      {trace ? (
        <>
          <div className="token-row muted">ZIPA raw: {formatTokens(trace.utteranceZipaRaw)}</div>
          <div className="token-row muted">
            ZIPA norm: {formatTokens(trace.utteranceZipaNormalized)}
          </div>
        </>
      ) : null}
      {previewOps.length > 0 ? (
        <div style={{ marginTop: "0.15rem" }}>
          <AlignmentView ops={previewOps} transcriptLabel="Transcript" zipaLabel="ZIPA" />
        </div>
      ) : null}
      {row.error ? <div className="error-pill">{row.error}</div> : null}
    </div>
  );
}

function lanePreviewOps(trace: PhoneticRescueTrace | null): PhoneticAlignmentOp[] {
  if (!trace) return [];
  const worstSpan = trace.spans.reduce<PhoneticRescueSpan | null>((worst, span) => {
    if (!worst) return span;
    const left = span.transcriptFeatureSimilarity ?? Infinity;
    const right = worst.transcriptFeatureSimilarity ?? Infinity;
    return left < right ? span : worst;
  }, null);
  if (worstSpan && worstSpan.alignment.length > 0) {
    return cropOps(worstSpan.alignment, 18);
  }
  return cropOps(trace.utteranceAlignment, 18);
}

function cropOps(ops: PhoneticAlignmentOp[], targetWidth: number): PhoneticAlignmentOp[] {
  if (ops.length <= targetWidth) return ops;
  const mismatchIndex = ops.findIndex((op) => op.kind !== "Match");
  if (mismatchIndex < 0) {
    return ops.slice(0, targetWidth);
  }
  const left = Math.max(0, mismatchIndex - Math.floor(targetWidth / 2));
  const right = Math.min(ops.length, left + targetWidth);
  return ops.slice(Math.max(0, right - targetWidth), right);
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
    <div className="prototype-lab prototype-stack judge-eval-layout">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Corpus Alignment Eval</strong>
            <span>Run the current ASR + ZIPA DP path over the recorded targeted corpus.</span>
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
        <>
          <section className="prototype-card eval-hero-card">
            <div className="eval-hero-main">
              <span className="eyebrow">Corpus utterance feature similarity</span>
              <div className="eval-hero-number">{formatMetric(overallMean)}</div>
              <div className="eval-hero-caption">
                Worst rows are sorted to the top for inspection.
              </div>
            </div>
            <div className="eval-stat-grid">
              {result.bucket_summaries.map((summary) => (
                <div className="eval-stat-card" key={summary.bucket}>
                  <span className="eval-stat-label">{summary.bucket}</span>
                  <strong>{formatMetric(summary.utterance_feature_similarity_mean)}</strong>
                  <span className="muted">{summary.rows} rows</span>
                </div>
              ))}
            </div>
          </section>

          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Rows</strong>
                <span>Worst utterance-level feature matches first.</span>
              </div>
              <span className="badge">{result.rows.length}</span>
            </header>
            <div style={{ display: "grid", gap: "0.75rem" }}>
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

          {selectedRow ? (
            <section className="prototype-card" style={{ gap: "0.75rem" }}>
              <header className="panel-header-row">
                <div>
                  <strong>Selected Row</strong>
                  <span>
                    {selectedRow.prompt_id} · {selectedRow.bucket} · take {selectedRow.take}
                  </span>
                </div>
                <span className="badge">{selectedRow.term}</span>
              </header>
              <div className="prototype-summary">
                <span>prompt {selectedRow.prompt_text}</span>
              </div>
              {selectedRow.prompt_notes ? (
                <div className="prototype-empty" style={{ textAlign: "left" }}>
                  {selectedRow.prompt_notes}
                </div>
              ) : null}
              <div className="prototype-summary">
                <span>ASR {selectedRow.asr_transcript}</span>
                <span>{selectedRow.wav_path}</span>
              </div>
              {selectedRow.trace ? (
                <PhoneticRescuePanel trace={toPhoneticTrace(selectedRow.trace)} />
              ) : (
                <div className="prototype-empty">
                  {selectedRow.error ?? "No phonetic trace available."}
                </div>
              )}
            </section>
          ) : null}
        </>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">No corpus eval run yet.</div>
        </section>
      )}
    </div>
  );
}
