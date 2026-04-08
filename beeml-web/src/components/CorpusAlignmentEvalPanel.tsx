import { useCallback, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  CorpusAlignmentEvalResult,
  CorpusAlignmentEvalRow,
  TranscribePhoneticTrace as RpcTranscribePhoneticTrace,
} from "../beeml.generated";
import { PhoneticRescuePanel } from "./PhoneticRescuePanel";
import type { PhoneticRescueTrace } from "../types";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
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
      transcriptSimilarity: span.transcript_similarity,
      transcriptFeatureSimilarity: span.transcript_feature_similarity,
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
    <button
      onClick={onSelect}
      className="failure-card"
      style={{
        gap: "0.55rem",
        textAlign: "left",
        border: selected ? "1px solid var(--accent)" : "1px solid var(--border)",
        background: selected ? "var(--bg-subtle)" : "var(--bg-elevated)",
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
      {row.error ? <div className="error-pill">{row.error}</div> : null}
    </button>
  );
}

export function CorpusAlignmentEvalPanel({ wsUrl }: { wsUrl: string }) {
  const [limit, setLimit] = useState(100);
  const [bucket, setBucket] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CorpusAlignmentEvalResult | null>(null);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);

  const runEval = useCallback(async () => {
    try {
      setStatus("Running corpus alignment eval...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const response = await client.runCorpusAlignmentEval({
        limit,
        bucket: bucket.trim() ? bucket.trim() : null,
      });
      if (!response.ok) throw new Error(response.error);
      setResult(response.value);
      setSelectedPromptId(response.value.rows[0]?.prompt_id ?? null);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [bucket, limit, wsUrl]);

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
              Run Eval
            </button>
          </div>
        </div>

        {(status || error) && (
          <div className="notice-row">
            {status ? <span className="status-pill">{status}</span> : null}
            {error ? <span className="error-pill">{error}</span> : null}
          </div>
        )}
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
