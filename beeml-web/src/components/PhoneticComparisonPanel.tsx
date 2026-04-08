import { useCallback, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type { PhoneticComparisonResult, PhoneticComparisonRow } from "../beeml.generated";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
}

function RowCard({ row }: { row: PhoneticComparisonRow }) {
  return (
    <article className="failure-card" style={{ gap: "0.55rem" }}>
      <div className="failure-topline">
        <span className="mini-badge">{row.term}</span>
        <span className="failure-score">
          nf {formatMetric(row.normalized_feature_similarity)}
        </span>
      </div>
      <div className="failure-transcript">{row.text}</div>
      <div className="failure-pills">
        <span className="failure-pill">raw {formatMetric(row.raw_similarity)}</span>
        <span className="failure-pill">reduced {formatMetric(row.reduced_similarity)}</span>
        <span className="failure-pill">normalized {formatMetric(row.normalized_similarity)}</span>
        <span className="failure-pill">
          feature {formatMetric(row.feature_similarity)}
        </span>
      </div>
      <div className="token-row" style={{ fontFamily: "'Manuale IPA', serif" }}>
        ZIPA raw: {row.zipa_raw.join(" ")}
      </div>
      <div className="token-row muted" style={{ fontFamily: "'Manuale IPA', serif" }}>
        eSpeak raw: {row.espeak_raw.join(" ")}
      </div>
      <div className="token-row" style={{ fontFamily: "'Manuale IPA', serif" }}>
        ZIPA norm: {row.zipa_normalized.join(" ")}
      </div>
      <div className="token-row muted" style={{ fontFamily: "'Manuale IPA', serif" }}>
        eSpeak norm: {row.espeak_normalized.join(" ")}
      </div>
    </article>
  );
}

export function PhoneticComparisonPanel({ wsUrl }: { wsUrl: string }) {
  const [limit, setLimit] = useState(40);
  const [term, setTerm] = useState("");
  const [useTranscript, setUseTranscript] = useState(true);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PhoneticComparisonResult | null>(null);

  const runComparison = useCallback(async () => {
    try {
      setStatus("Comparing ZIPA and eSpeak...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const response = await client.runPhoneticComparison({
        limit,
        term: term.trim() ? term.trim() : null,
        use_transcript: useTranscript,
      });
      if (!response.ok) throw new Error(response.error);
      setResult(response.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [limit, term, useTranscript, wsUrl]);

  return (
    <div className="prototype-lab prototype-stack judge-eval-layout">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Phonetic Comparison</strong>
            <span>Compare ZIPA output against eSpeak on phonetic-seed recordings.</span>
          </div>
          {result ? <span className="badge">{result.summary.rows} rows</span> : null}
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
              <span>term filter</span>
              <input value={term} onChange={(e) => setTerm(e.target.value)} placeholder="optional" />
            </label>
            <label>
              <span>compare text</span>
              <select
                value={useTranscript ? "transcript" : "source"}
                onChange={(e) => setUseTranscript(e.target.value === "transcript")}
              >
                <option value="transcript">transcript</option>
                <option value="source">source text</option>
              </select>
            </label>
          </div>
          <div className="control-actions">
            <button className="primary" onClick={() => void runComparison()}>
              Run Comparison
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
              <span className="eyebrow">Normalized feature similarity</span>
              <div className="eval-hero-number">
                {formatMetric(result.summary.normalized_feature_similarity_mean)}
              </div>
              <div className="eval-hero-caption">
                normalized token {formatMetric(result.summary.normalized_similarity_mean)} · raw feature{" "}
                {formatMetric(result.summary.feature_similarity_mean)}
              </div>
            </div>
            <div className="eval-stat-grid">
              <div className="eval-stat-card">
                <span className="eval-stat-label">Raw similarity</span>
                <strong>{formatMetric(result.summary.raw_similarity_mean)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Reduced similarity</span>
                <strong>{formatMetric(result.summary.reduced_similarity_mean)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Normalized similarity</span>
                <strong>{formatMetric(result.summary.normalized_similarity_mean)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Feature similarity</span>
                <strong>{formatMetric(result.summary.feature_similarity_mean)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Reduced exact</span>
                <strong>{result.summary.reduced_exact}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Normalized exact</span>
                <strong>{result.summary.normalized_exact}</strong>
              </div>
            </div>
          </section>

          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Rows</strong>
                <span>Sorted in dataset order for now; worst-case sorting can come next.</span>
              </div>
              <span className="badge">{result.rows.length}</span>
            </header>
            <div className="failure-grid">
              {result.rows.map((row) => (
                <RowCard
                  key={`${row.term}:${row.wav_path}:${row.text}`}
                  row={row}
                />
              ))}
            </div>
          </section>
        </>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">No comparison run yet.</div>
        </section>
      )}
    </div>
  );
}
