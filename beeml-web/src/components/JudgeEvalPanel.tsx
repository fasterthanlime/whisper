import { useCallback, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type { RetrievalPrototypeEvalResult } from "../beeml.generated";

function formatPercent(value: number) {
  return `${Math.round(value)}%`;
}

export function JudgeEvalPanel({
  wsUrl,
}: {
  wsUrl: string;
}) {
  const [limit, setLimit] = useState(120);
  const [maxSpanWords, setMaxSpanWords] = useState(4);
  const [shortlistLimit, setShortlistLimit] = useState(8);
  const [verifyLimit, setVerifyLimit] = useState(5);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RetrievalPrototypeEvalResult | null>(null);

  const runEval = useCallback(async () => {
    try {
      setStatus("Running eval...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const response = await client.runRetrievalPrototypeEval({
        limit,
        max_span_words: maxSpanWords,
        shortlist_limit: shortlistLimit,
        verify_limit: verifyLimit,
      });
      if (!response.ok) throw new Error(response.error);
      setResult(response.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [limit, maxSpanWords, shortlistLimit, verifyLimit, wsUrl]);

  const summary = useMemo(() => {
    if (!result || result.evaluated_cases === 0) return null;
    return {
      judgeAccuracy: (result.judge_correct / result.evaluated_cases) * 100,
      retrievalTop1: (result.top1_hits / result.evaluated_cases) * 100,
      retrievalTop3: (result.top3_hits / result.evaluated_cases) * 100,
      retrievalTop10: (result.top10_hits / result.evaluated_cases) * 100,
      replaceAccuracy: (result.judge_replace_correct / result.evaluated_cases) * 100,
      abstainAccuracy: (result.judge_abstain_correct / result.evaluated_cases) * 100,
    };
  }, [result]);

  return (
    <div className="prototype-lab prototype-stack judge-eval-layout">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Judge Eval</strong>
            <span>Run retrieval and judge summaries over the current teaching set.</span>
          </div>
          {result ? <span className="badge">{result.evaluated_cases} cases</span> : null}
        </header>

        <div className="control-bar">
          <div className="numeric-row">
            <label>
              <span>limit</span>
              <input type="number" min={1} max={500} value={limit} onChange={(e) => setLimit(Number(e.target.value) || 1)} />
            </label>
            <label>
              <span>max span words</span>
              <input type="number" min={1} max={8} value={maxSpanWords} onChange={(e) => setMaxSpanWords(Number(e.target.value) || 1)} />
            </label>
            <label>
              <span>shortlist</span>
              <input type="number" min={1} max={20} value={shortlistLimit} onChange={(e) => setShortlistLimit(Number(e.target.value) || 1)} />
            </label>
            <label>
              <span>verify</span>
              <input type="number" min={1} max={20} value={verifyLimit} onChange={(e) => setVerifyLimit(Number(e.target.value) || 1)} />
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

      {result && summary ? (
        <>
          <section className="prototype-card eval-hero-card">
            <div className="eval-hero-main">
              <span className="eyebrow">Judge accuracy</span>
              <div className="eval-hero-number">{formatPercent(summary.judgeAccuracy)}</div>
              <div className="eval-hero-caption">
                {result.judge_correct} correct out of {result.evaluated_cases} cases
              </div>
            </div>
            <div className="eval-stat-grid">
              <div className="eval-stat-card">
                <span className="eval-stat-label">Retrieval top 1</span>
                <strong>{formatPercent(summary.retrievalTop1)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Retrieval top 3</span>
                <strong>{formatPercent(summary.retrievalTop3)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Retrieval top 10</span>
                <strong>{formatPercent(summary.retrievalTop10)}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Replace correct</span>
                <strong>{result.judge_replace_correct}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Abstain correct</span>
                <strong>{result.judge_abstain_correct}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Judge failures</span>
                <strong>{result.judge_failures.length}</strong>
              </div>
            </div>
          </section>

          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Judge Failures</strong>
                <span>Most useful errors to inspect right now.</span>
              </div>
              <span className="badge">{result.judge_failures.length}</span>
            </header>
            {result.judge_failures.length > 0 ? (
              <div className="failure-grid">
                {result.judge_failures.slice(0, 18).map((failure) => (
                  <article key={failure.case_id} className="failure-card">
                    <div className="failure-topline">
                      <span className="mini-badge">{failure.suite}</span>
                      <span className="failure-score">{failure.chosen_probability.toFixed(3)}</span>
                    </div>
                    <div className="failure-transcript">{failure.transcript}</div>
                    <div className="failure-pills">
                      <span className="failure-pill failure-pill-expected">expected {failure.expected_action}</span>
                      <span className="failure-pill failure-pill-chosen">chose {failure.chosen_action}</span>
                      <span className="failure-pill">span {failure.chosen_span_text}</span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="prototype-empty">No judge failures in this run.</div>
            )}
          </section>

          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Retrieval Misses</strong>
                <span>Cases where the right term never made the shortlist.</span>
              </div>
              <span className="badge">{result.misses.length}</span>
            </header>
            {result.misses.length > 0 ? (
              <div className="failure-grid failure-grid-compact">
                {result.misses.slice(0, 12).map((miss) => (
                  <article key={`${miss.suite}:${miss.recording_id}`} className="failure-card miss-card">
                    <div className="failure-topline">
                      <span className="mini-badge">{miss.suite}</span>
                      <span className="mini-badge">{miss.term}</span>
                    </div>
                    <div className="failure-transcript">{miss.transcript}</div>
                    <div className="failure-pills">
                      <span className="failure-pill">best span {miss.best_span_text || "none"}</span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="prototype-empty">No retrieval misses in this run.</div>
            )}
          </section>
        </>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">No eval run yet.</div>
        </section>
      )}
    </div>
  );
}
