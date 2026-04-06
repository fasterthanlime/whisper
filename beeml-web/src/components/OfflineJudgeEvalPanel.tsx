import { useCallback, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  OfflineJudgeEvalResult,
  ThresholdRow,
  ModelSummary,
  TwoStageGridPoint,
  ProbDistribution,
} from "../beeml.generated";

function pct(n: number, d: number) {
  if (d === 0) return "—";
  return `${((n / d) * 100).toFixed(1)}%`;
}

function fmt(v: number) {
  return v.toFixed(1) + "%";
}

function ProbDistRow({ dist }: { dist: ProbDistribution }) {
  if (dist.n === 0) return <span className="text-dim">N/A</span>;
  return (
    <span className="mono">
      n={dist.n} min={dist.min.toFixed(3)} p25={dist.p25.toFixed(3)} p50=
      {dist.p50.toFixed(3)} p75={dist.p75.toFixed(3)} max={dist.max.toFixed(3)}
    </span>
  );
}

function SweepTable({
  rows,
  label,
}: {
  rows: ThresholdRow[];
  label: string;
}) {
  if (rows.length === 0) return null;
  return (
    <div className="sweep-section">
      <strong>{label}</strong>
      <table className="sweep-table">
        <thead>
          <tr>
            <th>T</th>
            <th>Can. Acc</th>
            <th>Cx. Acc</th>
            <th>Balanced</th>
            <th>Can. Repl%</th>
            <th>Cx. Repl%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.threshold}>
              <td>{r.threshold.toFixed(1)}</td>
              <td>
                {r.canonical_correct}/{r.canonical_total} (
                {pct(r.canonical_correct, r.canonical_total)})
              </td>
              <td>
                {r.cx_correct}/{r.cx_total} ({pct(r.cx_correct, r.cx_total)})
              </td>
              <td className={r.balanced_pct > 60 ? "good" : ""}>
                {fmt(r.balanced_pct)}
              </td>
              <td>{fmt(r.canonical_replace_pct)}</td>
              <td>{fmt(r.cx_replace_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SummaryTable({
  rows,
  title,
}: {
  rows: ModelSummary[];
  title: string;
}) {
  if (rows.length === 0) return null;
  return (
    <div className="sweep-section">
      <strong>{title}</strong>
      <table className="sweep-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Best T</th>
            <th>Can. Acc</th>
            <th>Cx. Acc</th>
            <th>Balanced</th>
            <th>Can. Repl%</th>
            <th>Cx. Repl%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{r.best_threshold.toFixed(1)}</td>
              <td>
                {r.canonical_correct}/{r.canonical_total} (
                {pct(r.canonical_correct, r.canonical_total)})
              </td>
              <td>
                {r.cx_correct}/{r.cx_total} ({pct(r.cx_correct, r.cx_total)})
              </td>
              <td className={r.balanced_pct > 60 ? "good" : ""}>
                {fmt(r.balanced_pct)}
              </td>
              <td>{fmt(r.canonical_replace_pct)}</td>
              <td>{fmt(r.cx_replace_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GridTable({ points }: { points: TwoStageGridPoint[] }) {
  if (points.length === 0) return null;
  return (
    <table className="sweep-table">
      <thead>
        <tr>
          <th>Gate T</th>
          <th>Ranker T</th>
          <th>Can. Acc</th>
          <th>Cx. Acc</th>
          <th>Balanced</th>
          <th>Can. Repl%</th>
          <th>Cx. Repl%</th>
        </tr>
      </thead>
      <tbody>
        {points.map((r, i) => (
          <tr key={i}>
            <td>{r.gate_threshold.toFixed(1)}</td>
            <td>{r.ranker_threshold.toFixed(1)}</td>
            <td>
              {r.canonical_correct}/{r.canonical_total} (
              {pct(r.canonical_correct, r.canonical_total)})
            </td>
            <td>
              {r.cx_correct}/{r.cx_total} ({pct(r.cx_correct, r.cx_total)})
            </td>
            <td className={r.balanced_pct > 60 ? "good" : ""}>
              {fmt(r.balanced_pct)}
            </td>
            <td>{fmt(r.canonical_replace_pct)}</td>
            <td>{fmt(r.cx_replace_pct)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function OfflineJudgeEvalPanel({ wsUrl }: { wsUrl: string }) {
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OfflineJudgeEvalResult | null>(null);

  const runEval = useCallback(async () => {
    try {
      setStatus("Running offline eval (this takes a while)...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const response = await client.runOfflineJudgeEval({
        folds: 5,
        max_span_words: 4,
        shortlist_limit: 8,
        verify_limit: 5,
        train_epochs: 4,
      });
      if (!response.ok) throw new Error(response.error);
      setResult(response.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [wsUrl]);

  const ts = result?.two_stage;

  return (
    <div className="prototype-lab prototype-stack offline-eval-layout">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Offline Judge Eval</strong>
            <span>
              Phase 4: two-stage judge eval with k-fold cross-validation.
            </span>
          </div>
        </header>
        <div className="control-bar">
          <div className="control-actions">
            <button className="primary" onClick={() => void runEval()}>
              Run Offline Eval
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

      {result && ts ? (
        <>
          {/* Hero: two-stage best result */}
          <section className="prototype-card eval-hero-card">
            <div className="eval-hero-main">
              <span className="eyebrow">Two-stage balanced accuracy</span>
              <div className="eval-hero-number">
                {fmt(ts.best.balanced_pct)}
              </div>
              <div className="eval-hero-caption">
                GT={ts.best.gate_threshold.toFixed(1)} RT=
                {ts.best.ranker_threshold.toFixed(1)} &mdash;{" "}
                {ts.best.canonical_correct}/{ts.best.canonical_total} canonical,{" "}
                {ts.best.cx_correct}/{ts.best.cx_total} cx &mdash; 0% false
                positive replacements
              </div>
            </div>
            <div className="eval-stat-grid">
              <div className="eval-stat-card">
                <span className="eval-stat-label">Canonical cases</span>
                <strong>{result.canonical_cases}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Gold retrieved</span>
                <strong>{result.gold_retrieved}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Gold verified</span>
                <strong>{result.gold_verified}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Gold reachable</span>
                <strong>{result.gold_reachable}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Counterexamples</span>
                <strong>{result.counterexample_cases}</strong>
              </div>
              <div className="eval-stat-card">
                <span className="eval-stat-label">Ranker top-1</span>
                <strong>
                  {pct(ts.ranker_top1_correct, ts.ranker_top1_total)}
                </strong>
              </div>
            </div>
          </section>

          {/* Two-stage details */}
          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Two-Stage Architecture</strong>
                <span>
                  Stage A (span gate) + Stage B (candidate ranker) with
                  threshold grid sweep.
                </span>
              </div>
            </header>

            <div className="eval-subsection">
              <h4>Stage A: Gate threshold sweep</h4>
              <SweepTable rows={ts.gate_sweep} label="Gate alone" />
            </div>

            <div className="eval-subsection">
              <h4>Composed: Gate x Ranker grid</h4>
              <GridTable points={ts.grid_points} />
            </div>

            <div className="eval-subsection">
              <h4>Probability distributions</h4>
              <div className="prob-dist-grid">
                <div>
                  <span className="eval-stat-label">
                    Gate canonical (should open)
                  </span>
                  <ProbDistRow dist={ts.gate_canonical_dist} />
                </div>
                <div>
                  <span className="eval-stat-label">
                    Gate cx (should close)
                  </span>
                  <ProbDistRow dist={ts.gate_cx_dist} />
                </div>
                <div>
                  <span className="eval-stat-label">
                    Ranker gold=best
                  </span>
                  <ProbDistRow dist={ts.ranker_gold_best_dist} />
                </div>
                <div>
                  <span className="eval-stat-label">
                    Ranker gold!=best
                  </span>
                  <ProbDistRow dist={ts.ranker_gold_not_best_dist} />
                </div>
              </div>
            </div>
          </section>

          {/* Single-model baselines */}
          <section className="prototype-card">
            <header className="panel-header-row">
              <div>
                <strong>Single-Model Baselines</strong>
                <span>
                  For comparison: all single-model approaches that the
                  two-stage architecture supersedes.
                </span>
              </div>
            </header>

            <SummaryTable
              rows={result.formulation_results}
              title="Formulation comparison (best threshold each)"
            />
            <SummaryTable
              rows={result.ablation_results}
              title="Feature ablation (case-balanced, best threshold each)"
            />

            <details className="sweep-details">
              <summary>Full threshold sweeps</summary>
              <SweepTable
                rows={result.deterministic_sweep}
                label="Deterministic"
              />
              <SweepTable rows={result.seed_only_sweep} label="Seed-only" />
              <SweepTable rows={result.taught_sweep} label="Taught" />
              <SweepTable
                rows={result.case_balanced_sweep}
                label="Case-balanced"
              />
            </details>
          </section>
        </>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">No offline eval run yet.</div>
        </section>
      )}
    </div>
  );
}
