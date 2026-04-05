import { memo, useEffect, useRef, useState } from "react";
import { channel } from "@bearcove/vox-core";
import { connectBeeMl } from "../beeml.generated";
import type {
  RetrievalPrototypeEvalProgress,
  RetrievalPrototypeEvalResult,
} from "../beeml.generated";

interface HistoryPoint {
  teachCount: number;
  judgeCorrect: number;
  evaluatedCases: number;
  pct: number;
}

function SparklineGraph({ history }: { history: HistoryPoint[] }) {
  if (history.length < 2) return null;

  const w = 280;
  const h = 64;
  const pad = { top: 4, bottom: 14, left: 2, right: 2 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const minPct = Math.min(...history.map(p => p.pct));
  const maxPct = Math.max(...history.map(p => p.pct));
  const range = Math.max(maxPct - minPct, 5);
  const yMin = Math.max(0, minPct - 2);
  const yMax = Math.min(100, maxPct + 2);
  const yRange = Math.max(yMax - yMin, 5);

  const points = history.map((p, i) => {
    const x = pad.left + (i / (history.length - 1)) * plotW;
    const y = pad.top + plotH - ((p.pct - yMin) / yRange) * plotH;
    return { x, y, p };
  });

  const linePath = points.map((pt, i) => `${i === 0 ? "M" : "L"}${pt.x.toFixed(1)},${pt.y.toFixed(1)}`).join(" ");
  const areaPath = linePath
    + ` L${points[points.length - 1].x.toFixed(1)},${(pad.top + plotH).toFixed(1)}`
    + ` L${points[0].x.toFixed(1)},${(pad.top + plotH).toFixed(1)} Z`;

  const last = points[points.length - 1];
  const prev = points.length >= 2 ? points[points.length - 2] : null;
  const trending = prev ? last.p.pct - prev.p.pct : 0;
  const color = trending >= 0 ? "var(--green, #22c55e)" : "var(--red, #ef4444)";

  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0.02} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill="url(#sparkFill)" />
      <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      {points.map((pt, i) => (
        <circle key={i} cx={pt.x} cy={pt.y} r={i === points.length - 1 ? 3 : 1.5}
          fill={i === points.length - 1 ? color : "var(--text-muted, #999)"} />
      ))}
      {/* Y axis labels */}
      <text x={pad.left} y={pad.top + 8} fontSize={8} fill="var(--text-dim, #666)">{Math.round(yMax)}%</text>
      <text x={pad.left} y={pad.top + plotH} fontSize={8} fill="var(--text-dim, #666)">{Math.round(yMin)}%</text>
      {/* X axis: teach count for first and last */}
      <text x={points[0].x} y={h - 2} fontSize={8} fill="var(--text-dim, #666)" textAnchor="start">#{history[0].teachCount}</text>
      <text x={last.x} y={h - 2} fontSize={8} fill="var(--text-dim, #666)" textAnchor="end">#{last.p.teachCount}</text>
    </svg>
  );
}

/** Self-contained eval score display with history graph. */
export const EvalScoreDisplay = memo(function EvalScoreDisplay({
  wsUrl,
  maxSpanWords,
  triggerCount,
}: {
  wsUrl: string;
  maxSpanWords: number;
  triggerCount: number;
}) {
  const [evalResult, setEvalResult] = useState<RetrievalPrototypeEvalResult | null>(null);
  const [prevEvalResult, setPrevEvalResult] = useState<RetrievalPrototypeEvalResult | null>(null);
  const [evalProgress, setEvalProgress] = useState<RetrievalPrototypeEvalProgress | null>(null);
  const [evalRunning, setEvalRunning] = useState(false);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const teachCountRef = useRef(triggerCount);
  teachCountRef.current = triggerCount;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setEvalRunning(true);
        setEvalProgress(null);
        const client = await connectBeeMl(wsUrl);
        const [progressTx, progressRx] = channel<RetrievalPrototypeEvalProgress>();

        const rpcPromise = client.runRetrievalPrototypeEval({
          limit: 500,
          max_span_words: maxSpanWords,
          shortlist_limit: 20,
          verify_limit: 20,
        }, progressTx).catch((e: unknown) => {
          console.error("eval RPC error:", e);
          return { ok: false as const, error: String(e) };
        });

        try {
          while (true) {
            const val = await progressRx.recv();
            if (val === null || cancelled) break;
            setEvalProgress(val);
          }
        } catch (e) {
          console.error("eval progress recv error:", e);
        }

        const response = await rpcPromise;
        if (cancelled) return;
        if (!response.ok) {
          console.error("eval failed:", response.error);
          return;
        }
        setEvalResult((prev) => {
          setPrevEvalResult(prev);
          return response.value;
        });
        const judgeableCount = response.value.canonical_gold_reachable + response.value.counterexample_cases;
        const judgeCorrectCount = response.value.canonical_judge_correct + response.value.counterexample_judge_correct;
        setHistory((prev) => [...prev, {
          teachCount: teachCountRef.current,
          judgeCorrect: judgeCorrectCount,
          evaluatedCases: judgeableCount,
          pct: judgeableCount > 0 ? Math.round((judgeCorrectCount / judgeableCount) * 100) : 0,
        }]);
      } finally {
        if (!cancelled) {
          setEvalRunning(false);
          setEvalProgress(null);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [wsUrl, maxSpanWords, triggerCount]);

  // Separate edit vs keep metrics
  const editPct = evalResult && evalResult.canonical_gold_reachable > 0
    ? Math.round((evalResult.canonical_judge_correct / evalResult.canonical_gold_reachable) * 100)
    : null;
  const keepPct = evalResult && evalResult.counterexample_cases > 0
    ? Math.round((evalResult.counterexample_judge_correct / evalResult.counterexample_cases) * 100)
    : null;

  // Combined for history tracking
  const judgeableCases = evalResult
    ? evalResult.canonical_gold_reachable + evalResult.counterexample_cases
    : null;
  const judgeCorrectCases = evalResult
    ? evalResult.canonical_judge_correct + evalResult.counterexample_judge_correct
    : null;
  const pct = judgeableCases && judgeableCases > 0
    ? Math.round((judgeCorrectCases! / judgeableCases) * 100)
    : null;
  const delta = evalResult && prevEvalResult
    ? (evalResult.canonical_judge_correct + evalResult.counterexample_judge_correct)
      - (prevEvalResult.canonical_judge_correct + prevEvalResult.counterexample_judge_correct)
    : null;
  const bestPct = history.length > 0 ? Math.max(...history.map(h => h.pct)) : null;

  return (
    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: "0.25rem", padding: "0.5rem 0" }}>
      {/* Two percentages: edit accuracy and keep accuracy */}
      <div style={{ fontVariantNumeric: "tabular-nums", fontSize: "2.0rem", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1, display: "flex", gap: "1rem", alignItems: "baseline" }}>
        {evalRunning && evalProgress ? (
          <span style={{ opacity: 0.5 }}>
            {Math.round((evalProgress.judge_correct / evalProgress.total) * 100)}%
          </span>
        ) : evalRunning ? (
          <span style={{ opacity: 0.3 }}>eval...</span>
        ) : (<>
          {editPct !== null && (
            <span>
              {editPct}%
              <span style={{ fontSize: "0.7rem", fontWeight: 500, opacity: 0.5, marginLeft: "0.25rem" }}>edit</span>
            </span>
          )}
          {keepPct !== null && (
            <span>
              {keepPct}%
              <span style={{ fontSize: "0.7rem", fontWeight: 500, opacity: 0.5, marginLeft: "0.25rem" }}>keep</span>
            </span>
          )}
          {delta !== null && delta !== 0 && (
            <span style={{
              fontSize: "1.2rem", fontWeight: 700,
              color: delta > 0 ? "var(--green, #22c55e)" : "var(--red, #ef4444)",
            }}>
              {delta > 0 ? `+${delta}` : delta}
            </span>
          )}
        </>)}
      </div>
      {/* Cases count + best */}
      {evalResult && (
        <div style={{ fontSize: "0.8rem", opacity: 0.5, fontVariantNumeric: "tabular-nums" }}>
          {judgeCorrectCases}/{judgeableCases} judgeable
          {bestPct !== null && bestPct > (pct ?? 0) && (
            <span style={{ marginLeft: "0.5rem", color: "var(--green, #22c55e)" }}>
              best {bestPct}%
            </span>
          )}
          {bestPct !== null && bestPct === pct && history.length > 1 && (
            <span style={{ marginLeft: "0.5rem", color: "var(--green, #22c55e)" }}>
              best
            </span>
          )}
        </div>
      )}

      {/* Per-stage funnel */}
      {evalResult && evalResult.canonical_cases > 0 && (
        <div style={{ fontSize: "0.75rem", opacity: 0.6, fontVariantNumeric: "tabular-nums", textAlign: "center", lineHeight: 1.6 }}>
          <span>retrieval {evalResult.canonical_shortlist_found}</span>
          <span style={{ opacity: 0.3 }}> · </span>
          <span>reachable {evalResult.canonical_gold_reachable}</span>
          <span style={{ opacity: 0.3 }}> · </span>
          <span>judge {evalResult.canonical_judge_correct}</span>
          <span style={{ opacity: 0.3 }}> / </span>
          <span>{evalResult.canonical_cases} canonical</span>
          {evalResult.counterexample_cases > 0 && (<>
            <br />
            <span>cx: {evalResult.counterexample_judge_correct}/{evalResult.counterexample_cases} correct</span>
            {evalResult.counterexample_replacement_built > 0 && (
              <span style={{ color: "var(--red, #ef4444)" }}> · {evalResult.counterexample_replacement_built} leaked</span>
            )}
          </>)}
          {(evalResult.unreachable_not_retrieved > 0 || evalResult.unreachable_missing_edits > 0 || evalResult.unreachable_surface_mismatch > 0) && (<>
            <br />
            <span style={{ opacity: 0.7 }}>unreachable: </span>
            {evalResult.unreachable_not_retrieved > 0 && <span>{evalResult.unreachable_not_retrieved} not retrieved</span>}
            {evalResult.unreachable_missing_edits > 0 && <span>{evalResult.unreachable_not_retrieved > 0 ? " · " : ""}{evalResult.unreachable_missing_edits} missing edits</span>}
            {evalResult.unreachable_surface_mismatch > 0 && <span>{(evalResult.unreachable_not_retrieved + evalResult.unreachable_missing_edits) > 0 ? " · " : ""}{evalResult.unreachable_surface_mismatch} surface</span>}
          </>)}
        </div>
      )}

      {/* Graph */}
      <SparklineGraph history={history} />
    </div>
  );
});
