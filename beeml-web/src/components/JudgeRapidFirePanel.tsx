import { useCallback, useEffect, useState } from "react";
import { channel } from "@bearcove/vox-core";
import { connectBeeMl } from "../beeml.generated";
import type {
  RapidFireChoice,
  RapidFireEdit,
  RetrievalPrototypeEvalProgress,
  RetrievalPrototypeEvalResult,
  RetrievalPrototypeProbeResult,
  RetrievalPrototypeTeachingCase,
  SpanDebugTrace,
} from "../beeml.generated";
import { makeApproximateWords } from "./retrievalPrototypeUtils";

/** Render a sentence with inline diffs for each edit. */
function SentenceWithEdits({ sentence, edits }: { sentence: string; edits: RapidFireEdit[] }) {
  if (edits.length === 0) return <>{sentence}</>;

  // Find each replacement_text in the sentence, left to right
  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const edit of edits) {
    const idx = sentence.indexOf(edit.replacement_text, cursor);
    if (idx === -1) continue;

    // Text before this edit
    if (idx > cursor) {
      parts.push(sentence.slice(cursor, idx));
    }

    parts.push(
      <span key={`edit-${idx}`} style={{ position: "relative", display: "inline" }}>
        <span style={{ color: "var(--green, #22c55e)", fontWeight: 600 }}>{edit.replacement_text}</span>
        <span style={{
          position: "absolute", left: 0, top: "100%",
          fontSize: "0.7rem", opacity: 0.5, whiteSpace: "nowrap",
          textDecoration: "line-through", pointerEvents: "none",
        }}>{edit.replaced_text}</span>
      </span>
    );

    cursor = idx + edit.replacement_text.length;
  }

  // Remaining text after last edit
  if (cursor < sentence.length) {
    parts.push(sentence.slice(cursor));
  }

  return <>{parts.length > 0 ? parts : sentence}</>;
}


export function JudgeRapidFirePanel({
  wsUrl,
}: {
  wsUrl: string;
}) {
  const [deckLimit, setDeckLimit] = useState(80);
  const [maxSpanWords, setMaxSpanWords] = useState(4);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cases, setCases] = useState<RetrievalPrototypeTeachingCase[]>([]);
  const [caseIndex, setCaseIndex] = useState(0);
  const [probeResult, setProbeResult] = useState<RetrievalPrototypeProbeResult | null>(null);
  const [teachingKey, setTeachingKey] = useState<string | null>(null);
  const [evalResult, setEvalResult] = useState<RetrievalPrototypeEvalResult | null>(null);
  const [prevEvalResult, setPrevEvalResult] = useState<RetrievalPrototypeEvalResult | null>(null);
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalProgress, setEvalProgress] = useState<RetrievalPrototypeEvalProgress | null>(null);
  const [teachCount, setTeachCount] = useState(0);

  const currentCase = cases[caseIndex] ?? null;

  const runEval = useCallback(async () => {
    try {
      setEvalRunning(true);
      setEvalProgress(null);
      const client = await connectBeeMl(wsUrl);
      const [progressTx, progressRx] = channel<RetrievalPrototypeEvalProgress>();

      // Receive progress updates in background
      const progressLoop = (async () => {
        while (true) {
          const val = await progressRx.recv();
          if (val === null) break;
          setEvalProgress(val);
        }
      })();

      const response = await client.runRetrievalPrototypeEval({
        limit: 500,
        max_span_words: maxSpanWords,
        shortlist_limit: 50,
        verify_limit: 50,
      }, progressTx);
      await progressLoop;
      if (!response.ok) return;
      setEvalResult((prev) => {
        setPrevEvalResult(prev);
        return response.value;
      });
    } finally {
      setEvalRunning(false);
      setEvalProgress(null);
    }
  }, [wsUrl, maxSpanWords]);

  const loadDeck = useCallback(async () => {
    try {
      setStatus("Loading teaching deck...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const result = await client.loadRetrievalPrototypeTeachingDeck({
        limit: deckLimit,
        include_counterexamples: false,
      });
      if (!result.ok) throw new Error(result.error);
      setCases(result.value.cases);
      setCaseIndex(0);
      setProbeResult(null);
      setEvalResult(null);
      setPrevEvalResult(null);
      setTeachCount(0);
      setStatus(null);
      void runEval();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [deckLimit, wsUrl, runEval]);

  const probeCurrentCase = useCallback(async () => {
    if (!currentCase) return;
    try {
      setStatus("Scoring...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const result = await client.probeRetrievalPrototype({
        transcript: currentCase.transcript,
        words: makeApproximateWords(currentCase.transcript),
        max_span_words: maxSpanWords,
        shortlist_limit: 50,
        verify_limit: 50,
        expected_source_text: currentCase.source_text,
      });
      if (!result.ok) throw new Error(result.error);
      setProbeResult(result.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [currentCase, maxSpanWords, wsUrl]);

  useEffect(() => {
    if (cases.length === 0) return;
    void probeCurrentCase();
  }, [caseIndex, cases.length, probeCurrentCase]);

  const rapidFire = probeResult?.rapid_fire ?? null;

  const teach = useCallback(
    async (choice: RapidFireChoice) => {
      if (!currentCase) return;
      try {
        const key = `${currentCase.case_id}:${choice.option_id}`;
        setTeachingKey(key);
        setStatus("Teaching...");
        setError(null);
        const client = await connectBeeMl(wsUrl);
        const result = await client.teachRetrievalPrototypeJudge({
          probe: {
            transcript: currentCase.transcript,
            words: makeApproximateWords(currentCase.transcript),
            max_span_words: maxSpanWords,
            shortlist_limit: 50,
            verify_limit: 50,
            expected_source_text: currentCase.source_text,
          },
          span_token_start: choice.span_token_start,
          span_token_end: choice.span_token_end,
          choose_keep_original: choice.choose_keep_original,
          chosen_alias_id: choice.choose_keep_original ? null : choice.chosen_alias_id,
          reject_group: false,
          rejected_group_spans: [],
          selected_component_choices: choice.component_choices.map(cc => ({
            component_id: cc.component_id,
            choose_keep_original: cc.choose_keep_original,
            span_token_start: cc.span_token_start,
            span_token_end: cc.span_token_end,
            chosen_alias_id: cc.chosen_alias_id,
            component_spans: cc.component_spans,
          })),
        });
        if (!result.ok) throw new Error(result.error);
        setProbeResult(result.value);
        setCaseIndex((index) => Math.min(index + 1, Math.max(cases.length - 1, 0)));
        setTeachCount((n) => n + 1);
        setStatus(null);
        void runEval();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      } finally {
        setTeachingKey(null);
      }
    },
    [cases.length, currentCase, maxSpanWords, wsUrl, runEval],
  );

  const skipCase = useCallback(() => {
    setCaseIndex((i) => Math.min(i + 1, Math.max(cases.length - 1, 0)));
  }, [cases.length]);

  // Pre-load: config form
  if (cases.length === 0) {
    return (
      <div className="prototype-lab prototype-stack">
        <section className="prototype-card">
          <header>
            <strong>Judge Rapid Fire</strong>
            <span>Load a deck, then pick the correct sentence for each case.</span>
          </header>
          <div className="control-bar">
            <div className="numeric-row">
              <label>
                <span>deck</span>
                <input type="number" min={1} max={500} value={deckLimit}
                  onChange={(e) => setDeckLimit(Number(e.target.value) || 1)} />
              </label>
              <label>
                <span>max span words</span>
                <input type="number" min={1} max={8} value={maxSpanWords}
                  onChange={(e) => setMaxSpanWords(Number(e.target.value) || 1)} />
              </label>
            </div>
            <button className="primary" aria-label="Load Deck" onClick={() => void loadDeck()}>Load Deck</button>
          </div>
          {(status || error) && (
            <div className="notice-row">
              {status && <span className="status-pill">{status}</span>}
              {error && <span className="error-pill">{error}</span>}
            </div>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className="prototype-lab prototype-stack">
      {/* Case header bar */}
      <section className="prototype-card prototype-card-tight rapid-fire-toolbar">
        <div className="rapid-fire-header-inline" style={{ justifyContent: "space-between", width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <strong>Rapid Fire</strong>
            <span className="mini-badge">{caseIndex + 1} / {cases.length}</span>
            {currentCase && <span className="mini-badge">id {currentCase.case_id}</span>}
            {currentCase && <span className="mini-badge">term {currentCase.target_term}</span>}
            {teachCount > 0 && <span className="mini-badge">{teachCount} taught</span>}
            {evalResult && (() => {
              const pct = Math.round((evalResult.judge_correct / evalResult.evaluated_cases) * 100);
              const delta = prevEvalResult
                ? evalResult.judge_correct - prevEvalResult.judge_correct
                : null;
              return (
                <span style={{ fontVariantNumeric: "tabular-nums", fontSize: "1.1rem", fontWeight: 700, letterSpacing: "-0.01em" }}>
                  {evalRunning && evalProgress ? (
                    <span style={{ opacity: 0.7 }}>
                      {evalProgress.judge_correct}/{evalProgress.evaluated} <span style={{ opacity: 0.4, fontSize: "0.85rem" }}>({evalProgress.total})</span>
                    </span>
                  ) : evalRunning ? (
                    <span style={{ opacity: 0.5 }}>eval...</span>
                  ) : (<>
                    {evalResult.judge_correct}/{evalResult.evaluated_cases} ({pct}%)
                    {delta !== null && delta !== 0 && (
                      <span style={{ color: delta > 0 ? "var(--green, #22c55e)" : "var(--red, #ef4444)", marginLeft: "0.25rem" }}>
                        {delta > 0 ? `+${delta}` : delta}
                      </span>
                    )}
                  </>)}
                </span>
              );
            })()}
            {evalRunning && !evalResult && evalProgress && (
              <span style={{ fontVariantNumeric: "tabular-nums", fontSize: "1.1rem", fontWeight: 700, opacity: 0.7 }}>
                {evalProgress.judge_correct}/{evalProgress.evaluated} <span style={{ opacity: 0.4, fontSize: "0.85rem" }}>({evalProgress.total})</span>
              </span>
            )}
            {evalRunning && !evalResult && !evalProgress && <span className="status-pill">eval...</span>}
            {status && <span className="status-pill">{status}</span>}
            {error && <span className="error-pill">{error}</span>}
          </div>
          <div className="control-actions control-actions-compact">
            <button className="compact-nav" aria-label="Previous case"
              onClick={() => setCaseIndex((i) => Math.max(i - 1, 0))}
              disabled={caseIndex === 0}>Prev</button>
            <button className="compact-nav" aria-label="Next case"
              onClick={() => setCaseIndex((i) => Math.min(i + 1, Math.max(cases.length - 1, 0)))}
              disabled={caseIndex >= cases.length - 1}>Next</button>
          </div>
        </div>
      </section>

      {currentCase && (
        <div className="choice-list">
          {/* Expected sentence (gold) */}
          <div className="choice-row choice-row-context choice-row-expected">
            <div className="choice-row-main">
              <div className="sentence-preview-line">{currentCase.source_text}</div>
            </div>
            <div className="choice-row-meta">
              <span className="choice-flags">expected</span>
              <span className="badge">{currentCase.suite}</span>
            </div>
          </div>

          {rapidFire ? (
            <>
              {/* Composed sentence choices — keep_original first, then edits */}
              {[...rapidFire.choices].sort((a, b) => {
                if (a.choose_keep_original !== b.choose_keep_original) return a.choose_keep_original ? -1 : 1;
                return 0;
              }).map((choice) => {
                const key = `${currentCase.case_id}:${choice.option_id}`;
                const classes = ["choice-button"];
                if (choice.choose_keep_original) classes.push("choice-button-keep");
                if (choice.is_judge_pick) classes.push("choice-button-current");
                if (choice.is_gold) classes.push("choice-button-gold");
                return (
                  <button key={key} className={classes.join(" ")}
                    disabled={teachingKey === key}
                    onClick={() => void teach(choice)}>
                    <div className="choice-row-main">
                      <div className="sentence-preview-line">
                        {choice.choose_keep_original ? (
                          choice.sentence
                        ) : (
                          <SentenceWithEdits sentence={choice.sentence} edits={choice.edits} />
                        )}
                      </div>
                    </div>
                    <div className="choice-row-meta">
                      <span className="choice-flags">
                        {choice.choose_keep_original ? "keep transcript" : ""}
                        {choice.choose_keep_original && (choice.is_judge_pick || choice.is_gold) ? " · " : ""}
                        {choice.is_judge_pick ? "judge" : ""}
                        {choice.is_judge_pick && choice.is_gold ? " · " : ""}
                        {choice.is_gold ? "gold" : ""}
                      </span>
                      <span className="choice-score">{choice.probability.toFixed(3)}</span>
                    </div>
                  </button>
                );
              })}

              {/* Skip — no useful choice available */}
              <button className="choice-button choice-button-keep"
                onClick={skipCase}>
                <div className="choice-row-main">
                  <div className="sentence-preview-line" style={{ color: "var(--text-muted)" }}>Skip this case</div>
                </div>
                <div className="choice-row-meta">
                  <span className="choice-flags">no teach</span>
                </div>
              </button>

              {/* Debug: span-level retrieval diagnostics */}
              {probeResult && (() => {
                const targetTerm = currentCase?.target_term?.toLowerCase() ?? "";
                const interestingSpans = probeResult.spans
                  .filter((s: SpanDebugTrace) => s.candidates.length > 0)
                  .sort((a: SpanDebugTrace, b: SpanDebugTrace) => {
                    const aHasTarget = a.candidates.some(c => c.term.toLowerCase() === targetTerm) ? 1 : 0;
                    const bHasTarget = b.candidates.some(c => c.term.toLowerCase() === targetTerm) ? 1 : 0;
                    if (aHasTarget !== bHasTarget) return bHasTarget - aHasTarget;
                    const aMax = Math.max(...a.candidates.map(c => c.features.acceptance_score));
                    const bMax = Math.max(...b.candidates.map(c => c.features.acceptance_score));
                    return bMax - aMax;
                  })
                  .slice(0, 16);
                return (
                  <details className="debug-search-expander">
                    <summary>
                      Debug search
                      {rapidFire.no_exact_match && <span className="debug-warn" style={{ marginLeft: "0.5rem" }}>no exact match</span>}
                      <span style={{ marginLeft: "0.5rem", opacity: 0.5 }}>
                        {rapidFire.search_mode} · {probeResult.spans.length} spans · {interestingSpans.length} with candidates
                      </span>
                    </summary>
                    <div className="debug-search-content" style={{ fontSize: "0.95rem" }}>
                      {/* Component composition: how edits got grouped and combined */}
                      {rapidFire.components.length > 0 && (
                        <div style={{ marginBottom: "0.75rem", borderBottom: "1px solid var(--border, #333)", paddingBottom: "0.5rem" }}>
                          <div style={{ fontWeight: 600, marginBottom: "0.25rem", fontSize: "0.85rem" }}>
                            Components ({rapidFire.components.length}) · {rapidFire.total_combinations} combinations · {rapidFire.search_mode}
                          </div>
                          {rapidFire.components.map((comp) => {
                            const spanRange = comp.spans.map(s => `${s.token_start}:${s.token_end}`).join(", ");
                            return (
                              <div key={comp.component_id} style={{ marginBottom: "0.5rem", borderLeft: "2px solid var(--border, #555)", paddingLeft: "0.5rem" }}>
                                <div style={{ fontSize: "0.8rem", fontWeight: 600 }}>
                                  component {comp.component_id} · tokens {spanRange}
                                </div>
                                {comp.hypotheses.map((hyp, hi) => (
                                  <div key={hi} style={{ fontSize: "0.8rem", marginLeft: "0.5rem", display: "flex", justifyContent: "space-between", gap: "1rem" }}>
                                    {hyp.choose_keep_original ? (
                                      <span style={{ opacity: 0.5 }}>keep original</span>
                                    ) : (
                                      <span>
                                        <span style={{ textDecoration: "line-through", opacity: 0.5 }}>{hyp.replaced_text}</span>
                                        {" → "}
                                        <span style={{ color: "var(--green, #22c55e)" }}>{hyp.replacement_text}</span>
                                      </span>
                                    )}
                                    <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{hyp.probability.toFixed(3)}</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {interestingSpans.map((s: SpanDebugTrace, si: number) => {
                        const hasTarget = s.candidates.some(c => c.term.toLowerCase() === targetTerm);
                        return (
                          <div key={si} style={{ marginBottom: "0.75rem", borderLeft: hasTarget ? "2px solid var(--green, #22c55e)" : "2px solid var(--border, #333)", paddingLeft: "0.5rem" }}>
                            <div style={{ fontWeight: 600, marginBottom: "0.15rem" }}>
                              "{s.span.text}" <span style={{ opacity: 0.5, fontWeight: 400 }}>tokens {s.span.token_start}:{s.span.token_end}</span>
                            </div>
                            <div style={{ fontSize: "0.8rem", opacity: 0.6, marginBottom: "0.25rem", fontFamily: "'Manuale IPA', serif" }}>
                              ipa: {s.span.ipa_tokens.join(" ")}
                            </div>
                            {s.candidates.slice(0, 4).map((c, ci) => {
                              const isTarget = c.term.toLowerCase() === targetTerm;
                              const failedFilters = c.filter_decisions.filter(f => !f.passed);
                              return (
                                <div key={ci} style={{
                                  fontSize: "0.8rem", marginLeft: "0.5rem", marginBottom: "0.15rem",
                                  color: isTarget ? "var(--green, #22c55e)" : undefined,
                                  opacity: c.accepted ? 1 : 0.6,
                                }}>
                                  <span style={{ fontWeight: isTarget ? 700 : 400 }}>{c.term}</span>
                                  <span style={{ opacity: 0.5 }}> ({c.alias_source.tag})</span>
                                  {" "}accept={c.features.acceptance_score.toFixed(2)}
                                  {" "}phonetic={c.features.phonetic_score.toFixed(2)}
                                  {" "}coarse={c.features.coarse_score.toFixed(2)}
                                  {c.accepted ? " \u2713" : ""}
                                  {failedFilters.length > 0 && (
                                    <span style={{ color: "var(--red, #ef4444)" }}>
                                      {" "}{failedFilters.map(f => `${f.name}: ${f.detail}`).join("; ")}
                                    </span>
                                  )}
                                  <div style={{ fontFamily: "'Manuale IPA', serif", opacity: 0.5, marginLeft: "1rem" }}>
                                    ipa: {c.alias_ipa_tokens.join(" ")}
                                  </div>
                                </div>
                              );
                            })}
                            {s.candidates.length > 4 && (
                              <div style={{ fontSize: "0.75rem", opacity: 0.4, marginLeft: "0.5rem" }}>
                                +{s.candidates.length - 4} more candidates
                              </div>
                            )}
                          </div>
                        );
                      })}
                      {interestingSpans.length === 0 && (
                        <div style={{ opacity: 0.5 }}>No spans produced any candidates.</div>
                      )}
                    </div>
                  </details>
                );
              })()}
            </>
          ) : (
            <div className="choice-row choice-row-context">
              <div className="choice-row-main">
                <span style={{ color: "var(--text-muted)" }}>No usable decision set for this case.</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
