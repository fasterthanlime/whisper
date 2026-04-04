import { useCallback, useEffect, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  RapidFireChoice,
  RapidFireEdit,
  RetrievalPrototypeProbeResult,
  RetrievalPrototypeTeachingCase,
} from "../beeml.generated";
import { makeApproximateWords } from "./retrievalPrototypeUtils";
import { EvalScoreDisplay } from "./EvalScoreDisplay";
import { DebugSearchPanel } from "./DebugSearchPanel";

function SentenceWithEdits({ sentence, edits }: { sentence: string; edits: RapidFireEdit[] }) {
  if (edits.length === 0) return <>{sentence}</>;

  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const edit of edits) {
    const idx = sentence.indexOf(edit.replacement_text, cursor);
    if (idx === -1) continue;

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
  const [teachCount, setTeachCount] = useState(0);
  const [evalTrigger, setEvalTrigger] = useState(0);

  const currentCase = cases[caseIndex] ?? null;

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
      setTeachCount(0);
      setStatus(null);
      setEvalTrigger((n) => n + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [deckLimit, wsUrl]);

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
        setEvalTrigger((n) => n + 1);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      } finally {
        setTeachingKey(null);
      }
    },
    [cases.length, currentCase, maxSpanWords, wsUrl],
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
      {/* Eval score + graph */}
      <section className="prototype-card" style={{ display: "flex", justifyContent: "center" }}>
        <EvalScoreDisplay wsUrl={wsUrl} maxSpanWords={maxSpanWords} triggerCount={evalTrigger} />
      </section>

      {/* Case header bar */}
      <section className="prototype-card prototype-card-tight rapid-fire-toolbar">
        <div className="rapid-fire-header-inline" style={{ justifyContent: "space-between", width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <strong>Rapid Fire</strong>
            <span className="mini-badge">{caseIndex + 1} / {cases.length}</span>
            {currentCase && <span className="mini-badge">id {currentCase.case_id}</span>}
            {currentCase && <span className="mini-badge">term {currentCase.target_term}</span>}
            {teachCount > 0 && <span className="mini-badge">{teachCount} taught</span>}
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
              {/* Choices — keep_original first, then edits */}
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

              {/* Skip */}
              <button className="choice-button choice-button-keep" onClick={skipCase}>
                <div className="choice-row-main">
                  <div className="sentence-preview-line" style={{ color: "var(--text-muted)" }}>Skip this case</div>
                </div>
                <div className="choice-row-meta">
                  <span className="choice-flags">no teach</span>
                </div>
              </button>

              {/* Debug panel — content only rendered when expanded */}
              {probeResult && (
                <DebugSearchPanel
                  probeResult={probeResult}
                  rapidFire={rapidFire}
                  targetTerm={currentCase?.target_term?.toLowerCase() ?? ""}
                />
              )}
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
