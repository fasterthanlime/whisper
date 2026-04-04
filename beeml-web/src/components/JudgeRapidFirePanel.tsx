import { useCallback, useEffect, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  RetrievalPrototypeProbeResult,
  RetrievalPrototypeTeachingCase,
} from "../beeml.generated";
import {
  buildOverlapGroups,
  buildTeachingChoices,
  dedupeJudgeOptions,
  makeApproximateWords,
  pickFocusSpan,
  type TeachingChoiceRow,
} from "./retrievalPrototypeUtils";

export function JudgeRapidFirePanel({
  wsUrl,
}: {
  wsUrl: string;
}) {
  const [deckLimit, setDeckLimit] = useState(80);
  const [maxSpanWords, setMaxSpanWords] = useState(4);
  const [shortlistLimit, setShortlistLimit] = useState(8);
  const [verifyLimit, setVerifyLimit] = useState(5);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cases, setCases] = useState<RetrievalPrototypeTeachingCase[]>([]);
  const [caseIndex, setCaseIndex] = useState(0);
  const [probeResult, setProbeResult] = useState<RetrievalPrototypeProbeResult | null>(null);
  const [teachingKey, setTeachingKey] = useState<string | null>(null);

  const currentCase = cases[caseIndex] ?? null;

  const loadDeck = useCallback(async () => {
    try {
      setStatus("Loading teaching deck...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const result = await client.loadRetrievalPrototypeTeachingDeck({
        limit: deckLimit,
        include_counterexamples: true,
      });
      if (!result.ok) throw new Error(result.error);
      setCases(result.value.cases);
      setCaseIndex(0);
      setProbeResult(null);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [deckLimit, wsUrl]);

  const probeCurrentCase = useCallback(async () => {
    if (!currentCase) return;
    try {
      setStatus("Scoring current case...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const result = await client.probeRetrievalPrototype({
        transcript: currentCase.transcript,
        words: makeApproximateWords(currentCase.transcript),
        max_span_words: maxSpanWords,
        shortlist_limit: shortlistLimit,
        verify_limit: verifyLimit,
      });
      if (!result.ok) throw new Error(result.error);
      setProbeResult(result.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [currentCase, maxSpanWords, shortlistLimit, verifyLimit, wsUrl]);

  useEffect(() => {
    if (cases.length === 0) return;
    void probeCurrentCase();
  }, [caseIndex, cases.length, probeCurrentCase]);

  const focusSpan = useMemo(() => {
    if (!probeResult || !currentCase) return null;
    const normalizedSpans = probeResult.spans.map((span) => ({
      ...span,
      judge_options: dedupeJudgeOptions(span.judge_options),
    }));
    return pickFocusSpan(normalizedSpans, currentCase);
  }, [currentCase, probeResult]);

  const focusGroup = useMemo(() => {
    if (!probeResult || !currentCase || !focusSpan) return [];
    const normalizedSpans = probeResult.spans.map((span) => ({
      ...span,
      judge_options: dedupeJudgeOptions(span.judge_options),
    }));
    const groups = buildOverlapGroups(normalizedSpans);
    return (
      groups.find((group) =>
        group.some(
          (span) =>
            span.span.token_start === focusSpan.span.token_start &&
            span.span.token_end === focusSpan.span.token_end,
        ),
      ) ?? [focusSpan]
    );
  }, [currentCase, focusSpan, probeResult]);

  const choiceRows = useMemo(() => {
    if (!currentCase || focusGroup.length === 0) return [];
    return buildTeachingChoices(currentCase.transcript, focusGroup, currentCase);
  }, [currentCase, focusGroup]);

  const hasExactChoice = choiceRows.some((row) => row.isGold);

  const teach = useCallback(
    async (row: TeachingChoiceRow) => {
      if (!currentCase) return;
      try {
        const { span, option } = row;
        const key = `${currentCase.case_id}:${span.span.token_start}:${span.span.token_end}:${option.alias_id ?? "keep"}`;
        setTeachingKey(key);
        setStatus("Teaching judge...");
        setError(null);
        const client = await connectBeeMl(wsUrl);
        const result = await client.teachRetrievalPrototypeJudge({
          probe: {
            transcript: currentCase.transcript,
            words: makeApproximateWords(currentCase.transcript),
            max_span_words: maxSpanWords,
            shortlist_limit: shortlistLimit,
            verify_limit: verifyLimit,
          },
          span_token_start: span.span.token_start,
          span_token_end: span.span.token_end,
          choose_keep_original: option.is_keep_original,
          chosen_alias_id: option.is_keep_original ? null : option.alias_id,
        });
        if (!result.ok) throw new Error(result.error);
        setProbeResult(result.value);
        setCaseIndex((index) => Math.min(index + 1, Math.max(cases.length - 1, 0)));
        setStatus(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      } finally {
        setTeachingKey(null);
      }
    },
    [cases.length, currentCase, maxSpanWords, shortlistLimit, verifyLimit, wsUrl],
  );

  return (
    <div className="prototype-lab prototype-stack">
      <section className="prototype-card prototype-card-tight rapid-fire-toolbar">
        {cases.length === 0 ? (
          <>
            <header className="panel-header-row">
              <div>
                <strong>Judge Rapid Fire</strong>
                <span>Load a deck, then click the correct sentence row.</span>
              </div>
            </header>
            <div className="control-bar">
              <div className="numeric-row">
                <label>
                  <span>deck</span>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    value={deckLimit}
                    onChange={(e) => setDeckLimit(Number(e.target.value) || 1)}
                  />
                </label>
                <label>
                  <span>max span words</span>
                  <input
                    type="number"
                    min={1}
                    max={8}
                    value={maxSpanWords}
                    onChange={(e) => setMaxSpanWords(Number(e.target.value) || 1)}
                  />
                </label>
                <label>
                  <span>shortlist</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={shortlistLimit}
                    onChange={(e) => setShortlistLimit(Number(e.target.value) || 1)}
                  />
                </label>
                <label>
                  <span>verify</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={verifyLimit}
                    onChange={(e) => setVerifyLimit(Number(e.target.value) || 1)}
                  />
                </label>
              </div>
              <div className="control-actions">
                <button className="primary" onClick={() => void loadDeck()}>
                  Load Deck
                </button>
              </div>
            </div>
          </>
        ) : (
          <header className="panel-header-row panel-header-compact">
            <div className="rapid-fire-header-inline">
              <strong>Rapid Fire</strong>
              <span className="mini-badge">
                {caseIndex + 1} / {cases.length}
              </span>
            </div>
            <div className="control-actions control-actions-compact">
              <button
                className="compact-nav"
                onClick={() => setCaseIndex((index) => Math.max(index - 1, 0))}
                disabled={caseIndex === 0}
                title="Previous case"
              >
                Prev
              </button>
              <button
                className="compact-nav"
                onClick={() =>
                  setCaseIndex((index) =>
                    Math.min(index + 1, Math.max(cases.length - 1, 0)),
                  )
                }
                disabled={caseIndex >= cases.length - 1}
                title="Next case"
              >
                Next
              </button>
            </div>
          </header>
        )}

        {(status || error) && (
          <div className="notice-row">
            {status ? <span className="status-pill">{status}</span> : null}
            {error ? <span className="error-pill">{error}</span> : null}
          </div>
        )}
      </section>

      {currentCase ? (
        <>
          <section className="prototype-card rapid-fire-hero">
            <div className="rapid-fire-hero-top">
              <span className="eyebrow">Current sentence</span>
              <span className="badge">{currentCase.suite}</span>
            </div>
            <h1 className="rapid-fire-transcript">{currentCase.transcript}</h1>
            <div className="rapid-fire-metrics">
              <div className="hero-metric hero-metric-primary">
                <span className="hero-metric-label">Expected answer</span>
                <strong>
                  {currentCase.should_abstain ? "Keep original" : currentCase.target_term}
                </strong>
              </div>
              <div className="hero-metric">
                <span className="hero-metric-label">Original source</span>
                <span>{currentCase.source_text}</span>
              </div>
              {currentCase.surface_form ? (
                <div className="hero-metric">
                  <span className="hero-metric-label">Surface form</span>
                  <span>{currentCase.surface_form}</span>
                </div>
              ) : null}
            </div>
          </section>

          {focusSpan ? (
            <section className="prototype-card focus-panel">
              <div className="focus-panel-header">
                <div>
                  <span className="eyebrow">Choose correction</span>
                  <div className="focus-span-text">Pick the best sentence</div>
                </div>
                <span className="badge">
                  {choiceRows.length} options
                </span>
              </div>
              <div className="focus-instruction">
                Click the correct full-sentence edit. Overlapping spans compete here together.
              </div>
              {!hasExactChoice ? (
                <div className="span-warning">
                  None of these choices reconstructs the expected sentence exactly. This is probably the wrong span.
                </div>
              ) : null}
              <div className="choice-list">
                {choiceRows.map((row) => {
                  const { span, option, preview, isGold } = row;
                  const key = `${currentCase.case_id}:${row.id}`;
                  const classes = ["choice-button"];
                  if (option.is_keep_original) classes.push("choice-button-keep");
                  if (option.chosen) classes.push("choice-button-current");
                  if (isGold) classes.push("choice-button-gold");
                  return (
                    <button
                      key={key}
                      className={classes.join(" ")}
                      disabled={teachingKey === key}
                      onClick={() => void teach(row)}
                    >
                      <div className="choice-row-main">
                        <div className="sentence-preview-line">
                          {preview.before ? <span>{preview.before} </span> : null}
                          <mark>{preview.focus}</mark>
                          {preview.after ? <span> {preview.after}</span> : null}
                        </div>
                      </div>
                      <div className="choice-row-meta">
                        <span className="choice-flags">
                          {option.is_keep_original ? "keep original" : option.term}
                          {option.chosen ? " · current" : ""}
                          {isGold ? " · gold" : ""}
                        </span>
                        <span className="choice-meta-pill">span {span.span.text}</span>
                        <span className="choice-meta-pill">
                          ipa {span.span.ipa_tokens.join(" ")}
                        </span>
                        <span className="choice-score">{option.probability.toFixed(3)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          ) : (
            <section className="prototype-card prototype-card-tight">
              <div className="prototype-empty">No usable focus span for this case yet.</div>
            </section>
          )}
        </>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">Load a deck to start rapid-fire teaching.</div>
        </section>
      )}
    </div>
  );
}
