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
      setStatus("Scoring...");
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

  // After merging overlapping spans into sentence-level rows, the judge pick
  // should be the top visible row, not any stale per-span `chosen` flag.
  const judgePickId = useMemo(() => {
    if (choiceRows.length === 0) return null;
    return choiceRows[0].id;
  }, [choiceRows]);

  const teach = useCallback(
    async (row: TeachingChoiceRow) => {
      if (!currentCase) return;
      try {
        const { span, option } = row;
        const key = `${currentCase.case_id}:${span.span.token_start}:${span.span.token_end}:${option.alias_id ?? "keep"}`;
        setTeachingKey(key);
        setStatus("Teaching...");
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
          reject_group: false,
          rejected_group_spans: [],
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
              <label>
                <span>shortlist</span>
                <input type="number" min={1} max={20} value={shortlistLimit}
                  onChange={(e) => setShortlistLimit(Number(e.target.value) || 1)} />
              </label>
              <label>
                <span>verify</span>
                <input type="number" min={1} max={20} value={verifyLimit}
                  onChange={(e) => setVerifyLimit(Number(e.target.value) || 1)} />
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

  // Loaded: rapid fire interface
  const expected = currentCase?.should_abstain ? "Keep original" : currentCase?.target_term;

  return (
    <div className="prototype-lab prototype-stack">
      {/* Case header bar */}
      <section className="prototype-card prototype-card-tight rapid-fire-toolbar">
        <div className="rapid-fire-header-inline" style={{ justifyContent: "space-between", width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <strong>Rapid Fire</strong>
            <span className="mini-badge">{caseIndex + 1} / {cases.length}</span>
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
            {/* Row: transcript (what we got) */}
            <div className="choice-row choice-row-context">
              <div className="choice-row-main">
                <div className="sentence-preview-line">{currentCase.transcript}</div>
              </div>
              <div className="choice-row-meta">
                <span className="choice-flags">transcript</span>
                <span className="badge">{currentCase.suite}</span>
              </div>
            </div>

            {/* Row: expected correct sentence */}
            <div className="choice-row choice-row-context choice-row-expected">
              <div className="choice-row-main">
                <div className="sentence-preview-line">{currentCase.source_text}</div>
              </div>
              <div className="choice-row-meta">
                <span className="choice-flags">expected</span>
              </div>
            </div>

            {/* Separator + instruction */}
            {focusSpan ? (
              <>
                <div className="choice-row-divider">
                  <span>{choiceRows.length} candidates — scores are relative judge weights</span>
                  {!hasExactChoice && (
                    <span className="choice-row-divider-warn">no exact match</span>
                  )}
                </div>

                {choiceRows.map((row) => {
                  const { span, option, preview, isGold, candidateFeatures } = row;
                  const isJudgePick = row.id === judgePickId;
                  const key = `${currentCase.case_id}:${row.id}`;
                  const classes = ["choice-button"];
                  if (option.is_keep_original) classes.push("choice-button-keep");
                  if (isJudgePick) classes.push("choice-button-current");
                  if (isGold) classes.push("choice-button-gold");
                  return (
                    <button key={key} className={classes.join(" ")}
                      disabled={teachingKey === key}
                      onClick={() => void teach(row)}>
                      <div className="choice-row-main">
                        <div className="sentence-preview-line">
                          {preview.before ? <span>{preview.before} </span> : null}
                          {option.is_keep_original ? (
                            <span>{span.span.text}</span>
                          ) : (
                            <>
                              <span className="edit-from">{span.span.text}</span>
                              <span className="edit-arrow">{" => "}</span>
                              <span className="edit-to">{preview.focus}</span>
                            </>
                          )}
                          {preview.after ? <span> {preview.after}</span> : null}
                        </div>
                        {candidateFeatures && (
                          <div className="choice-detail">
                            {candidateFeatures.verified
                              ? <span className="choice-detail-verified">verified</span>
                              : <span className="choice-detail-unverified">unverified</span>}
                          </div>
                        )}
                      </div>
                      <div className="choice-row-meta">
                        <span className="choice-flags">
                          {isJudgePick ? "judge" : ""}
                          {isJudgePick && isGold ? " · " : ""}
                          {isGold ? "gold" : ""}
                        </span>
                        <span className="choice-score">{option.probability.toFixed(3)}</span>
                      </div>
                    </button>
                  );
                })}

                <button className="choice-button choice-button-skip"
                  aria-label="None of these"
                  onClick={skipCase}>
                  <div className="choice-row-main">
                    <div className="sentence-preview-line" style={{ color: "var(--text-muted)" }}>
                      None of these
                    </div>
                  </div>
                  <div className="choice-row-meta">
                    <span className="choice-flags">skip</span>
                  </div>
                </button>
              </>
            ) : (
              <div className="choice-row choice-row-context">
                <div className="choice-row-main">
                  <span style={{ color: "var(--text-muted)" }}>No usable focus span for this case.</span>
                </div>
              </div>
            )}
          </div>
      )}
    </div>
  );
}
