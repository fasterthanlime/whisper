import { useCallback, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  AlignedWord,
  RetrievalCandidateDebug,
  RetrievalPrototypeProbeResult,
  TermInspectionResult,
} from "../beeml.generated";

function makeApproximateWords(transcript: string): AlignedWord[] {
  const words = transcript
    .trim()
    .split(/\s+/)
    .map((word) => word.trim())
    .filter(Boolean);
  return words.map((word, index) => ({
    word,
    start: index * 0.4,
    end: index * 0.4 + 0.35,
  }));
}

function formatAliasSource(source: { tag: string }) {
  return source.tag;
}

function formatIndexView(view: { tag: string }) {
  return view.tag;
}

function candidateSortKey(candidate: RetrievalCandidateDebug) {
  return candidate.phonetic_score * 1000 + candidate.coarse_score * 100;
}

function dedupeCandidatesByTerm(candidates: RetrievalCandidateDebug[]) {
  const bestByTerm = new Map<string, RetrievalCandidateDebug>();

  for (const candidate of candidates) {
    const existing = bestByTerm.get(candidate.term);
    if (!existing || candidateSortKey(candidate) > candidateSortKey(existing)) {
      bestByTerm.set(candidate.term, candidate);
    }
  }

  return [...bestByTerm.values()].sort(
    (a, b) => candidateSortKey(b) - candidateSortKey(a),
  );
}

export function RetrievalPrototypeLab({
  wsUrl,
  setWsUrl,
}: {
  wsUrl: string;
  setWsUrl: (value: string) => void;
}) {
  const [term, setTerm] = useState("AArch64");
  const [transcript, setTranscript] = useState("arc sixty four");
  const [maxSpanWords, setMaxSpanWords] = useState(4);
  const [shortlistLimit, setShortlistLimit] = useState(8);
  const [verifyLimit, setVerifyLimit] = useState(5);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [termResult, setTermResult] = useState<TermInspectionResult | null>(null);
  const [probeResult, setProbeResult] = useState<RetrievalPrototypeProbeResult | null>(null);

  const runInspect = useCallback(async () => {
    try {
      setStatus("Inspecting term...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const result = await client.inspectTerm({ term });
      if (!result.ok) throw new Error(result.error);
      setTermResult(result.value);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [term, wsUrl]);

  const runProbe = useCallback(async () => {
    try {
      setStatus("Probing retrieval prototype...");
      setError(null);
      const client = await connectBeeMl(wsUrl);
      const words = makeApproximateWords(transcript);
      const result = await client.probeRetrievalPrototype({
        transcript,
        words,
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
  }, [maxSpanWords, shortlistLimit, transcript, verifyLimit, wsUrl]);

  const interestingSpans = useMemo(() => {
    if (!probeResult) return [];
    return probeResult.spans
      .map((span) => ({
        ...span,
        candidates: dedupeCandidatesByTerm(span.candidates),
      }))
      .sort((a, b) => {
        const aBest = Math.max(...a.candidates.map(candidateSortKey), -Infinity);
        const bBest = Math.max(...b.candidates.map(candidateSortKey), -Infinity);
        return bBest - aBest;
      });
  }, [probeResult]);

  return (
    <div className="prototype-lab">
      <div className="prototype-toolbar">
        <label className="ws-label">
          <span>ws</span>
          <input
            className="ws-input"
            value={wsUrl}
            onChange={(e) => setWsUrl(e.target.value)}
            placeholder="ws://127.0.0.1:9944"
          />
        </label>
        {status && <span className="status">{status}</span>}
        {error && <span className="error">{error}</span>}
      </div>

      <div className="prototype-grid">
        <section className="prototype-card">
          <header>
            <strong>Term Inspector</strong>
            <span>Inspect aliases in the current seed lexicon</span>
          </header>
          <div className="prototype-form">
            <input
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder="AArch64"
            />
            <button className="primary" onClick={runInspect}>
              Inspect Term
            </button>
          </div>
          {termResult ? (
            <div className="term-alias-list">
              {termResult.aliases.map((alias, index) => (
                <article key={`${alias.alias_text}-${index}`} className="alias-card">
                  <div className="alias-title-row">
                    <strong>{alias.alias_text}</strong>
                    <span className="badge">{formatAliasSource(alias.alias_source)}</span>
                  </div>
                  <div className="token-row">{alias.ipa_tokens.join(" ")}</div>
                  <div className="token-row muted">
                    reduced: {alias.reduced_ipa_tokens.join(" ")}
                  </div>
                  <div className="flag-row">
                    {alias.identifier_flags.acronym_like && <span className="mini-badge">acronym</span>}
                    {alias.identifier_flags.has_digits && <span className="mini-badge">digits</span>}
                    {alias.identifier_flags.snake_like && <span className="mini-badge">snake</span>}
                    {alias.identifier_flags.camel_like && <span className="mini-badge">camel</span>}
                    {alias.identifier_flags.symbol_like && <span className="mini-badge">symbol</span>}
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="prototype-empty">No term inspection data yet.</div>
          )}
        </section>

        <section className="prototype-card">
          <header>
            <strong>Retrieval Prototype Probe</strong>
            <span>Probe the current q-gram shortlist and verifier path</span>
          </header>
          <div className="prototype-form prototype-form-vertical">
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              rows={4}
              placeholder="arc sixty four"
            />
            <div className="numeric-row">
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
            <button className="primary" onClick={runProbe}>
              Probe Retrieval Prototype
            </button>
          </div>
          {probeResult ? (
            <div className="span-list">
              <div className="prototype-summary">
                <span>spans {probeResult.spans.length}</span>
                <span>
                  with candidates{" "}
                  {probeResult.spans.filter((span) => span.candidates.length > 0).length}
                </span>
              </div>
              {interestingSpans.map((span) => (
                <article
                  key={`${span.span.token_start}-${span.span.token_end}-${span.span.text}`}
                  className="span-card"
                >
                  <div className="span-header-row">
                    <strong>{span.span.text}</strong>
                    <span className="badge">
                      {span.span.token_start}:{span.span.token_end}
                    </span>
                  </div>
                  <div className="token-row">{span.span.ipa_tokens.join(" ")}</div>
                  {span.candidates.length > 0 ? (
                    <div className="candidate-list">
                      {span.candidates.map((candidate) => (
                        <div
                          key={`${span.span.text}-${candidate.alias_id}`}
                          className="candidate-card"
                        >
                          <div className="candidate-title-row">
                            <strong>{candidate.term}</strong>
                            <span className="mini-badge">
                              {formatAliasSource(candidate.alias_source)}
                            </span>
                          </div>
                          <div className="candidate-meta">
                            <span>alias: {candidate.alias_text}</span>
                            <span>
                              lane:{" "}
                              {candidate.lane_hits
                                .map((lane) => formatIndexView(lane.view))
                                .join(", ")}
                            </span>
                          </div>
                          <div className="candidate-score-row">
                            <span>coarse {candidate.coarse_score.toFixed(3)}</span>
                            <span>view {candidate.best_view_score.toFixed(3)}</span>
                            <span>support {candidate.cross_view_support}</span>
                            <span>token {candidate.token_bonus.toFixed(2)}</span>
                            <span>phone {candidate.phone_bonus.toFixed(2)}</span>
                            <span>length {candidate.extra_length_penalty.toFixed(2)}</span>
                            <span>phonetic {candidate.phonetic_score.toFixed(3)}</span>
                            <span>q {candidate.lane_hits[0]?.qgram_overlap ?? 0}</span>
                            <span>q-total {candidate.total_qgram_overlap}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="candidate-empty">No candidates for this span.</div>
                  )}
                </article>
              ))}
            </div>
          ) : (
            <div className="prototype-empty">No probe data yet.</div>
          )}
        </section>
      </div>
    </div>
  );
}
