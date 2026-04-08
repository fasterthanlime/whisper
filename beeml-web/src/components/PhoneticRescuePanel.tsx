import type { PhoneticRescueTrace } from "../types";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
}

export function PhoneticRescuePanel({ trace }: { trace: PhoneticRescueTrace }) {
  const spans = trace.spans.filter((span) =>
    span.candidates.some((candidate) => (candidate.similarityDelta ?? 0) > 0),
  );

  return (
    <section className="prototype-card" style={{ marginTop: "0.75rem" }}>
      <header className="panel-header-row">
        <div>
          <strong>ZIPA Rescue</strong>
          <span>Segment ZIPA vs transcript span and candidate surfaces.</span>
        </div>
        <span className="badge">{spans.length} positive spans</span>
      </header>

      <div className="prototype-summary">
        <span>utterance norm {formatMetric(trace.utteranceFeatureSimilarity)}</span>
        <span>utterance raw {formatMetric(trace.utteranceSimilarity)}</span>
      </div>

      <div className="token-row" style={{ fontFamily: "'Manuale IPA', serif" }}>
        ZIPA: {trace.utteranceZipaNormalized.join(" ")}
      </div>
      <div className="token-row muted" style={{ fontFamily: "'Manuale IPA', serif" }}>
        transcript eSpeak: {trace.utteranceTranscriptNormalized.join(" ")}
      </div>

      {spans.length > 0 ? (
        <div className="failure-grid" style={{ marginTop: "0.75rem" }}>
          {spans.slice(0, 12).map((span) => (
            <article
              key={`${span.tokenStart}:${span.tokenEnd}:${span.spanText}`}
              className="failure-card"
              style={{ gap: "0.55rem" }}
            >
              <div className="failure-topline">
                <span className="mini-badge">
                  {span.tokenStart}:{span.tokenEnd}
                </span>
                <span className="failure-score">
                  base {formatMetric(span.transcriptFeatureSimilarity)}
                </span>
              </div>
              <div className="failure-transcript">{span.spanText}</div>
              <div className="token-row" style={{ fontFamily: "'Manuale IPA', serif" }}>
                ZIPA: {span.zipaNormalized.join(" ")}
              </div>
              <div className="token-row muted" style={{ fontFamily: "'Manuale IPA', serif" }}>
                transcript: {span.transcriptNormalized.join(" ")}
              </div>
              <div className="accepted-edits" style={{ marginTop: "0.25rem" }}>
                {span.candidates
                  .filter((candidate) => (candidate.similarityDelta ?? 0) > 0)
                  .slice(0, 4)
                  .map((candidate) => (
                    <span
                      key={`${span.tokenStart}:${span.tokenEnd}:${candidate.aliasText}`}
                      className="accepted-edit"
                    >
                      <span className="replacement">{candidate.aliasText}</span>
                      <span className="score">
                        {formatMetric(candidate.featureSimilarity)}
                      </span>
                      <span className="delta positive">
                        Δ +{(candidate.similarityDelta ?? 0).toFixed(4)}
                      </span>
                    </span>
                  ))}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="prototype-empty" style={{ marginTop: "0.75rem" }}>
          No positive rescue deltas in this transcription.
        </div>
      )}
    </section>
  );
}
