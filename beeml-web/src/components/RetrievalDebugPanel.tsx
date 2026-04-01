import { useCallback, useState } from "react";
import type { RetrievalDebugResult, RetrievalDebugSpan } from "../types";
import { phoneticRetrievalDebug } from "../api";

const DEFAULT_TRANSCRIPT =
  "We should build the release binary for ARC sixty four so it runs natively on the ARM server.";

export function RetrievalDebugPanel() {
  const [transcript, setTranscript] = useState(DEFAULT_TRANSCRIPT);
  const [maxSpanWords, setMaxSpanWords] = useState(4);
  const [maxCandidates, setMaxCandidates] = useState(5);
  const [maxSpans, setMaxSpans] = useState(12);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RetrievalDebugResult | null>(null);

  const handleRun = useCallback(async () => {
    try {
      setStatus("Running retrieval...");
      setError(null);
      const next = await phoneticRetrievalDebug({
        transcript,
        maxSpanWords,
        maxCandidatesPerSpan: maxCandidates,
        maxSpans,
      });
      setResult(next);
      setStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [maxCandidates, maxSpanWords, maxSpans, transcript]);

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          padding: "0.75rem 1rem",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface-alt)",
        }}
      >
        <textarea
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          rows={3}
          placeholder="Transcript to probe"
          style={{
            width: "100%",
            resize: "vertical",
            font: "inherit",
            background: "var(--bg-surface)",
            color: "var(--text)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: "0.6rem 0.75rem",
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            flexWrap: "wrap",
          }}
        >
          <NumberField
            label="Span words"
            value={maxSpanWords}
            min={1}
            max={6}
            onChange={setMaxSpanWords}
          />
          <NumberField
            label="Candidates"
            value={maxCandidates}
            min={1}
            max={24}
            onChange={setMaxCandidates}
          />
          <NumberField
            label="Returned spans"
            value={maxSpans}
            min={1}
            max={256}
            onChange={setMaxSpans}
          />
          <button className="primary" onClick={handleRun} disabled={!transcript.trim() || !!status}>
            PROBE
          </button>
          {status && <span style={{ fontSize: "0.8rem", color: "var(--accent)" }}>{status}</span>}
          {error && <span style={{ fontSize: "0.8rem", color: "var(--danger)" }}>{error}</span>}
          {result && (
            <div style={{ display: "flex", gap: "0.75rem", marginLeft: "auto", flexWrap: "wrap" }}>
              <Metric label="Aliases" value={result.summary.aliasCount} />
              <Metric label="Spans" value={result.summary.returnedSpans} />
              <Metric label="Total ms" value={result.timing.totalMs} />
            </div>
          )}
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "1rem" }}>
        {!result ? (
          <div style={{ color: "var(--text-muted)" }}>
            Probe transcript spans and inspect shortlist vs verified candidates.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <TimingRow result={result} />
            {result.spans.map((span) => (
              <SpanCard key={`${span.tokenStart}-${span.tokenEnd}-${span.charStart}`} span={span} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.85rem" }}>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10) || min)}
        style={{ width: 72 }}
      />
    </label>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "0.3rem 0.7rem",
        minWidth: 88,
      }}
    >
      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
        {label}
      </div>
      <div style={{ fontSize: "1rem", fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function TimingRow({ result }: { result: RetrievalDebugResult }) {
  const t = result.timing;
  return (
    <div
      style={{
        display: "flex",
        gap: "0.75rem",
        flexWrap: "wrap",
        padding: "0.75rem",
        border: "1px solid var(--border)",
        borderRadius: 8,
        background: "var(--bg-surface)",
      }}
    >
      <Metric label="DB" value={t.dbMs} />
      <Metric label="Lexicon" value={t.lexiconMs} />
      <Metric label="Index" value={t.indexMs} />
      <Metric label="Span Enum" value={t.spanEnumerationMs} />
      <Metric label="Shortlist" value={t.shortlistMs} />
      <Metric label="Verify" value={t.verifyMs} />
      <Metric label="Total" value={t.totalMs} />
    </div>
  );
}

function SpanCard({ span }: { span: RetrievalDebugSpan }) {
  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 10,
        background: "var(--bg-surface)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "1rem",
          padding: "0.75rem 1rem",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface-alt)",
          flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontSize: "1rem", fontWeight: 700 }}>{span.text}</div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            tokens {span.tokenStart}-{span.tokenEnd} · chars {span.charStart}-{span.charEnd}
          </div>
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          IPA {span.ipaTokens.join(" ")}
        </div>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1rem",
          padding: "1rem",
        }}
      >
        <CandidateTable title="Shortlist" candidates={span.shortlist} />
        <CandidateTable title="Verified" candidates={span.verified} showPhonetic />
      </div>
    </div>
  );
}

function CandidateTable({
  title,
  candidates,
  showPhonetic = false,
}: {
  title: string;
  candidates: Array<
    RetrievalDebugSpan["shortlist"][number] | RetrievalDebugSpan["verified"][number]
  >;
  showPhonetic?: boolean;
}) {
  return (
    <div>
      <div style={{ fontSize: "0.8rem", fontWeight: 700, marginBottom: "0.5rem" }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {candidates.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>No candidates</div>
        ) : (
          candidates.map((candidate) => (
            <div
              key={`${title}-${candidate.aliasId}`}
              style={{
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: "0.65rem 0.75rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.2rem",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
                <div>
                  <strong>{candidate.term}</strong>
                  <span style={{ color: "var(--text-muted)" }}> via {candidate.aliasSource}</span>
                </div>
                <div style={{ fontVariantNumeric: "tabular-nums" }}>
                  {showPhonetic && "phoneticScore" in candidate && candidate.phoneticScore != null
                    ? candidate.phoneticScore.toFixed(3)
                    : candidate.coarseScore.toFixed(3)}
                </div>
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                alias {candidate.aliasText} · {candidate.matchedView} · qgrams {candidate.qgramOverlap}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>
                phone Δ {candidate.phoneCountDelta}
                {candidate.tokenCountMatch !== undefined
                  ? ` · token-count ${candidate.tokenCountMatch ? "match" : "mismatch"}`
                  : ""}
                {showPhonetic && "phoneticScore" in candidate && candidate.phoneticScore != null
                  ? ` · coarse ${candidate.coarseScore.toFixed(3)}`
                  : ""}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
