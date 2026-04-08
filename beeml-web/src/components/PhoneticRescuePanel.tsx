import type {
  PhoneticAlignmentOp,
  PhoneticRescueSpan,
  PhoneticRescueTrace,
} from "../types";

function formatMetric(value: number | null | undefined) {
  return value == null ? "n/a" : value.toFixed(4);
}

function formatTokens(tokens: string[]) {
  return tokens.length > 0 ? tokens.join(" ") : "∅";
}

function opBorder(kind: PhoneticAlignmentOp["kind"]) {
  switch (kind) {
    case "Match":
      return "1px solid color-mix(in srgb, var(--lane-qwen) 30%, transparent)";
    case "Substitute":
      return "1px solid color-mix(in srgb, var(--lane-zipa) 65%, var(--lane-espeak) 35%)";
    case "Insert":
    case "Delete":
      return "1px dashed color-mix(in srgb, var(--text-dim) 50%, transparent)";
  }
}

function opBg(kind: PhoneticAlignmentOp["kind"]) {
  switch (kind) {
    case "Match":
      return "color-mix(in srgb, var(--lane-qwen-bg) 65%, transparent)";
    case "Substitute":
      return "color-mix(in srgb, var(--lane-zipa-bg) 55%, var(--lane-espeak-bg) 45%)";
    case "Insert":
    case "Delete":
      return "color-mix(in srgb, var(--bg-subtle) 70%, transparent)";
  }
}

function AlignmentLane({
  label,
  values,
  color,
  columnCount,
}: {
  label: string;
  values: { text: string; kind: PhoneticAlignmentOp["kind"]; cost: number }[];
  color: string;
  columnCount: number;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "78px minmax(0, 1fr)", gap: "0.65rem" }}>
      <div style={{ color, fontWeight: 700, paddingTop: "0.3rem" }}>{label}</div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${columnCount}, minmax(2.25rem, max-content))`,
          gap: "0.25rem",
          alignItems: "center",
          paddingBottom: "0.1rem",
          width: "max-content",
        }}
      >
        {values.map((value, index) => (
          <span
            key={`${label}:${index}`}
            title={`${value.kind} cost ${value.cost.toFixed(3)}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              minHeight: "2rem",
              padding: "0.15rem 0.45rem",
              borderRadius: "0.5rem",
              border: opBorder(value.kind),
              background: opBg(value.kind),
              fontFamily: "'Manuale IPA', serif",
              color,
              opacity: value.text === "∅" ? 0.55 : 1,
            }}
          >
            {value.text}
          </span>
        ))}
      </div>
    </div>
  );
}

export function AlignmentView({
  ops,
  transcriptLabel,
  zipaLabel,
}: {
  ops: PhoneticAlignmentOp[];
  transcriptLabel: string;
  zipaLabel: string;
}) {
  const transcript = ops.map((op) => ({
    text: op.transcriptToken ?? "∅",
    kind: op.kind,
    cost: op.cost,
  }));
  const zipa = ops.map((op) => ({
    text: op.zipaToken ?? "∅",
    kind: op.kind,
    cost: op.cost,
  }));
  const columnCount = Math.max(ops.length, 1);

  return (
    <div
      style={{
        display: "grid",
        gap: "0.45rem",
        marginTop: "0.5rem",
      }}
    >
      <div style={{ overflowX: "auto", paddingBottom: "0.15rem" }}>
        <div style={{ width: "max-content", minWidth: "100%" }}>
          <AlignmentLane
            label={transcriptLabel}
            values={transcript}
            color="var(--lane-espeak)"
            columnCount={columnCount}
          />
          <AlignmentLane
            label={zipaLabel}
            values={zipa}
            color="var(--lane-zipa)"
            columnCount={columnCount}
          />
        </div>
      </div>
    </div>
  );
}

function SpanCard({ span }: { span: PhoneticRescueSpan }) {
  return (
    <article key={`${span.tokenStart}:${span.tokenEnd}:${span.spanText}`} className="failure-card" style={{ gap: "0.55rem" }}>
      <div className="failure-topline">
        <span className="mini-badge">
          {span.tokenStart}:{span.tokenEnd}
        </span>
        <span className="mini-badge">
          ZIPA {span.zipaNormStart}:{span.zipaNormEnd}
        </span>
        <span className="mini-badge">{span.alignmentSource}</span>
        <span className="mini-badge">{span.anchorConfidence.toLowerCase()}</span>
        <span className="failure-score">
          base {formatMetric(span.transcriptFeatureSimilarity)}
        </span>
      </div>
      <div className="failure-transcript">{span.spanText}</div>
      <div className="failure-pills">
        <span className="failure-pill">
          phones {span.transcriptPhoneCount}{"->"}
          {span.chosenZipaPhoneCount}
        </span>
        <span className="failure-pill">
          proj {formatMetric(span.projectedAlignmentScore)}
        </span>
        <span className="failure-pill">
          chosen {formatMetric(span.chosenAlignmentScore)}
        </span>
        <span className="failure-pill">
          gap {formatMetric(span.alignmentScoreGap)}
        </span>
      </div>
      <AlignmentView ops={span.alignment} transcriptLabel="Transcript" zipaLabel="ZIPA" />
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
              <span className="score">{formatMetric(candidate.featureSimilarity)}</span>
              <span className="delta positive">
                Δ +{(candidate.similarityDelta ?? 0).toFixed(4)}
              </span>
            </span>
          ))}
      </div>
    </article>
  );
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
          <span>Utterance and span-level DP alignment between transcript eSpeak and ZIPA.</span>
        </div>
        <span className="badge">{spans.length} positive spans</span>
      </header>

      <div className="prototype-summary">
        <span>utterance norm {formatMetric(trace.utteranceFeatureSimilarity)}</span>
        <span>utterance raw {formatMetric(trace.utteranceSimilarity)}</span>
      </div>

      <div className="prototype-stack" style={{ gap: "0.35rem" }}>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          transcript norm: {formatTokens(trace.utteranceTranscriptNormalized)}
        </div>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          ZIPA raw: {formatTokens(trace.utteranceZipaRaw)}
        </div>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          ZIPA norm: {formatTokens(trace.utteranceZipaNormalized)}
        </div>
      </div>

      <AlignmentView
        ops={trace.utteranceAlignment}
        transcriptLabel="Transcript"
        zipaLabel="ZIPA"
      />

      {spans.length > 0 ? (
        <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.75rem" }}>
          {spans.slice(0, 12).map((span) => (
            <SpanCard key={`${span.tokenStart}:${span.tokenEnd}:${span.spanText}`} span={span} />
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
