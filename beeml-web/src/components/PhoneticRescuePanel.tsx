import { useMemo, useState } from "react";

import type {
  PhoneticAlignmentOp,
  PhoneticRescueTrace,
  PhoneticWordAlignment,
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

function WordInspector({ word }: { word: PhoneticWordAlignment }) {
  return (
    <article
      key={`${word.tokenStart}:${word.tokenEnd}:${word.wordText}`}
      className="failure-card"
      style={{ gap: "0.55rem" }}
    >
      <div className="failure-topline">
        <span className="mini-badge">
          {word.tokenStart}:{word.tokenEnd}
        </span>
        <span className="mini-badge">
          ZIPA {word.zipaNormStart}:{word.zipaNormEnd}
        </span>
      </div>
      <div className="failure-transcript">{word.wordText}</div>
      <div className="failure-pills">
        <span className="failure-pill">
          phones {word.transcriptNormalized.length}{"->"}{word.zipaNormalized.length}
        </span>
      </div>
      <div className="prototype-stack" style={{ gap: "0.35rem" }}>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          transcript IPA: {formatTokens(word.transcriptNormalized)}
        </div>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          ZIPA raw: {formatTokens(word.zipaRaw)}
        </div>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          ZIPA norm: {formatTokens(word.zipaNormalized)}
        </div>
      </div>
      <AlignmentView ops={word.alignment} transcriptLabel="Transcript" zipaLabel="ZIPA" />
    </article>
  );
}

export function PhoneticRescuePanel({ trace }: { trace: PhoneticRescueTrace }) {
  const wordAlignments = useMemo(
    () => [...trace.wordAlignments].sort((a, b) => a.tokenStart - b.tokenStart),
    [trace.wordAlignments],
  );
  const [selectedWordTokenStart, setSelectedWordTokenStart] = useState<number | null>(
    wordAlignments[0]?.tokenStart ?? null,
  );
  const selectedWord =
    wordAlignments.find((word) => word.tokenStart === selectedWordTokenStart) ??
    wordAlignments[0] ??
    null;

  return (
    <section className="prototype-card" style={{ marginTop: "0.75rem" }}>
      <header className="panel-header-row">
        <div>
          <strong>ZIPA Rescue</strong>
          <span>Utterance alignment between transcript IPA and ZIPA, with word-level inspection.</span>
        </div>
        <span className="badge">{wordAlignments.length} words</span>
      </header>

      <div className="prototype-summary">
        <span>rev {trace.snapshotRevision.toString()}</span>
        <span>utterance norm {formatMetric(trace.utteranceFeatureSimilarity)}</span>
        <span>utterance raw {formatMetric(trace.utteranceSimilarity)}</span>
        <span>tail volatile {trace.tailAmbiguity.volatileTokenCount}</span>
      </div>

      <div className="prototype-stack" style={{ gap: "0.35rem" }}>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          aligned transcript: {trace.alignedTranscript || "∅"}
        </div>
        {trace.pendingText ? (
          <div className="token-row muted" style={{ userSelect: "text" }}>
            pending tail: {trace.pendingText}
          </div>
        ) : null}
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

      {wordAlignments.length > 0 ? (
        <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.75rem" }}>
          <div className="prototype-stack" style={{ gap: "0.45rem" }}>
            <div className="token-row muted">Click a word to inspect its grouped IPA and corresponding ZIPA slice.</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
              {wordAlignments.map((word) => {
                const selected = selectedWord?.tokenStart === word.tokenStart;
                return (
                  <button
                    key={`${word.tokenStart}:${word.tokenEnd}:${word.wordText}`}
                    type="button"
                    className="mini-badge"
                    onClick={() => setSelectedWordTokenStart(word.tokenStart)}
                    style={{
                      cursor: "pointer",
                      userSelect: "text",
                      border: selected ? "1px solid var(--accent)" : "1px solid var(--border)",
                      background: selected ? "var(--bg-subtle)" : "var(--bg-elevated)",
                    }}
                    title={`${word.tokenStart}:${word.tokenEnd}`}
                  >
                    {word.wordText}
                  </button>
                );
              })}
            </div>
          </div>
          {selectedWord ? <WordInspector word={selectedWord} /> : null}
        </div>
      ) : (
        <div className="prototype-empty" style={{ marginTop: "0.75rem" }}>
          No word-level alignment slices available in this transcription.
        </div>
      )}
    </section>
  );
}
