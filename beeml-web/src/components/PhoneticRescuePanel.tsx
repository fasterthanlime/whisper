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

function alignmentRowValues(
  ops: PhoneticAlignmentOp[],
  side: "transcript" | "zipa",
) {
  return ops.map((op) => ({
    text: side === "transcript" ? (op.transcriptToken ?? "∅") : (op.zipaToken ?? "∅"),
    kind: op.kind,
    cost: op.cost,
  }));
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
  const transcript = alignmentRowValues(ops, "transcript");
  const zipa = alignmentRowValues(ops, "zipa");
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

function WordGroup({
  word,
  label,
  selected,
  onSelect,
}: {
  word: PhoneticWordAlignment;
  label: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const transcript = alignmentRowValues(word.alignment, "transcript");
  const zipa = alignmentRowValues(word.alignment, "zipa");

  return (
    <button
      type="button"
      className={`word-group-card${selected ? " is-selected" : ""}`}
      onClick={onSelect}
      title={`${word.tokenStart}:${word.tokenEnd} · ZIPA ${word.zipaNormStart}:${word.zipaNormEnd}`}
    >
      <div className="word-group-accolade">
        <span className="word-group-accolade-label">{label}</span>
      </div>
      <div className="word-group-meta">
        <span>
          {word.tokenStart}:{word.tokenEnd}
        </span>
        <span>
          ZIPA {word.zipaNormStart}:{word.zipaNormEnd}
        </span>
      </div>
      <div className="word-group-lanes">
        <div className="word-group-lane">
          <span className="word-group-lane-label">T</span>
          <div className="word-group-box-row">
            {transcript.map((value, index) => (
              <span
                key={`t:${word.tokenStart}:${index}`}
                className={`ipa-token-box op-${value.kind.toLowerCase()}${value.text === "∅" ? " is-gap" : ""}`}
                title={`Transcript · ${value.kind} · cost ${value.cost.toFixed(3)}`}
              >
                {value.text}
              </span>
            ))}
          </div>
        </div>
        <div className="word-group-lane">
          <span className="word-group-lane-label">Z</span>
          <div className="word-group-box-row">
            {zipa.map((value, index) => (
              <span
                key={`z:${word.tokenStart}:${index}`}
                className={`ipa-token-box op-${value.kind.toLowerCase()}${value.text === "∅" ? " is-gap" : ""}`}
                title={`ZIPA · ${value.kind} · cost ${value.cost.toFixed(3)}`}
              >
                {value.text}
              </span>
            ))}
          </div>
        </div>
      </div>
    </button>
  );
}

export function PhoneticRescuePanel({ trace }: { trace: PhoneticRescueTrace }) {
  const wordAlignments = useMemo(
    () => [...trace.wordAlignments].sort((a, b) => a.tokenStart - b.tokenStart),
    [trace.wordAlignments],
  );
  const wordGroups = useMemo(() => {
    const counts = new Map<string, number>();
    const seen = new Map<string, number>();
    for (const word of wordAlignments) {
      counts.set(word.wordText, (counts.get(word.wordText) ?? 0) + 1);
    }
    return wordAlignments.map((word) => {
      const occurrence = (seen.get(word.wordText) ?? 0) + 1;
      seen.set(word.wordText, occurrence);
      const total = counts.get(word.wordText) ?? 1;
      return {
        word,
        label: total > 1 ? `${word.wordText} #${occurrence}` : word.wordText,
      };
    });
  }, [wordAlignments]);
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
          <strong>Alignment Inspector</strong>
          <span>Utterance alignment first, then grouped per-word transcript and ZIPA phones.</span>
        </div>
        <span className="badge">{wordAlignments.length} words</span>
      </header>

      <div className="prototype-stack" style={{ gap: "0.35rem" }}>
        <div className="token-row muted" style={{ userSelect: "text" }}>
          aligned transcript: {trace.alignedTranscript || "∅"}
        </div>
        {trace.pendingText ? (
          <div className="token-row muted" style={{ userSelect: "text" }}>
            pending tail: {trace.pendingText}
          </div>
        ) : null}
      </div>

      <AlignmentView
        ops={trace.utteranceAlignment}
        transcriptLabel="Transcript"
        zipaLabel="ZIPA"
      />

      <details className="phonetic-debug-details">
        <summary>Utterance diagnostics</summary>
        <div className="prototype-summary" style={{ marginTop: "0.55rem" }}>
          <span>rev {trace.snapshotRevision.toString()}</span>
          <span>utterance norm {formatMetric(trace.utteranceFeatureSimilarity)}</span>
          <span>utterance raw {formatMetric(trace.utteranceSimilarity)}</span>
          <span>tail volatile {trace.tailAmbiguity.volatileTokenCount}</span>
        </div>
        <div className="prototype-stack" style={{ gap: "0.35rem", marginTop: "0.55rem" }}>
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
      </details>

      {wordAlignments.length > 0 ? (
        <div className="word-workbench">
          <div className="word-workbench-copy">Click a grouped word block to zoom into its local transcript and ZIPA alignment.</div>
          <div className="word-group-strip">
            {wordGroups.map(({ word, label }) => (
              <WordGroup
                key={`${word.tokenStart}:${word.tokenEnd}:${word.wordText}`}
                word={word}
                label={label}
                selected={selectedWord?.tokenStart === word.tokenStart}
                onSelect={() => setSelectedWordTokenStart(word.tokenStart)}
              />
            ))}
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
