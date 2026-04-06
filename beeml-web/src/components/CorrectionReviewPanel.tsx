import { useState, useCallback, useRef, useEffect } from "react";

// ── Data model ───────────────────────────────────────────────────────

interface Correction {
  id: number;
  /** Character range in the raw transcript */
  start: number;
  end: number;
  /** The misheard text */
  originalText: string;
  /** The corrected term */
  replacementText: string;
  /** Canonical term name */
  term: string;
  /** Gate × ranker confidence */
  confidence: number;
}

/** A user-added correction (from highlighting text) */
interface UserCorrection {
  id: number;
  start: number;
  end: number;
  originalText: string;
  replacementText: string;
}

/** Resolution state for each correction */
type Resolution = "pending" | "accepted" | "rejected";

interface InsertionRecord {
  rawTranscript: string;
  corrections: Correction[];
}

const MOCK_SCENARIOS: { label: string; data: InsertionRecord }[] = [
  {
    label: "3 corrections",
    data: {
      rawTranscript:
        "I was working with Sir Day Jason and Tokyo to parse the Jason file",
      corrections: [
        {
          id: 1,
          start: 19,
          end: 32,
          originalText: "Sir Day Jason",
          replacementText: "serde_json",
          term: "serde_json",
          confidence: 0.92,
        },
        {
          id: 2,
          start: 37,
          end: 42,
          originalText: "Tokyo",
          replacementText: "tokio",
          term: "tokio",
          confidence: 0.87,
        },
        {
          id: 3,
          start: 56,
          end: 61,
          originalText: "Jason",
          replacementText: "JSON",
          term: "JSON",
          confidence: 0.95,
        },
      ],
    },
  },
  {
    label: "1 correction",
    data: {
      rawTranscript: "then I ran cube cuddle to deploy the pods",
      corrections: [
        {
          id: 1,
          start: 11,
          end: 22,
          originalText: "cube cuddle",
          replacementText: "kubectl",
          term: "kubectl",
          confidence: 0.89,
        },
      ],
    },
  },
  {
    label: "2 corrections",
    data: {
      rawTranscript:
        "we need to target arch 64 and deploy to bear cove",
      corrections: [
        {
          id: 1,
          start: 18,
          end: 25,
          originalText: "arch 64",
          replacementText: "AArch64",
          term: "AArch64",
          confidence: 0.78,
        },
        {
          id: 2,
          start: 40,
          end: 49,
          originalText: "bear cove",
          replacementText: "bearcove",
          term: "bearcove",
          confidence: 0.71,
        },
      ],
    },
  },
  {
    label: "No corrections",
    data: {
      rawTranscript: "the weather is nice today",
      corrections: [],
    },
  },
];

// ── Track segment model ──────────────────────────────────────────────

type Segment =
  | { kind: "plain"; text: string }
  | {
      kind: "split";
      id: number;
      originalText: string;
      replacementText: string;
      confidence?: number;
      resolution: Resolution;
      isUserAdded: boolean;
      editing: boolean;
    };

function buildSegments(
  rawTranscript: string,
  corrections: Correction[],
  userCorrections: UserCorrection[],
  resolutions: Map<number, Resolution>,
  editingId: number | null,
): Segment[] {
  // Merge system + user corrections, sorted by start
  const allSplits = [
    ...corrections.map((c) => ({
      id: c.id,
      start: c.start,
      end: c.end,
      originalText: c.originalText,
      replacementText: c.replacementText,
      confidence: c.confidence,
      isUserAdded: false,
    })),
    ...userCorrections.map((c) => ({
      id: c.id,
      start: c.start,
      end: c.end,
      originalText: c.originalText,
      replacementText: c.replacementText,
      confidence: undefined,
      isUserAdded: true,
    })),
  ].sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;

  for (const s of allSplits) {
    if (cursor < s.start) {
      segments.push({ kind: "plain", text: rawTranscript.slice(cursor, s.start) });
    }
    segments.push({
      kind: "split",
      id: s.id,
      originalText: s.originalText,
      replacementText: s.replacementText,
      confidence: s.confidence,
      resolution: resolutions.get(s.id) ?? "pending",
      isUserAdded: s.isUserAdded,
      editing: editingId === s.id,
    });
    cursor = s.end;
  }

  if (cursor < rawTranscript.length) {
    segments.push({ kind: "plain", text: rawTranscript.slice(cursor) });
  }

  return segments;
}

// ── Components ───────────────────────────────────────────────────────

function PlainSegment({ text }: { text: string }) {
  return <span className="track-plain">{text}</span>;
}

function SplitSegment({
  seg,
  onResolve,
  onEditChange,
  onEditSubmit,
}: {
  seg: Extract<Segment, { kind: "split" }>;
  onResolve: (id: number, resolution: Resolution) => void;
  onEditChange: (id: number, text: string) => void;
  onEditSubmit: (id: number) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (seg.editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [seg.editing]);

  if (seg.resolution === "rejected") {
    // Resolved to original — single track, show original
    return (
      <span
        className="track-resolved track-resolved-original"
        onClick={() => onResolve(seg.id, "pending")}
        title="Click to re-open"
      >
        {seg.originalText}
      </span>
    );
  }

  if (seg.resolution === "accepted") {
    // Resolved to correction — single track, show replacement
    return (
      <span
        className="track-resolved track-resolved-replacement"
        onClick={() => onResolve(seg.id, "pending")}
        title="Click to re-open"
      >
        {seg.replacementText}
      </span>
    );
  }

  // Pending — show split
  return (
    <span className="track-split">
      <span
        className="track-lane track-lane-original"
        onClick={() => onResolve(seg.id, "rejected")}
        title="Keep original"
      >
        {seg.originalText}
      </span>
      <span
        className="track-lane track-lane-replacement"
        onClick={() => onResolve(seg.id, "accepted")}
        title="Accept correction"
      >
        {seg.editing ? (
          <input
            ref={inputRef}
            className="track-edit-input"
            value={seg.replacementText}
            onChange={(e) => onEditChange(seg.id, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onEditSubmit(seg.id);
              if (e.key === "Escape") onResolve(seg.id, "rejected");
            }}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          seg.replacementText
        )}
      </span>
    </span>
  );
}

function TrackView({
  segments,
  rawTranscript,
  onResolve,
  onEditChange,
  onEditSubmit,
  onUserSelect,
}: {
  segments: Segment[];
  rawTranscript: string;
  onResolve: (id: number, resolution: Resolution) => void;
  onEditChange: (id: number, text: string) => void;
  onEditSubmit: (id: number) => void;
  onUserSelect: (start: number, end: number, text: string) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);

  const handleMouseUp = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !trackRef.current) return;

    const range = sel.getRangeAt(0);
    // Only handle selections within plain segments
    const startNode = range.startContainer;
    const endNode = range.endContainer;

    const startEl = startNode.parentElement?.closest(".track-plain");
    const endEl = endNode.parentElement?.closest(".track-plain");
    if (!startEl || !endEl) return;

    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    // Find the character offset in the raw transcript
    const idx = rawTranscript.indexOf(selectedText);
    if (idx === -1) return;

    sel.removeAllRanges();
    onUserSelect(idx, idx + selectedText.length, selectedText);
  }, [rawTranscript, onUserSelect]);

  return (
    <div
      className="track-container"
      ref={trackRef}
      onMouseUp={handleMouseUp}
    >
      {segments.map((seg, i) =>
        seg.kind === "plain" ? (
          <PlainSegment key={i} text={seg.text} />
        ) : (
          <SplitSegment
            key={seg.id}
            seg={seg}
            onResolve={onResolve}
            onEditChange={onEditChange}
            onEditSubmit={onEditSubmit}
          />
        )
      )}
    </div>
  );
}

/** Compose final text from resolutions */
function composeFinalText(
  rawTranscript: string,
  corrections: Correction[],
  userCorrections: UserCorrection[],
  resolutions: Map<number, Resolution>,
): string {
  const allSplits = [
    ...corrections.map((c) => ({
      id: c.id, start: c.start, end: c.end,
      replacementText: c.replacementText,
    })),
    ...userCorrections.map((c) => ({
      id: c.id, start: c.start, end: c.end,
      replacementText: c.replacementText,
    })),
  ].sort((a, b) => b.start - a.start); // reverse for safe replacement

  let result = rawTranscript;
  for (const s of allSplits) {
    const res = resolutions.get(s.id) ?? "pending";
    // "accepted" or "pending" (default to accepting system corrections)
    if (res !== "rejected") {
      result =
        result.slice(0, s.start) + s.replacementText + result.slice(s.end);
    }
  }
  return result;
}

// ── Main panel ───────────────────────────────────────────────────────

export function CorrectionReviewPanel() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const scenario = MOCK_SCENARIOS[scenarioIdx];

  const [resolutions, setResolutions] = useState<Map<number, Resolution>>(
    () => new Map()
  );
  const [userCorrections, setUserCorrections] = useState<UserCorrection[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [nextId, setNextId] = useState(1000);

  const switchScenario = useCallback((idx: number) => {
    setScenarioIdx(idx);
    setResolutions(new Map());
    setUserCorrections([]);
    setEditingId(null);
  }, []);

  const handleResolve = useCallback((id: number, resolution: Resolution) => {
    setResolutions((prev) => {
      const next = new Map(prev);
      next.set(id, resolution);
      return next;
    });
  }, []);

  const handleEditChange = useCallback((id: number, text: string) => {
    setUserCorrections((prev) =>
      prev.map((c) => (c.id === id ? { ...c, replacementText: text } : c))
    );
  }, []);

  const handleEditSubmit = useCallback((id: number) => {
    setEditingId(null);
    setResolutions((prev) => {
      const next = new Map(prev);
      next.set(id, "accepted");
      return next;
    });
  }, []);

  const handleUserSelect = useCallback(
    (start: number, end: number, text: string) => {
      const id = nextId;
      setNextId((n) => n + 1);
      setUserCorrections((prev) => [
        ...prev,
        { id, start, end, originalText: text, replacementText: "" },
      ]);
      setEditingId(id);
    },
    [nextId]
  );

  const segments = buildSegments(
    scenario.data.rawTranscript,
    scenario.data.corrections,
    userCorrections,
    resolutions,
    editingId,
  );

  const pendingCount = segments.filter(
    (s) => s.kind === "split" && s.resolution === "pending"
  ).length;

  const totalSplits = segments.filter((s) => s.kind === "split").length;
  const acceptedCount = segments.filter(
    (s) => s.kind === "split" && s.resolution === "accepted"
  ).length;
  const rejectedCount = segments.filter(
    (s) => s.kind === "split" && s.resolution === "rejected"
  ).length;

  const finalText = composeFinalText(
    scenario.data.rawTranscript,
    scenario.data.corrections,
    userCorrections,
    resolutions,
  );

  return (
    <div className="prototype-lab prototype-stack">
      {/* Scenario picker */}
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Correction Review — UI Prototype</strong>
            <span>
              Mock data. Simulates the ROpt-C panel after dictating.
            </span>
          </div>
        </header>
        <div className="control-bar">
          <div className="control-actions">
            {MOCK_SCENARIOS.map((s, i) => (
              <button
                key={i}
                className={scenarioIdx === i ? "primary" : ""}
                onClick={() => switchScenario(i)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* The track view */}
      <section className="prototype-card cr-panel">
        {totalSplits === 0 && userCorrections.length === 0 ? (
          <div className="cr-empty">
            <p>No corrections were applied to this transcript.</p>
            <p className="text-dim">
              Select text in the sentence below to suggest a correction.
            </p>
          </div>
        ) : (
          <div className="cr-status-bar">
            {pendingCount > 0 ? (
              <span className="cr-status-pending">
                {pendingCount} unresolved
              </span>
            ) : null}
            {acceptedCount > 0 ? (
              <span className="cr-status-accepted">
                {acceptedCount} accepted
              </span>
            ) : null}
            {rejectedCount > 0 ? (
              <span className="cr-status-rejected">
                {rejectedCount} reverted
              </span>
            ) : null}
            <span className="cr-status-hint">
              Click original (top) to keep it. Click correction (bottom) to accept.
              Select plain text to add a correction.
            </span>
          </div>
        )}

        <TrackView
          segments={segments}
          rawTranscript={scenario.data.rawTranscript}
          onResolve={handleResolve}
          onEditChange={handleEditChange}
          onEditSubmit={handleEditSubmit}
          onUserSelect={handleUserSelect}
        />

        {/* Result preview */}
        <div className="cr-preview">
          <span className="cr-preview-label">Result:</span>
          <span className="cr-preview-text">{finalText}</span>
        </div>

        {/* Actions */}
        <div className="cr-actions">
          <button
            className="primary"
            disabled={pendingCount > 0}
            title={
              pendingCount > 0
                ? "Resolve all corrections first"
                : "Apply corrections"
            }
          >
            Apply
          </button>
          <button
            onClick={() => {
              const next = new Map<number, Resolution>();
              for (const s of segments) {
                if (s.kind === "split") next.set(s.id, "rejected");
              }
              setResolutions(next);
            }}
          >
            Keep All Originals
          </button>
          <button
            onClick={() => {
              const next = new Map<number, Resolution>();
              for (const s of segments) {
                if (s.kind === "split") next.set(s.id, "accepted");
              }
              setResolutions(next);
            }}
          >
            Accept All
          </button>
        </div>
      </section>

      {/* IME notes */}
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Interaction model</strong>
          </div>
        </header>
        <div className="cr-notes">
          <ul>
            <li>
              The sentence is a <strong>track</strong> that splits at
              corrections: original on top, correction on bottom.
            </li>
            <li>
              Click the <strong>original</strong> (top lane) to resolve the
              split back to the original text.
            </li>
            <li>
              Click the <strong>correction</strong> (bottom lane) to accept it.
            </li>
            <li>
              <strong>Select any plain text</strong> to create a new split —
              a text input appears for you to type the correct term.
            </li>
            <li>
              Resolved splits merge back into the single track. Click a
              resolved word to re-open the split.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
