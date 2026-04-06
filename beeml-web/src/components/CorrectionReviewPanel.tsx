import { useState, useCallback, useRef, useEffect } from "react";

// ── Data model ───────────────────────────────────────────────────────

interface Correction {
  id: number;
  start: number;
  end: number;
  originalText: string;
  replacementText: string;
  term: string;
  confidence: number;
}

interface UserCorrection {
  id: number;
  start: number;
  end: number;
  originalText: string;
  replacementText: string;
}

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
          id: 1, start: 19, end: 32,
          originalText: "Sir Day Jason", replacementText: "serde_json",
          term: "serde_json", confidence: 0.92,
        },
        {
          id: 2, start: 37, end: 42,
          originalText: "Tokyo", replacementText: "tokio",
          term: "tokio", confidence: 0.87,
        },
        {
          id: 3, start: 56, end: 61,
          originalText: "Jason", replacementText: "JSON",
          term: "JSON", confidence: 0.95,
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
          id: 1, start: 11, end: 22,
          originalText: "cube cuddle", replacementText: "kubectl",
          term: "kubectl", confidence: 0.89,
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
          id: 1, start: 18, end: 25,
          originalText: "arch 64", replacementText: "AArch64",
          term: "AArch64", confidence: 0.78,
        },
        {
          id: 2, start: 40, end: 49,
          originalText: "bear cove", replacementText: "bearcove",
          term: "bearcove", confidence: 0.71,
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
  const allSplits = [
    ...corrections.map((c) => ({
      id: c.id, start: c.start, end: c.end,
      originalText: c.originalText, replacementText: c.replacementText,
      confidence: c.confidence, isUserAdded: false,
    })),
    ...userCorrections.map((c) => ({
      id: c.id, start: c.start, end: c.end,
      originalText: c.originalText, replacementText: c.replacementText,
      confidence: undefined, isUserAdded: true,
    })),
  ].sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;

  for (const s of allSplits) {
    if (cursor < s.start) {
      segments.push({ kind: "plain", text: rawTranscript.slice(cursor, s.start) });
    }
    segments.push({
      kind: "split", id: s.id,
      originalText: s.originalText, replacementText: s.replacementText,
      confidence: s.confidence, resolution: resolutions.get(s.id) ?? "pending",
      isUserAdded: s.isUserAdded, editing: editingId === s.id,
    });
    cursor = s.end;
  }

  if (cursor < rawTranscript.length) {
    segments.push({ kind: "plain", text: rawTranscript.slice(cursor) });
  }

  return segments;
}

// ── Components ───────────────────────────────────────────────────────

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

    const startEl = sel.anchorNode?.parentElement?.closest(".track-plain");
    const endEl = sel.focusNode?.parentElement?.closest(".track-plain");
    if (!startEl || !endEl) return;

    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    const idx = rawTranscript.indexOf(selectedText);
    if (idx === -1) return;

    sel.removeAllRanges();
    onUserSelect(idx, idx + selectedText.length, selectedText);
  }, [rawTranscript, onUserSelect]);

  return (
    <div className="track-container" ref={trackRef} onMouseUp={handleMouseUp}>
      {segments.map((seg, i) =>
        seg.kind === "plain" ? (
          <span key={i} className="track-plain">{seg.text}</span>
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
  ].sort((a, b) => b.start - a.start);

  let result = rawTranscript;
  for (const s of allSplits) {
    const res = resolutions.get(s.id) ?? "pending";
    if (res !== "rejected") {
      result =
        result.slice(0, s.start) + s.replacementText + result.slice(s.end);
    }
  }
  return result;
}

// ── The window mockup ────────────────────────────────────────────────

function CorrectionWindow({
  scenario,
  onClose,
}: {
  scenario: InsertionRecord;
  onClose: () => void;
}) {
  const [resolutions, setResolutions] = useState<Map<number, Resolution>>(
    () => new Map()
  );
  const [userCorrections, setUserCorrections] = useState<UserCorrection[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [nextId, setNextId] = useState(1000);

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
    scenario.rawTranscript,
    scenario.corrections,
    userCorrections,
    resolutions,
    editingId,
  );

  const pendingCount = segments.filter(
    (s) => s.kind === "split" && s.resolution === "pending"
  ).length;

  const finalText = composeFinalText(
    scenario.rawTranscript,
    scenario.corrections,
    userCorrections,
    resolutions,
  );

  // Handle keyboard
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !editingId) {
        e.preventDefault();
        onClose();
      }
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, editingId]);

  const hasCorrections = scenario.corrections.length > 0 || userCorrections.length > 0;

  return (
    <div className="cw-backdrop">
      <div className="cw-window">
        {/* Title bar */}
        <div className="cw-titlebar">
          <div className="cw-titlebar-dots">
            <span className="cw-dot cw-dot-close" onClick={onClose} />
            <span className="cw-dot cw-dot-min" />
            <span className="cw-dot cw-dot-max" />
          </div>
          <span className="cw-title">Review Corrections</span>
          <div className="cw-titlebar-spacer" />
        </div>

        {/* Body */}
        <div className="cw-body">
          {hasCorrections ? (
            <>
              <div className="cw-instructions">
                Click the <strong>top</strong> lane to keep the original.
                Click the <strong>bottom</strong> lane to accept the correction.
                {" "}Select any text to suggest your own correction.
              </div>

              <TrackView
                segments={segments}
                rawTranscript={scenario.rawTranscript}
                onResolve={handleResolve}
                onEditChange={handleEditChange}
                onEditSubmit={handleEditSubmit}
                onUserSelect={handleUserSelect}
              />

              <div className="cw-result">
                <span className="cw-result-text">{finalText}</span>
              </div>
            </>
          ) : (
            <div className="cw-no-corrections">
              No corrections for this transcript.
              <span className="text-dim">
                Select text above to suggest a correction.
              </span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="cw-footer">
          <div className="cw-shortcut-hints">
            <span className="cw-kbd">esc</span>
            <span className="cw-hint-text">dismiss</span>
          </div>
          <button
            className="cw-apply-btn"
            onClick={onClose}
            disabled={pendingCount > 0}
            title={pendingCount > 0 ? `Resolve ${pendingCount} remaining` : ""}
          >
            {pendingCount > 0 ? (
              <>{pendingCount} unresolved</>
            ) : (
              <>
                Apply
                <span className="cw-kbd cw-kbd-inline">&#x23CE;</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Outer shell (scenario picker + window trigger) ───────────────────

export function CorrectionReviewPanel() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [windowOpen, setWindowOpen] = useState(true);

  return (
    <div className="prototype-lab prototype-stack">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Correction Review — Window Mockup</strong>
            <span>
              Simulates the panel that appears when the user presses ROpt-C.
            </span>
          </div>
        </header>
        <div className="control-bar">
          <div className="control-actions">
            {MOCK_SCENARIOS.map((s, i) => (
              <button
                key={i}
                className={scenarioIdx === i ? "primary" : ""}
                onClick={() => {
                  setScenarioIdx(i);
                  setWindowOpen(true);
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
        {!windowOpen && (
          <div className="control-bar">
            <div className="control-actions">
              <button className="primary" onClick={() => setWindowOpen(true)}>
                Open correction window (ROpt-C)
              </button>
            </div>
          </div>
        )}
      </section>

      {windowOpen && (
        <CorrectionWindow
          key={scenarioIdx}
          scenario={MOCK_SCENARIOS[scenarioIdx].data}
          onClose={() => setWindowOpen(false)}
        />
      )}
    </div>
  );
}
