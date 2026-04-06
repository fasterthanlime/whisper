import { useState, useCallback } from "react";

// ── Mock data ────────────────────────────────────────────────────────

interface Correction {
  id: number;
  /** Character range in the raw transcript */
  originalStart: number;
  originalEnd: number;
  /** The misheard text */
  originalText: string;
  /** The corrected term */
  replacementText: string;
  /** Canonical term name */
  term: string;
  /** Gate × ranker confidence */
  confidence: number;
  /** User accept/reject state */
  accepted: boolean;
}

interface InsertionRecord {
  rawTranscript: string;
  correctedText: string;
  corrections: Correction[];
}

const MOCK_SCENARIOS: { label: string; data: InsertionRecord }[] = [
  {
    label: "3 corrections (serde_json, tokio, JSON)",
    data: {
      rawTranscript:
        "I was working with Sir Day Jason and Tokyo to parse the Jason file",
      correctedText:
        "I was working with serde_json and tokio to parse the JSON file",
      corrections: [
        {
          id: 1,
          originalStart: 22,
          originalEnd: 36,
          originalText: "Sir Day Jason",
          replacementText: "serde_json",
          term: "serde_json",
          confidence: 0.92,
          accepted: true,
        },
        {
          id: 2,
          originalStart: 41,
          originalEnd: 46,
          originalText: "Tokyo",
          replacementText: "tokio",
          term: "tokio",
          confidence: 0.87,
          accepted: true,
        },
        {
          id: 3,
          originalStart: 61,
          originalEnd: 66,
          originalText: "Jason",
          replacementText: "JSON",
          term: "JSON",
          confidence: 0.95,
          accepted: true,
        },
      ],
    },
  },
  {
    label: "1 correction (kubectl)",
    data: {
      rawTranscript: "then I ran cube cuddle to deploy the pods",
      correctedText: "then I ran kubectl to deploy the pods",
      corrections: [
        {
          id: 1,
          originalStart: 11,
          originalEnd: 23,
          originalText: "cube cuddle",
          replacementText: "kubectl",
          term: "kubectl",
          confidence: 0.89,
          accepted: true,
        },
      ],
    },
  },
  {
    label: "2 corrections, one wrong (AArch64, bearcove)",
    data: {
      rawTranscript:
        "we need to target arch 64 and deploy to bear cove",
      correctedText:
        "we need to target AArch64 and deploy to bearcove",
      corrections: [
        {
          id: 1,
          originalStart: 23,
          originalEnd: 30,
          originalText: "arch 64",
          replacementText: "AArch64",
          term: "AArch64",
          confidence: 0.78,
          accepted: true,
        },
        {
          id: 2,
          originalStart: 45,
          originalEnd: 54,
          originalText: "bear cove",
          replacementText: "bearcove",
          term: "bearcove",
          confidence: 0.71,
          accepted: true,
        },
      ],
    },
  },
  {
    label: "No corrections",
    data: {
      rawTranscript: "the weather is nice today",
      correctedText: "the weather is nice today",
      corrections: [],
    },
  },
];

// ── Components ───────────────────────────────────────────────────────

/** Renders the sentence with inline correction annotations */
function AnnotatedSentence({
  record,
}: {
  record: InsertionRecord;
}) {
  const { rawTranscript, corrections } = record;

  // Build segments: alternating plain text and correction spans
  const segments: React.ReactNode[] = [];
  let cursor = 0;

  const sorted = [...corrections].sort(
    (a, b) => a.originalStart - b.originalStart
  );

  for (const c of sorted) {
    // Plain text before this correction
    if (cursor < c.originalStart) {
      segments.push(
        <span key={`plain-${cursor}`} className="cr-plain">
          {rawTranscript.slice(cursor, c.originalStart)}
        </span>
      );
    }

    // The correction span
    segments.push(
      <span
        key={`correction-${c.id}`}
        className={`cr-correction ${c.accepted ? "cr-accepted" : "cr-rejected"}`}
      >
        <span className="cr-original">{c.originalText}</span>
        <span className="cr-arrow"> → </span>
        <span className="cr-replacement">{c.replacementText}</span>
      </span>
    );

    cursor = c.originalEnd;
  }

  // Trailing text
  if (cursor < rawTranscript.length) {
    segments.push(
      <span key={`plain-${cursor}`} className="cr-plain">
        {rawTranscript.slice(cursor)}
      </span>
    );
  }

  return <div className="cr-sentence">{segments}</div>;
}

/** One row in the correction list */
function CorrectionRow({
  correction,
  onToggle,
}: {
  correction: Correction;
  onToggle: (id: number) => void;
}) {
  return (
    <label className={`cr-row ${correction.accepted ? "" : "cr-row-rejected"}`}>
      <input
        type="checkbox"
        checked={correction.accepted}
        onChange={() => onToggle(correction.id)}
      />
      <span className="cr-row-replacement">{correction.replacementText}</span>
      <span className="cr-row-original">was: "{correction.originalText}"</span>
      <span className="cr-row-confidence">
        {(correction.confidence * 100).toFixed(0)}%
      </span>
    </label>
  );
}

/** The "what would be inserted" preview */
function ResultPreview({ record }: { record: InsertionRecord }) {
  const { rawTranscript, corrections } = record;

  // Build the final text by applying accepted corrections
  const sorted = [...corrections].sort(
    (a, b) => b.originalStart - a.originalStart // reverse order for safe replacement
  );
  let result = rawTranscript;
  for (const c of sorted) {
    if (c.accepted) {
      result =
        result.slice(0, c.originalStart) +
        c.replacementText +
        result.slice(c.originalEnd);
    }
  }

  return (
    <div className="cr-preview">
      <span className="cr-preview-label">Result:</span>
      <span className="cr-preview-text">{result}</span>
    </div>
  );
}

// ── Main panel ───────────────────────────────────────────────────────

export function CorrectionReviewPanel() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const scenario = MOCK_SCENARIOS[scenarioIdx];

  // Deep clone corrections so we can toggle accept/reject
  const [corrections, setCorrections] = useState<Correction[]>(
    () => scenario.data.corrections.map((c) => ({ ...c }))
  );

  const [dismissed, setDismissed] = useState(false);
  const [applied, setApplied] = useState(false);

  const switchScenario = useCallback(
    (idx: number) => {
      setScenarioIdx(idx);
      setCorrections(
        MOCK_SCENARIOS[idx].data.corrections.map((c) => ({ ...c }))
      );
      setDismissed(false);
      setApplied(false);
    },
    []
  );

  const toggleCorrection = useCallback((id: number) => {
    setCorrections((prev) =>
      prev.map((c) => (c.id === id ? { ...c, accepted: !c.accepted } : c))
    );
  }, []);

  const record: InsertionRecord = {
    ...scenario.data,
    corrections,
  };

  const acceptedCount = corrections.filter((c) => c.accepted).length;

  return (
    <div className="prototype-lab prototype-stack">
      {/* Scenario picker */}
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Correction Review — UI Prototype</strong>
            <span>
              Mock data. Simulates the ROpt-C panel a user would see after
              dictating.
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

      {/* The actual correction review panel */}
      {!dismissed && !applied ? (
        <section className="prototype-card cr-panel">
          {corrections.length === 0 ? (
            <div className="cr-empty">
              <p>No corrections were applied.</p>
              <p className="text-dim">
                The transcript was inserted as-is. If you think a term should
                have been corrected, you can add it to your vocabulary.
              </p>
              <div className="cr-actions">
                <button onClick={() => setDismissed(true)}>Dismiss</button>
              </div>
            </div>
          ) : (
            <>
              {/* Notification bar */}
              <div className="cr-notification">
                <span className="cr-notification-count">
                  {acceptedCount} correction{acceptedCount !== 1 ? "s" : ""}{" "}
                  applied
                </span>
                <span className="cr-notification-hint">
                  Uncheck to revert individual corrections
                </span>
              </div>

              {/* Annotated sentence */}
              <AnnotatedSentence record={record} />

              {/* Correction list */}
              <div className="cr-list">
                {corrections.map((c) => (
                  <CorrectionRow
                    key={c.id}
                    correction={c}
                    onToggle={toggleCorrection}
                  />
                ))}
              </div>

              {/* Result preview */}
              <ResultPreview record={record} />

              {/* Actions */}
              <div className="cr-actions">
                <button
                  className="primary"
                  onClick={() => setApplied(true)}
                >
                  Apply
                </button>
                <button
                  onClick={() => {
                    setCorrections((prev) =>
                      prev.map((c) => ({ ...c, accepted: false }))
                    );
                  }}
                >
                  Revert All
                </button>
                <button onClick={() => setDismissed(true)}>Dismiss</button>
              </div>
            </>
          )}
        </section>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="cr-result-message">
            {applied ? (
              <>
                <strong>Applied.</strong>{" "}
                {acceptedCount} correction{acceptedCount !== 1 ? "s" : ""} kept,{" "}
                {corrections.length - acceptedCount} reverted.
                <span className="text-dim">
                  {" "}
                  Teaching signals would be sent to the judge.
                </span>
              </>
            ) : (
              <>
                <strong>Dismissed.</strong>{" "}
                <span className="text-dim">
                  Corrections remain as-is in the text.
                </span>
              </>
            )}
            <button
              style={{ marginLeft: "1rem" }}
              onClick={() => {
                setDismissed(false);
                setApplied(false);
              }}
            >
              Reset
            </button>
          </div>
        </section>
      )}

      {/* Context: what the IME would do */}
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>IME behavior notes</strong>
          </div>
        </header>
        <div className="cr-notes">
          <ul>
            <li>
              <strong>Insert:</strong> IME calls{" "}
              <code>insertText(correctedText, replacementRange: NSNotFound)</code>{" "}
              — user sees corrected text immediately.
            </li>
            <li>
              <strong>Track:</strong> IME saves the insertion range (cursor pos +
              length) and the raw transcript.
            </li>
            <li>
              <strong>Review:</strong> ROpt-C opens this panel. User toggles
              corrections.
            </li>
            <li>
              <strong>Apply:</strong> IME calls{" "}
              <code>
                insertText(finalText, replacementRange: savedRange)
              </code>{" "}
              to replace the inserted text with the user's chosen mix.
            </li>
            <li>
              <strong>Teach:</strong> Each accepted/rejected correction becomes a
              training signal for the two-stage judge.
            </li>
            <li>
              <strong>Invalidate:</strong> If the user moves the cursor or edits
              the text before reviewing, the insertion record is invalidated.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
