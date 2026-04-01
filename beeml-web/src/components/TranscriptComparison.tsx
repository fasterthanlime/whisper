import type { AcceptedEdit } from "../types";

function DiffLine({ label, text, color }: { label: string; text: string; color?: string }) {
  return (
    <div style={{ display: "flex", gap: "1rem", marginBottom: "0.5rem", alignItems: "baseline" }}>
      <span
        style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          color: color ?? "var(--text-muted)",
          textTransform: "uppercase",
          width: 80,
          flexShrink: 0,
          textAlign: "right",
        }}
      >
        {label}
      </span>
      <span style={{ fontFamily: "system-ui", fontSize: "0.95rem" }}>{text || "—"}</span>
    </div>
  );
}

export function TranscriptComparison({
  transcriptLabel,
  transcript,
  expected,
  corrected,
  accepted,
}: {
  transcriptLabel: string;
  transcript: string;
  expected?: string;
  corrected: string;
  accepted: AcceptedEdit[];
}) {
  return (
    <div>
      <DiffLine label={transcriptLabel} text={transcript} />
      {expected && <DiffLine label="Expected" text={expected} />}
      <DiffLine label="Reranker" text={corrected} color="var(--warning)" />

      {accepted.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", marginBottom: "0.5rem" }}>
            ACCEPTED EDITS
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            {accepted.map((edit, i) => (
              <span
                key={i}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.3rem",
                  padding: "0.25em 0.6em",
                  borderRadius: 6,
                  background: "var(--bg-surface-alt)",
                  border: "1px solid var(--border)",
                  fontSize: "0.85rem",
                }}
              >
                <span style={{ textDecoration: "line-through", opacity: 0.6 }}>{edit.original}</span>
                <span>→</span>
                <span style={{ fontWeight: 600 }}>{edit.replacement}</span>
                {edit.score != null && (
                  <span style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>
                    {edit.score.toFixed(2)}
                  </span>
                )}
                {edit.delta != null && (
                  <span style={{ fontSize: "0.7rem", color: edit.delta < 0 ? "var(--danger)" : "var(--success)" }}>
                    Δ {edit.delta > 0 ? "+" : ""}{edit.delta.toFixed(2)}
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
