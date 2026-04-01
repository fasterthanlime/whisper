import { useState, useCallback, useRef, useEffect } from "react";
import type { EvalInspectorData, BakeoffResult, BakeoffEntry } from "../types";
import {
  startBakeoff,
  getJob,
  parseBakeoffResult,
  bakeoffDetail,
  recordingAudioUrl,
} from "../api";
import { EvalInspector } from "./EvalInspector";

const DEFAULT_TRAIN_ID = 262;
const DEFAULT_LIMIT = 5;
const POLL_INTERVAL = 2000;

export function HumanEvalPanel() {
  const [trainId] = useState<number>(DEFAULT_TRAIN_ID);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [caseIdsInput, setCaseIdsInput] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BakeoffResult | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<BakeoffEntry | null>(null);
  const [inspectorData, setInspectorData] = useState<EvalInspectorData | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleSelectEntry = useCallback(
    async (entry: BakeoffEntry) => {
      setSelectedEntry(entry);
      setLoadingDetail(true);
      try {
        const data = await bakeoffDetail({
          recordingId: entry.recordingId,
          transcript: entry.transcript,
          expected: entry.expected,
          prototype: entry.prototype,
          trainId,
        });
        data.expected = entry.expected;
        setInspectorData(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoadingDetail(false);
      }
    },
    [trainId],
  );

  const handleRun = useCallback(async () => {
    try {
      setError(null);
      setResult(null);
      setSelectedEntry(null);
      setInspectorData(null);
      setStatus("Starting bakeoff...");

      const caseIds = caseIdsInput
        .split(/[,\s]+/)
        .map((value) => value.trim())
        .filter(Boolean);

      const { jobId } = await startBakeoff({
        limit,
        trainId,
        caseIds: caseIds.length ? caseIds : undefined,
        randomize: caseIds.length ? false : true,
      });
      setStatus(`Job #${jobId} running...`);

      // Poll
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(jobId);
          if (job.status === "completed") {
            clearInterval(pollRef.current);
            const parsed = parseBakeoffResult(job);
            setResult(parsed);
            setStatus(null);
            // Auto-select first case
            if (parsed?.entries.length) {
              handleSelectEntry(parsed.entries[0]);
            }
          } else if (job.status === "failed") {
            clearInterval(pollRef.current);
            setError(`Job #${jobId} failed`);
            setStatus(null);
          }
          // else still running
        } catch (e) {
          clearInterval(pollRef.current);
          setError(e instanceof Error ? e.message : String(e));
          setStatus(null);
        }
      }, POLL_INTERVAL);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [caseIdsInput, limit, trainId, handleSelectEntry]);

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
      {/* Controls bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "0.75rem 1rem",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface-alt)",
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          train #{trainId}
        </span>

        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.85rem" }}>
          Cases:
          <input
            type="number"
            min={1}
            max={200}
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value) || 5)}
            style={{ width: 60 }}
          />
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.85rem" }}>
          Case IDs:
          <input
            type="text"
            value={caseIdsInput}
            onChange={(e) => setCaseIdsInput(e.target.value)}
            placeholder="hum-183 or hum-183,hum-205"
            style={{ width: 260 }}
          />
        </label>

        <button className="primary" onClick={handleRun} disabled={!!status}>
          EVALUATE
        </button>

        {status && (
          <span style={{ fontSize: "0.8rem", color: "var(--accent)" }}>{status}</span>
        )}
        {error && (
          <span style={{ fontSize: "0.8rem", color: "var(--danger)" }}>{error}</span>
        )}

        {/* Summary cards */}
        {result && (
          <div style={{ display: "flex", gap: "0.75rem", marginLeft: "auto" }}>
            <SummaryBadge label="Total" value={result.summary.n} />
            <SummaryBadge label="Correct" value={result.summary.prototype} color="var(--success)" />
            <SummaryBadge label="Wrong" value={result.summary.prototypeWrong} color="var(--danger)" />
          </div>
        )}
      </div>

      {/* Case selector + inspector */}
      {result && (
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.5rem 1rem", borderBottom: "1px solid var(--border)" }}>
          <label style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Case:</label>
          <select
            value={selectedEntry?.caseId ?? ""}
            onChange={(e) => {
              const entry = result.entries.find((r) => r.caseId === e.target.value);
              if (entry) handleSelectEntry(entry);
            }}
            style={{ minWidth: 300 }}
          >
            <option value="">Select a case...</option>
            {result.entries.map((entry) => (
              <option key={entry.caseId} value={entry.caseId}>
                {entry.prototypeOk ? "✓" : "✗"} {entry.term} ({entry.caseId})
                {entry.hitCount > 1 ? ` · ${entry.hitCount} hits` : ""}
                {!entry.prototypeOk ? ` · ${entry.analysis.failureReason}` : ""}
              </option>
            ))}
          </select>
          {loadingDetail && (
            <span style={{ fontSize: "0.8rem", color: "var(--accent)" }}>Loading detail...</span>
          )}
        </div>
      )}

      {inspectorData ? (
        <EvalInspector
          data={inspectorData}
          audioUrl={selectedEntry ? recordingAudioUrl(selectedEntry.recordingId) : undefined}
        />
      ) : (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-muted)",
          }}
        >
          {result ? "Select a case to inspect" : "Configure and run an evaluation batch"}
        </div>
      )}
    </div>
  );
}

function SummaryBadge({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "0.25rem 0.75rem",
        textAlign: "center",
        minWidth: 70,
      }}
    >
      <div style={{ fontSize: "0.65rem", textTransform: "uppercase", color: "var(--text-muted)" }}>
        {label}
      </div>
      <div style={{ fontSize: "1.2rem", fontWeight: 700, color: color ?? "var(--text)" }}>
        {value}
      </div>
    </div>
  );
}
