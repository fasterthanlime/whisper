import { useState, useCallback } from "react";
import type { EvalInspectorData } from "../types";
import { asrDual, correctPrototype } from "../api";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { EvalInspector } from "./EvalInspector";

const DEFAULT_TRAIN_ID = 262;

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      // strip the data:...;base64, prefix
      const result = reader.result as string;
      resolve(result.split(",", 2)[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export function LiveEvalPanel() {
  const recorder = useAudioRecorder();
  const [trainId] = useState(DEFAULT_TRAIN_ID);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inspectorData, setInspectorData] = useState<EvalInspectorData | null>(null);
  const [audioObjectUrl, setAudioObjectUrl] = useState<string | undefined>(undefined);

  const handleRecord = useCallback(async () => {
    if (recorder.state === "recording") {
      setStatus("Stopping recording...");
      const blob = await recorder.stop();

      // Create object URL for playback
      const url = URL.createObjectURL(blob);
      setAudioObjectUrl(url);

      try {
        setStatus("Running ASR...");
        setError(null);
        const asr = await asrDual(blob);
        console.log("parakeet_alignment raw:", JSON.stringify(asr.parakeet_alignment, null, 2));

        setStatus("Running correction...");
        // Send audio as base64 so the server can run ZIPA alignment
        const wavBase64 = await blobToBase64(blob);
        const data = await correctPrototype({
          transcript: asr.parakeet,
          audioWavBase64: wavBase64,
          trainId,
        });

        // Patch in the parakeet alignment from ASR
        data.parakeetAlignment = asr.parakeet_alignment;
        data.transcriptLabel = "Parakeet";
        console.log("reranker:", data.prototype.reranker);
        console.log("sentenceCandidates:", data.prototype.sentenceCandidates);
        console.log("alignments.transcript:", data.alignments.transcript);

        setInspectorData(data);
        setStatus(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      }
    } else {
      setError(null);
      setInspectorData(null);
      if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
      setAudioObjectUrl(undefined);
      await recorder.start();
    }
  }, [recorder, trainId, audioObjectUrl]);

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
        }}
      >
        <button
          className={recorder.state === "recording" ? "danger" : "primary"}
          onClick={handleRecord}
          disabled={recorder.state === "processing"}
        >
          {recorder.state === "recording" ? "STOP" : "RECORD"}
        </button>

        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          train #{trainId}
        </span>

        {status && (
          <span style={{ fontSize: "0.8rem", color: "var(--accent)" }}>{status}</span>
        )}
        {error && (
          <span style={{ fontSize: "0.8rem", color: "var(--danger)" }}>{error}</span>
        )}
      </div>

      {/* Inspector or empty state */}
      {inspectorData ? (
        <EvalInspector data={inspectorData} audioUrl={audioObjectUrl} />
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
          {recorder.state === "recording" ? (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>🎤</div>
              <div>Recording... click STOP when done</div>
            </div>
          ) : (
            <div>Press RECORD to start</div>
          )}
        </div>
      )}
    </div>
  );
}
