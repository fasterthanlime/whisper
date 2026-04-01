import { useCallback, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { EvalInspector } from "./EvalInspector";
import type { EvalInspectorData, TimedToken } from "../types";
import type { ForcedAlignItem } from "../beeml.generated";

function qwenWordsToTimedTokens(words: ForcedAlignItem[]): TimedToken[] {
  return words.map((word) => ({
    w: word.word,
    s: word.start_time,
    e: word.end_time,
  }));
}

function toInspectorData(transcript: string, qwenWords: ForcedAlignItem[]): EvalInspectorData {
  const qwenTokens = qwenWordsToTimedTokens(qwenWords);
  return {
    transcript,
    transcriptLabel: "BeeML",
    transcriptSource: "transcript",
    parakeetAlignment: qwenTokens,
    alignments: {
      timingSource: "qwen-forced-aligner",
      transcript: qwenTokens,
    },
    prototype: {
      corrected: transcript,
      accepted: [],
      proposals: [],
      sentenceCandidates: [],
    },
  };
}

export function TranscribeDemoPanel() {
  const recorder = useAudioRecorder();
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:9944");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inspectorData, setInspectorData] = useState<EvalInspectorData | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | undefined>(undefined);

  const handleRecord = useCallback(async () => {
    if (recorder.state === "recording") {
      setStatus("Stopping recording...");
      const blob = await recorder.stop();

      if (audioUrl) URL.revokeObjectURL(audioUrl);
      const nextAudioUrl = URL.createObjectURL(blob);
      setAudioUrl(nextAudioUrl);

      try {
        setStatus("Connecting to BeeML...");
        setError(null);
        const client = await connectBeeMl(wsUrl);

        setStatus("Transcribing...");
        const bytes = new Uint8Array(await blob.arrayBuffer());
        const result = await client.transcribeWav(bytes);
        if (!result.ok) {
          throw new Error(result.error);
        }

        setInspectorData(toInspectorData(result.value.transcript, result.value.qwen_words));
        setStatus(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      }
    } else {
      setError(null);
      setInspectorData(null);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(undefined);
      await recorder.start();
    }
  }, [audioUrl, recorder, wsUrl]);

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          padding: "0.75rem 1rem",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface-alt)",
          flexWrap: "wrap",
        }}
      >
        <button
          className={recorder.state === "recording" ? "danger" : "primary"}
          onClick={handleRecord}
          disabled={recorder.state === "processing"}
        >
          {recorder.state === "recording" ? "STOP" : "RECORD"}
        </button>

        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>ws</span>
          <input
            value={wsUrl}
            onChange={(e) => setWsUrl(e.target.value)}
            placeholder="ws://127.0.0.1:9944"
            disabled={recorder.state === "recording"}
            style={{ minWidth: 280 }}
          />
        </label>

        {status && <span style={{ fontSize: "0.8rem", color: "var(--accent)" }}>{status}</span>}
        {error && <span style={{ fontSize: "0.8rem", color: "var(--danger)" }}>{error}</span>}
      </div>

      {inspectorData ? (
        <EvalInspector data={inspectorData} audioUrl={audioUrl} />
      ) : (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-muted)",
            padding: "1rem",
            textAlign: "center",
          }}
        >
          {recorder.state === "recording"
            ? "Recording... click STOP to transcribe"
            : "Press RECORD to capture audio and transcribe with BeeML"}
        </div>
      )}
    </div>
  );
}
