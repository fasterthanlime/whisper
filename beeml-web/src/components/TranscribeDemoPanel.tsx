import { useCallback, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { EvalInspector } from "./EvalInspector";
import type { EvalInspectorData, TimedToken } from "../types";
import type { AlignedWord } from "../beeml.generated";

function alignedWordsToTimedTokens(words: AlignedWord[]): TimedToken[] {
  return words.map((word) => ({
    w: word.word,
    s: word.start,
    e: word.end,
  }));
}

function toInspectorData(transcript: string, words: AlignedWord[]): EvalInspectorData {
  const qwenTokens = alignedWordsToTimedTokens(words);
  return {
    transcript,
    transcriptLabel: "BeeML",
    transcriptSource: "transcript",
    qwenAlignment: qwenTokens,
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

        setInspectorData(toInspectorData(result.value.transcript, result.value.words));
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
    <div className="demo-panel">
      <div className="demo-toolbar">
        <button
          className={recorder.state === "recording" ? "danger" : "primary"}
          onClick={handleRecord}
          disabled={recorder.state === "processing"}
        >
          {recorder.state === "recording" ? "STOP" : "RECORD"}
        </button>

        <label className="ws-label">
          <span>ws</span>
          <input
            className="ws-input"
            value={wsUrl}
            onChange={(e) => setWsUrl(e.target.value)}
            placeholder="ws://127.0.0.1:9944"
            disabled={recorder.state === "recording"}
          />
        </label>

        {status && <span className="status">{status}</span>}
        {error && <span className="error">{error}</span>}
      </div>

      {inspectorData ? (
        <EvalInspector data={inspectorData} audioUrl={audioUrl} />
      ) : (
        <div className="demo-empty">
          {recorder.state === "recording"
            ? "Recording... click STOP to transcribe"
            : "Press RECORD to capture audio and transcribe with BeeML"}
        </div>
      )}
    </div>
  );
}
