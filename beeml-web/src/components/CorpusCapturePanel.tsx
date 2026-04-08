import { useCallback, useEffect, useMemo, useState } from "react";
import { connectBeeMl } from "../beeml.generated";
import type {
  CorpusCapturePlanResult,
  CorpusCapturePrompt,
  CorpusCaptureRecording,
} from "../beeml.generated";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

function byPromptId(recordings: CorpusCaptureRecording[]) {
  const map = new Map<string, CorpusCaptureRecording[]>();
  for (const recording of recordings) {
    const rows = map.get(recording.prompt_id) ?? [];
    rows.push(recording);
    map.set(recording.prompt_id, rows);
  }
  for (const rows of map.values()) {
    rows.sort((a, b) => Number(b.take - a.take || b.created_at_unix_ms - a.created_at_unix_ms));
  }
  return map;
}

function formatTimestamp(unixMs: bigint) {
  return new Date(Number(unixMs)).toLocaleString();
}

export function CorpusCapturePanel({ wsUrl }: { wsUrl: string }) {
  const recorder = useAudioRecorder();
  const [plan, setPlan] = useState<CorpusCapturePlanResult | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const loadPlan = useCallback(async () => {
    const client = await connectBeeMl(wsUrl);
    const result = await client.getCorpusCapturePlan();
    if (!result.ok) throw new Error(result.error);
    setPlan(result.value);
    return result.value;
  }, [wsUrl]);

  useEffect(() => {
    void (async () => {
      try {
        setStatus("Loading capture prompts...");
        setError(null);
        const nextPlan = await loadPlan();
        const recordingsByPrompt = byPromptId(nextPlan.recordings);
        const firstUnrecorded = nextPlan.prompts.findIndex(
          (prompt) => !recordingsByPrompt.has(prompt.prompt_id),
        );
        setCurrentIndex(firstUnrecorded >= 0 ? firstUnrecorded : 0);
        setStatus(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      }
    })();
  }, [loadPlan]);

  const recordingsByPrompt = useMemo(
    () => byPromptId(plan?.recordings ?? []),
    [plan?.recordings],
  );
  const recordedCount = Array.from(recordingsByPrompt.keys()).length;
  const currentPrompt: CorpusCapturePrompt | null = plan?.prompts[currentIndex] ?? null;
  const currentRecordings = currentPrompt
    ? recordingsByPrompt.get(currentPrompt.prompt_id) ?? []
    : [];
  const latestRecording = currentRecordings[0] ?? null;

  const moveToPrompt = useCallback(
    (nextIndex: number) => {
      if (!plan) return;
      setCurrentIndex(Math.max(0, Math.min(plan.prompts.length - 1, nextIndex)));
      setNotes("");
    },
    [plan],
  );

  const moveToNextUnrecorded = useCallback(() => {
    if (!plan) return;
    const nextIndex = plan.prompts.findIndex(
      (prompt, index) => index > currentIndex && !recordingsByPrompt.has(prompt.prompt_id),
    );
    if (nextIndex >= 0) {
      moveToPrompt(nextIndex);
    } else {
      moveToPrompt(Math.min(currentIndex + 1, plan.prompts.length - 1));
    }
  }, [currentIndex, moveToPrompt, plan, recordingsByPrompt]);

  const handleRecord = useCallback(async () => {
    if (!currentPrompt) return;

    if (recorder.state === "recording") {
      try {
        setStatus("Saving recording...");
        setError(null);
        const blob = await recorder.stop();
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
        const nextAudioUrl = URL.createObjectURL(blob);
        setAudioUrl(nextAudioUrl);

        const client = await connectBeeMl(wsUrl);
        const bytes = new Uint8Array(await blob.arrayBuffer());
        const result = await client.saveCorpusRecording({
          prompt_id: currentPrompt.prompt_id,
          ordinal: currentPrompt.ordinal,
          term: currentPrompt.term,
          text: currentPrompt.text,
          wav_bytes: bytes,
          notes: notes.trim() ? notes.trim() : null,
        });
        if (!result.ok) throw new Error(result.error);

        setPlan((prev) =>
          prev
            ? {
                ...prev,
                recordings: [...prev.recordings, result.value.recording],
              }
            : prev,
        );
        setNotes("");
        setStatus(`Saved take ${result.value.recording.take}.`);
        moveToNextUnrecorded();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      }
      return;
    }

    try {
      setError(null);
      setStatus(`Recording prompt ${currentPrompt.ordinal}...`);
      await recorder.start();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }, [audioUrl, currentPrompt, moveToNextUnrecorded, notes, recorder, wsUrl]);

  return (
    <div className="prototype-lab prototype-stack judge-eval-layout">
      <section className="prototype-card prototype-card-tight">
        <header className="panel-header-row">
          <div>
            <strong>Corpus Capture</strong>
            <span>Record a 100-prompt ZIPA alignment corpus, one utterance at a time.</span>
          </div>
          {plan ? (
            <span className="badge">
              {recordedCount} / {plan.prompts.length} prompts recorded
            </span>
          ) : null}
        </header>

        <div className="prototype-summary">
          <span>corpus dir {plan?.corpus_dir ?? "loading..."}</span>
          {currentPrompt ? <span>prompt {currentPrompt.ordinal}</span> : null}
          {latestRecording ? <span>latest take {latestRecording.take}</span> : null}
        </div>

        {(status || error) && (
          <div className="notice-row">
            {status ? <span className="status-pill">{status}</span> : null}
            {error ? <span className="error-pill">{error}</span> : null}
          </div>
        )}
      </section>

      {currentPrompt ? (
        <section className="prototype-card" style={{ gap: "1rem" }}>
          <div className="failure-topline">
            <span className="mini-badge">{currentPrompt.ordinal}</span>
            <span className="mini-badge">{currentPrompt.term}</span>
            <span className="failure-score">{currentPrompt.prompt_id}</span>
          </div>

          <div
            style={{
              fontSize: "1.6rem",
              lineHeight: 1.45,
              fontWeight: 600,
            }}
          >
            {currentPrompt.text}
          </div>

          <div className="control-bar">
            <div className="control-actions">
              <button
                onClick={() => moveToPrompt(currentIndex - 1)}
                disabled={currentIndex <= 0}
              >
                Prev
              </button>
              <button
                className={recorder.state === "recording" ? "danger" : "primary"}
                onClick={() => void handleRecord()}
              >
                {recorder.state === "recording" ? "Stop & Save" : "Record"}
              </button>
              <button
                onClick={() => moveToPrompt(currentIndex + 1)}
                disabled={!plan || currentIndex >= plan.prompts.length - 1}
              >
                Next
              </button>
              <button onClick={moveToNextUnrecorded} disabled={!plan}>
                Next Unrecorded
              </button>
            </div>
          </div>

          <label style={{ display: "grid", gap: "0.4rem" }}>
            <span className="muted">Notes</span>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Optional notes about the take, hesitation, accent, noise..."
              rows={3}
            />
          </label>

          {audioUrl ? (
            <div style={{ display: "grid", gap: "0.5rem" }}>
              <strong>Latest local playback</strong>
              <audio controls src={audioUrl} />
            </div>
          ) : null}

          {latestRecording ? (
            <div className="prototype-summary">
              <span>saved {formatTimestamp(latestRecording.created_at_unix_ms)}</span>
              <span>{latestRecording.num_bytes} bytes</span>
              <span>{latestRecording.wav_path}</span>
            </div>
          ) : (
            <div className="prototype-empty">No saved takes for this prompt yet.</div>
          )}
        </section>
      ) : (
        <section className="prototype-card prototype-card-tight">
          <div className="prototype-empty">No prompts loaded.</div>
        </section>
      )}

      {plan ? (
        <section className="prototype-card">
          <header className="panel-header-row">
            <div>
              <strong>Prompt Queue</strong>
              <span>Jump around, but the intended workflow is one-by-one capture.</span>
            </div>
            <span className="badge">{plan.prompts.length}</span>
          </header>

          <div style={{ display: "grid", gap: "0.55rem" }}>
            {plan.prompts.map((prompt, index) => {
              const saved = recordingsByPrompt.get(prompt.prompt_id) ?? [];
              const latest = saved[0] ?? null;
              const active = index === currentIndex;
              return (
                <button
                  key={prompt.prompt_id}
                  onClick={() => moveToPrompt(index)}
                  style={{
                    textAlign: "left",
                    borderRadius: "0.75rem",
                    border: active ? "1px solid var(--accent)" : "1px solid var(--border)",
                    background: active ? "var(--bg-subtle)" : "var(--bg-elevated)",
                    padding: "0.8rem 0.95rem",
                    display: "grid",
                    gap: "0.35rem",
                  }}
                >
                  <div className="failure-topline">
                    <span className="mini-badge">{prompt.ordinal}</span>
                    <span className="mini-badge">{prompt.term}</span>
                    {latest ? (
                      <span className="failure-score">take {latest.take}</span>
                    ) : (
                      <span className="failure-score">unrecorded</span>
                    )}
                  </div>
                  <div style={{ fontWeight: 600 }}>{prompt.text}</div>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}
    </div>
  );
}
