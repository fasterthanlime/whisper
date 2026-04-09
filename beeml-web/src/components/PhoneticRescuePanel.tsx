import { useEffect, useMemo, useRef, useState } from "react";

import { connectBeeMl } from "../beeml.generated";

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

function controlGlyph(
  controlState:
    | {
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null,
  kind: "original" | "transcript" | "zipa",
) {
  if (controlState?.kind !== kind) return "▶";
  return controlState.phase === "loading" ? "…" : "■";
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function controlStateClass(
  controlState:
    | {
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null,
  kind: "original" | "transcript" | "zipa",
) {
  if (controlState?.kind !== kind) return "";
  return controlState.phase === "loading" ? " is-loading" : " is-playing";
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
    <div
      className="alignment-lane"
      style={{ display: "grid", gridTemplateColumns: "78px minmax(0, 1fr)", gap: "0.65rem" }}
    >
      <div className="alignment-lane-label" style={{ color, fontWeight: 700, paddingTop: "0.3rem" }}>
        {label}
      </div>
      <div
        className="alignment-lane-boxes"
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
      className="alignment-view"
      style={{
        display: "grid",
        gap: "0.45rem",
        marginTop: "0.5rem",
      }}
    >
      <div className="alignment-view-scroll" style={{ overflowX: "auto", paddingBottom: "0.15rem" }}>
        <div className="alignment-view-inner" style={{ width: "max-content", minWidth: "100%" }}>
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

function WordGroup({
  word,
  label,
  onPlayOriginal,
  onPlayTranscript,
  onPlayZipa,
  controlState,
}: {
  word: PhoneticWordAlignment;
  label: string;
  onPlayOriginal: () => void;
  onPlayTranscript: () => void;
  onPlayZipa: () => void;
  controlState:
    | {
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null;
}) {
  const transcript = alignmentRowValues(word.alignment, "transcript");
  const zipa = alignmentRowValues(word.alignment, "zipa");
  const [copiedLane, setCopiedLane] = useState<"transcript" | "zipa" | null>(null);

  const copyLane = async (kind: "transcript" | "zipa") => {
    const values = kind === "transcript" ? transcript : zipa;
    const text = values
      .map((value) => value.text)
      .filter((value) => value !== "∅")
      .join(" ");
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedLane(kind);
      window.setTimeout(() => {
        setCopiedLane((current) => (current === kind ? null : current));
      }, 900);
    } catch {
      // Ignore clipboard failures silently; the button title already hints at the action.
    }
  };

  return (
    <div
      className={`word-group-card${controlState ? " is-selected" : ""}`}
      title={`${word.tokenStart}:${word.tokenEnd} · ZIPA ${word.zipaNormStart}:${word.zipaNormEnd}`}
    >
      <div className="word-group-inline-lanes">
        <div className="word-group-inline-lane">
          <button
            type="button"
            className={`word-group-inline-play-button${controlStateClass(controlState, "original")}`}
            onClick={onPlayOriginal}
            title={`Play original audio for ${label}`}
            aria-label={`Play original audio for ${label}`}
          >
            {controlGlyph(controlState, "original")}
          </button>
          <div className="word-group-inline-box-row">
            <span className="word-group-text-box">{label}</span>
          </div>
        </div>
        <div className="word-group-inline-lane">
          <button
            type="button"
            className={`word-group-inline-play-button${controlStateClass(controlState, "transcript")}`}
            onClick={onPlayTranscript}
            title={`Speak transcript IPA for ${label}`}
            aria-label={`Speak transcript IPA for ${label}`}
          >
            {controlGlyph(controlState, "transcript")}
          </button>
          <button
            type="button"
            className={`word-group-inline-box-row word-group-copy-row${copiedLane === "transcript" ? " is-copied" : ""}`}
            onClick={() => void copyLane("transcript")}
            title={`Copy transcript IPA for ${label}`}
            aria-label={`Copy transcript IPA for ${label}`}
          >
            {transcript.map((value, index) => (
              <span
                key={`t:${word.tokenStart}:${index}`}
                className={`ipa-token-box op-${value.kind.toLowerCase()}${value.text === "∅" ? " is-gap" : ""}`}
                title={`Transcript · ${value.kind} · cost ${value.cost.toFixed(3)}`}
              >
                {value.text}
              </span>
            ))}
          </button>
        </div>
        <div className="word-group-inline-lane">
          <button
            type="button"
            className={`word-group-inline-play-button${controlStateClass(controlState, "zipa")}`}
            onClick={onPlayZipa}
            title={`Speak ZIPA IPA for ${label}`}
            aria-label={`Speak ZIPA IPA for ${label}`}
          >
            {controlGlyph(controlState, "zipa")}
          </button>
          <button
            type="button"
            className={`word-group-inline-box-row word-group-copy-row${copiedLane === "zipa" ? " is-copied" : ""}`}
            onClick={() => void copyLane("zipa")}
            title={`Copy ZIPA IPA for ${label}`}
            aria-label={`Copy ZIPA IPA for ${label}`}
          >
            {zipa.map((value, index) => (
              <span
                key={`z:${word.tokenStart}:${index}`}
                className={`ipa-token-box op-${value.kind.toLowerCase()}${value.text === "∅" ? " is-gap" : ""}`}
                title={`ZIPA · ${value.kind} · cost ${value.cost.toFixed(3)}`}
              >
                {value.text}
              </span>
            ))}
          </button>
        </div>
      </div>
    </div>
  );
}

function GlobalLaneButton({
  label,
  kind,
  controlState,
  onClick,
}: {
  label: string;
  kind: "original" | "transcript" | "zipa";
  controlState:
    | {
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`word-group-inline-play-button global-lane-play-button${controlStateClass(controlState, kind)}`}
      onClick={onClick}
      title={`Play all ${label}`}
      aria-label={`Play all ${label}`}
    >
      <span className="global-lane-play-glyph">{controlGlyph(controlState, kind)}</span>
      <span>{label}</span>
    </button>
  );
}

async function decodeWavBytes(
  context: AudioContext,
  bytes: Uint8Array,
): Promise<AudioBuffer> {
  const copy = Uint8Array.from(bytes);
  return await context.decodeAudioData(copy.buffer);
}

export function PhoneticRescuePanel({
  trace,
  wsUrl,
  sourceAudioUrl,
  sourceAudioPath,
}: {
  trace: PhoneticRescueTrace;
  wsUrl?: string;
  sourceAudioUrl?: string;
  sourceAudioPath?: string;
}) {
  const wordAlignments = useMemo(
    () => [...trace.wordAlignments].sort((a, b) => a.tokenStart - b.tokenStart),
    [trace.wordAlignments],
  );
  const wordGroups = useMemo(
    () =>
      wordAlignments.map((word) => ({
        word,
        label: word.wordText,
      })),
    [wordAlignments],
  );
  const [activeControl, setActiveControl] = useState<
    | {
        tokenStart: number;
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null
  >(null);
  const [activeGlobalControl, setActiveGlobalControl] = useState<
    | {
        kind: "original" | "transcript" | "zipa";
        phase: "loading" | "playing";
      }
    | null
  >(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playingSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const originalBufferRef = useRef<AudioBuffer | null>(null);
  const originalBufferKeyRef = useRef<string | null>(null);
  const playbackRunIdRef = useRef(0);
  const synthBufferCacheRef = useRef<Map<string, Promise<AudioBuffer>>>(new Map());

  useEffect(() => {
    originalBufferRef.current = null;
    originalBufferKeyRef.current = null;
  }, [sourceAudioUrl, sourceAudioPath]);

  useEffect(() => {
    synthBufferCacheRef.current.clear();
  }, [wsUrl]);

  useEffect(
    () => () => {
      playingSourceRef.current?.stop();
      playingSourceRef.current = null;
      audioContextRef.current?.close().catch(() => {});
      audioContextRef.current = null;
    },
    [],
  );

  const ensureAudioContext = async () => {
    const existing = audioContextRef.current;
    if (existing) {
      if (existing.state === "suspended") await existing.resume();
      return existing;
    }
    const next = new AudioContext();
    audioContextRef.current = next;
    return next;
  };

  const stopCurrentSource = () => {
    const source = playingSourceRef.current;
    if (source) {
      source.onended = null;
      source.stop();
      playingSourceRef.current = null;
    }
  };

  const stopPlayback = () => {
    playbackRunIdRef.current += 1;
    stopCurrentSource();
    setActiveControl(null);
    setActiveGlobalControl(null);
  };

  const playBufferRange = async (
    buffer: AudioBuffer,
    startSec: number,
    endSec: number,
    tokenStart: number,
    kind: "original" | "transcript" | "zipa",
  ) => {
    const context = await ensureAudioContext();
    stopCurrentSource();
    const runId = playbackRunIdRef.current;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const safeStart = Math.max(0, Math.min(startSec, buffer.duration));
    const safeEnd = Math.max(safeStart, Math.min(endSec, buffer.duration));
    const duration = Math.max(0.01, safeEnd - safeStart);
    playingSourceRef.current = source;
    setActiveControl({ tokenStart, kind, phase: "playing" });
    await new Promise<void>((resolve) => {
      source.onended = () => {
        if (playingSourceRef.current === source) {
          playingSourceRef.current = null;
        }
        if (playbackRunIdRef.current === runId) {
          setActiveControl(null);
        }
        resolve();
      };
      source.start(0, safeStart, duration);
    });
  };

  const ensureOriginalBuffer = async () => {
    const key = sourceAudioPath ?? sourceAudioUrl ?? null;
    if (!key) {
      throw new Error("original audio is unavailable for this trace");
    }
    if (originalBufferRef.current && originalBufferKeyRef.current === key) {
      return originalBufferRef.current;
    }
    const context = await ensureAudioContext();
    let wavBytes: Uint8Array;
    if (sourceAudioUrl) {
      const response = await fetch(sourceAudioUrl);
      if (!response.ok) {
        throw new Error(`failed to load original audio: ${response.status}`);
      }
      wavBytes = new Uint8Array(await response.arrayBuffer());
    } else {
      if (!wsUrl || !sourceAudioPath) {
        throw new Error("original audio is unavailable for this trace");
      }
      const client = await connectBeeMl(wsUrl);
      const response = await client.loadAudioFile({ path: sourceAudioPath });
      if (!response.ok) throw new Error(response.error);
      wavBytes = new Uint8Array(response.value.wav_bytes);
    }
    const buffer = await decodeWavBytes(context, wavBytes);
    originalBufferRef.current = buffer;
    originalBufferKeyRef.current = key;
    return buffer;
  };

  const synthCacheKey = (kind: "transcript" | "zipa", phonemes: string[]) =>
    `${kind}|af_sarah|1|${phonemes.join(" ")}`;

  const synthWordBuffer = async (
    kind: "transcript" | "zipa",
    phonemes: string[],
  ) => {
    if (!wsUrl) throw new Error("phoneme synthesis is unavailable for this trace");
    const client = await connectBeeMl(wsUrl);
    const response = await client.synthesizePhonemes({
      phonemes: phonemes.join(" "),
      voice: "af_sarah",
      speed: 1,
    });
    if (!response.ok) throw new Error(response.error);
    const context = await ensureAudioContext();
    return await decodeWavBytes(context, new Uint8Array(response.value.wav_bytes));
  };

  const getSynthBufferCached = (
    kind: "transcript" | "zipa",
    phonemes: string[],
  ) => {
    const key = synthCacheKey(kind, phonemes);
    const existing = synthBufferCacheRef.current.get(key);
    if (existing) return existing;
    const promise = synthWordBuffer(kind, phonemes).catch((error) => {
      synthBufferCacheRef.current.delete(key);
      throw error;
    });
    synthBufferCacheRef.current.set(key, promise);
    return promise;
  };

  const synthWord = async (
    word: PhoneticWordAlignment,
    kind: "transcript" | "zipa",
    phonemes: string[],
  ) => {
    if (!wsUrl) return;
    try {
      setControlError(null);
      setActiveGlobalControl(null);
      setActiveControl({ tokenStart: word.tokenStart, kind, phase: "loading" });
      const buffer = await getSynthBufferCached(kind, phonemes);
      await playBufferRange(buffer, 0, buffer.duration, word.tokenStart, kind);
    } catch (error) {
      setControlError(error instanceof Error ? error.message : String(error));
      setActiveControl(null);
    }
  };

  const playOriginalWord = async (word: PhoneticWordAlignment) => {
    try {
      setControlError(null);
      setActiveGlobalControl(null);
      setActiveControl({ tokenStart: word.tokenStart, kind: "original", phase: "loading" });
      const buffer = await ensureOriginalBuffer();
      await playBufferRange(buffer, word.startSec, word.endSec, word.tokenStart, "original");
    } catch (error) {
      setControlError(error instanceof Error ? error.message : String(error));
      setActiveControl(null);
    }
  };

  const playLaneSequence = async (kind: "original" | "transcript" | "zipa") => {
    try {
      setControlError(null);
      stopPlayback();
      const runId = playbackRunIdRef.current;
      setActiveGlobalControl({ kind, phase: "loading" });
      let originalBuffer: AudioBuffer | null = null;
      if (kind === "original") {
        originalBuffer = await ensureOriginalBuffer();
      }
      if (playbackRunIdRef.current !== runId) return;
      setActiveGlobalControl({ kind, phase: "playing" });
      for (const word of wordAlignments) {
        if (playbackRunIdRef.current !== runId) return;
        if (kind === "original") {
          await playBufferRange(originalBuffer!, word.startSec, word.endSec, word.tokenStart, kind);
        } else {
          const buffer = await getSynthBufferCached(
            kind,
            kind === "transcript" ? word.transcriptNormalized : word.zipaNormalized,
          );
          if (playbackRunIdRef.current !== runId) return;
          await playBufferRange(buffer, 0, buffer.duration, word.tokenStart, kind);
        }
        if (playbackRunIdRef.current !== runId) return;
        setActiveControl(null);
        await sleep(140);
      }
      if (playbackRunIdRef.current === runId) {
        setActiveGlobalControl(null);
      }
    } catch (error) {
      setControlError(error instanceof Error ? error.message : String(error));
      setActiveControl(null);
      setActiveGlobalControl(null);
    }
  };

  useEffect(() => {
    if (!wsUrl || wordAlignments.length === 0) return;
    void Promise.allSettled([
      ...wordAlignments.map((word) => getSynthBufferCached("transcript", word.transcriptNormalized)),
      ...wordAlignments.map((word) => getSynthBufferCached("zipa", word.zipaNormalized)),
    ]);
  }, [wsUrl, wordAlignments]);

  return (
    <section className="prototype-card" style={{ marginTop: "0.75rem" }}>
      {wordAlignments.length > 0 ? (
        <div className="word-workbench">
          <div className="phonetic-debug-toolbar">
            <details className="phonetic-debug-details phonetic-debug-details-inline">
              <summary>Utterance alignment</summary>
              <div className="prototype-stack phonetic-panel-summary" style={{ gap: "0.35rem", marginTop: "0.55rem" }}>
                <div className="token-row muted phonetic-panel-summary-row" style={{ userSelect: "text" }}>
                  aligned transcript: {trace.alignedTranscript || "∅"}
                </div>
                {trace.pendingText ? (
                  <div className="token-row muted phonetic-panel-summary-row" style={{ userSelect: "text" }}>
                    pending tail: {trace.pendingText}
                  </div>
                ) : null}
              </div>
              <AlignmentView
                ops={trace.utteranceAlignment}
                transcriptLabel="Transcript"
                zipaLabel="ZIPA"
              />
            </details>

            <details className="phonetic-debug-details phonetic-debug-details-inline">
              <summary>Utterance diagnostics</summary>
              <div className="prototype-summary phonetic-debug-metrics" style={{ marginTop: "0.55rem" }}>
                <span>rev {trace.snapshotRevision.toString()}</span>
                <span>utterance norm {formatMetric(trace.utteranceFeatureSimilarity)}</span>
                <span>utterance raw {formatMetric(trace.utteranceSimilarity)}</span>
                <span>tail volatile {trace.tailAmbiguity.volatileTokenCount}</span>
              </div>
              <div className="prototype-stack phonetic-debug-token-stack" style={{ gap: "0.35rem", marginTop: "0.55rem" }}>
                <div className="token-row muted phonetic-debug-token-row" style={{ userSelect: "text" }}>
                  transcript norm: {formatTokens(trace.utteranceTranscriptNormalized)}
                </div>
                <div className="token-row muted phonetic-debug-token-row" style={{ userSelect: "text" }}>
                  ZIPA raw: {formatTokens(trace.utteranceZipaRaw)}
                </div>
                <div className="token-row muted phonetic-debug-token-row" style={{ userSelect: "text" }}>
                  ZIPA norm: {formatTokens(trace.utteranceZipaNormalized)}
                </div>
              </div>
            </details>
          </div>

          <div className="global-lane-toolbar">
            <GlobalLaneButton
              label="Original"
              kind="original"
              controlState={activeGlobalControl}
              onClick={() => void playLaneSequence("original")}
            />
            <GlobalLaneButton
              label="Transcript IPA"
              kind="transcript"
              controlState={activeGlobalControl}
              onClick={() => void playLaneSequence("transcript")}
            />
            <GlobalLaneButton
              label="ZIPA"
              kind="zipa"
              controlState={activeGlobalControl}
              onClick={() => void playLaneSequence("zipa")}
            />
          </div>

          {controlError && (
            <div className="notice-row">
              <span className="error-pill">{controlError}</span>
            </div>
          )}
          <div className="word-group-strip">
            {wordGroups.map(({ word, label }) => (
              <WordGroup
                key={`${word.tokenStart}:${word.tokenEnd}:${word.wordText}`}
                word={word}
                label={label}
                onPlayOriginal={() => void playOriginalWord(word)}
                onPlayTranscript={() => void synthWord(word, "transcript", word.transcriptNormalized)}
                onPlayZipa={() => void synthWord(word, "zipa", word.zipaNormalized)}
                controlState={
                  activeControl?.tokenStart === word.tokenStart
                    ? { kind: activeControl.kind, phase: activeControl.phase }
                    : null
                }
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="prototype-empty phonetic-panel-empty" style={{ marginTop: "0.75rem" }}>
          No word-level alignment slices available in this transcription.
        </div>
      )}
    </section>
  );
}
