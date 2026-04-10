import { useEffect, useMemo, useRef, useState } from "react";

import { connectBeeMl } from "../beeml.generated";

import type {
  PhoneticAlignmentOp,
  PhoneticRescueTrace,
  PhoneticWordAlignment,
} from "../types";

type TimelineBar = {
  label: string;
  startSec: number;
  endSec: number;
  title: string;
};

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

function formatSeconds(value: number) {
  return `${value.toFixed(2)}s`;
}

function timelineTicks(duration: number) {
  const count = Math.max(2, Math.ceil(duration));
  return Array.from({ length: count + 1 }, (_, index) => {
    const sec = Math.min(duration, index);
    return { sec, left: duration > 0 ? (sec / duration) * 100 : 0 };
  });
}

function TimelineLane({
  label,
  bars,
  duration,
  tone,
  onPlayRange,
}: {
  label: string;
  bars: TimelineBar[];
  duration: number;
  tone: "qwen" | "zipa" | "phone";
  onPlayRange: (startSec: number, endSec: number) => void;
}) {
  return (
    <div className="phonetic-timeline-lane">
      <div className="phonetic-timeline-lane-label">{label}</div>
      <div className="phonetic-timeline-lane-track">
        {bars.map((bar, index) => {
          const left = duration > 0 ? (bar.startSec / duration) * 100 : 0;
          const width = duration > 0 ? ((bar.endSec - bar.startSec) / duration) * 100 : 0;
          return (
            <button
              type="button"
              key={`${label}:${index}:${bar.startSec}:${bar.endSec}`}
              className={`phonetic-timeline-bar tone-${tone}`}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.6)}%` }}
              title={bar.title}
              onClick={() => onPlayRange(bar.startSec, bar.endSec)}
            >
              <span>{bar.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PhoneticTimingTimeline({
  wordAlignments,
  phoneSpans,
  onPlayRange,
  zoom,
  onZoomChange,
}: {
  wordAlignments: PhoneticWordAlignment[];
  phoneSpans: PhoneticRescueTrace["utteranceZipaPhoneSpans"];
  onPlayRange: (startSec: number, endSec: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
}) {
  const qwenBars = wordAlignments.map<TimelineBar>((word) => ({
    label: word.wordText,
    startSec: word.startSec,
    endSec: word.endSec,
    title: `${word.wordText} · Qwen ${formatSeconds(word.startSec)}-${formatSeconds(word.endSec)}`,
  }));
  const zipaWordBars = wordAlignments
    .filter((word) => word.zipaStartSec != null && word.zipaEndSec != null)
    .map<TimelineBar>((word) => ({
      label: word.wordText,
      startSec: word.zipaStartSec!,
      endSec: word.zipaEndSec!,
      title: `${word.wordText} · ZIPA ${formatSeconds(word.zipaStartSec!)}-${formatSeconds(word.zipaEndSec!)} · raw phones ${word.zipaRawPhoneStart ?? "?"}..${word.zipaRawPhoneEnd ?? "?"}`,
    }));
  const phoneBars = phoneSpans.map<TimelineBar>((span) => ({
    label: span.token,
    startSec: span.startSec,
    endSec: span.endSec,
    title: `${span.token} · ${formatSeconds(span.startSec)}-${formatSeconds(span.endSec)} · frames ${span.startFrame}..${span.endFrame}`,
  }));
  const duration = Math.max(
    0,
    ...qwenBars.map((bar) => bar.endSec),
    ...zipaWordBars.map((bar) => bar.endSec),
    ...phoneBars.map((bar) => bar.endSec),
  );

  if (duration <= 0) {
    return null;
  }
  const widthPx = Math.max(920, Math.ceil(duration * 240 * zoom));
  const zoomOptions = [0.75, 1, 1.5, 2, 3, 4];

  return (
    <div className="phonetic-timeline-card">
      <div className="phonetic-timeline-header">
        <div className="phonetic-timeline-header-top">
          <strong>Timing</strong>
          <div className="phonetic-timeline-zoom-group">
            <span className="phonetic-timeline-zoom-label">zoom</span>
            {zoomOptions.map((option) => (
              <button
                key={`timeline-zoom-${option}`}
                type="button"
                className={`phonetic-timeline-zoom-button${Math.abs(option - zoom) < 0.01 ? " is-active" : ""}`}
                onClick={() => onZoomChange(option)}
              >
                {option}x
              </button>
            ))}
          </div>
        </div>
        <span>
          Qwen word timings, ZIPA projected word windows, and raw ZIPA phone spans on one axis.
        </span>
      </div>
      <div className="phonetic-timeline-scroll">
        <div className="phonetic-timeline-inner" style={{ width: `${widthPx}px` }}>
          <div className="phonetic-timeline-ruler">
            <div className="phonetic-timeline-lane-label">time</div>
            <div className="phonetic-timeline-ruler-track">
              {timelineTicks(duration).map((tick) => (
                <div
                  key={`tick:${tick.sec}`}
                  className="phonetic-timeline-tick"
                  style={{ left: `${tick.left}%` }}
                >
                  <span>{formatSeconds(tick.sec)}</span>
                </div>
              ))}
            </div>
          </div>
          <TimelineLane
            label="Qwen"
            bars={qwenBars}
            duration={duration}
            tone="qwen"
            onPlayRange={onPlayRange}
          />
          <TimelineLane
            label="ZIPA words"
            bars={zipaWordBars}
            duration={duration}
            tone="zipa"
            onPlayRange={onPlayRange}
          />
          <TimelineLane
            label="ZIPA phones"
            bars={phoneBars}
            duration={duration}
            tone="phone"
            onPlayRange={onPlayRange}
          />
        </div>
      </div>
    </div>
  );
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

function AsrAlternativesView({
  tokens,
}: {
  tokens: PhoneticRescueTrace["asrAlternatives"];
}) {
  if (tokens.length === 0) {
    return <div className="token-row muted">No token alternatives captured for this utterance.</div>;
  }

  const observedText = tokens.map((token) => token.chosenText).join("");

  const displayTokenText = (text: string) => {
    if (!text) return "∅";
    return text.replaceAll(" ", "␠");
  };

  return (
    <div className="asr-alt-section">
      <div className="asr-alt-observed-line">
        <span className="asr-alt-observed-label">Observed</span>
        <span className="asr-alt-observed-text">{observedText || "∅"}</span>
      </div>
      <div className="asr-alt-sequence-scroll">
        <div className="asr-alt-sequence">
          {tokens.map((token) => {
            const alternatives = token.alternatives.filter(
              (alternative, index, all) =>
                alternative.text !== token.chosenText &&
                all.findIndex((candidate) => candidate.text === alternative.text) === index,
            );

            return (
              <div
                key={`asr-alt:${token.tokenIndex}`}
                className="asr-alt-token"
                title={`token ${token.tokenIndex} · margin ${token.margin.toFixed(3)} · concentration ${token.concentration.toFixed(3)} · rev ${token.revision.toString()}`}
              >
                <div className="asr-alt-token-chosen">{displayTokenText(token.chosenText)}</div>
                <div className="asr-alt-token-alts">
                  {alternatives.slice(0, 3).map((alternative) => (
                    <span
                      key={`${token.tokenIndex}:${alternative.tokenId}`}
                      className="asr-alt-token-alt"
                    >
                      {displayTokenText(alternative.text)}
                    </span>
                  ))}
                </div>
                <div className="asr-alt-token-metrics muted">
                  m {token.margin.toFixed(2)} · c {token.concentration.toFixed(2)}
                </div>
              </div>
            );
          })}
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
  onSelectCut,
  cutSelected,
  controlState,
}: {
  word: PhoneticWordAlignment;
  label: string;
  onPlayOriginal: () => void;
  onPlayTranscript: () => void;
  onPlayZipa: () => void;
  onSelectCut?: (targetCommittedTokens: number) => void;
  cutSelected?: boolean;
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
          <div className="word-group-inline-box-row" style={{ gap: "0.4rem" }}>
            {onSelectCut ? (
              <button
                type="button"
                onClick={() => onSelectCut(word.tokenEnd)}
                className="word-group-inline-play-button"
                title={`Select cut after "${label}" at token ${word.tokenEnd}`}
                aria-label={`Select cut after ${label}`}
                style={{
                  minWidth: "2rem",
                  borderColor: cutSelected
                    ? "color-mix(in srgb, var(--lane-zipa) 75%, transparent)"
                    : undefined,
                  background: cutSelected
                    ? "color-mix(in srgb, var(--lane-zipa-bg) 55%, transparent)"
                    : undefined,
                }}
              >
                ✂
              </button>
            ) : null}
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
  onSimulateCut,
}: {
  trace: PhoneticRescueTrace;
  wsUrl?: string;
  onSimulateCut?: (targetCommittedTokens: number) => Promise<void>;
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
  const [selectedCutTokenEnd, setSelectedCutTokenEnd] = useState<number | null>(null);
  const [simulatingCut, setSimulatingCut] = useState(false);
  const [simulateCutError, setSimulateCutError] = useState<string | null>(null);
  const [timelineZoom, setTimelineZoom] = useState(2);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playingSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const sessionBufferRef = useRef<AudioBuffer | null>(null);
  const sessionBufferKeyRef = useRef<string | null>(null);
  const playbackRunIdRef = useRef(0);
  const synthBufferCacheRef = useRef<Map<string, Promise<AudioBuffer>>>(new Map());

  useEffect(() => {
    sessionBufferRef.current = null;
    sessionBufferKeyRef.current = null;
  }, [trace.sessionAudioF32, trace.sessionAudioSampleRateHz]);

  useEffect(() => {
    setSelectedCutTokenEnd(null);
    setSimulatingCut(false);
    setSimulateCutError(null);
  }, [trace.snapshotRevision]);

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

  const ensureSessionBuffer = async () => {
    const key = `${trace.snapshotRevision.toString()}:${trace.sessionAudioSampleRateHz}:${trace.sessionAudioF32.length}`;
    if (!trace.sessionAudioF32.length || !trace.sessionAudioSampleRateHz) {
      throw new Error("session audio is unavailable for this trace");
    }
    if (sessionBufferRef.current && sessionBufferKeyRef.current === key) {
      return sessionBufferRef.current;
    }
    const context = await ensureAudioContext();
    const channel = new Float32Array(trace.sessionAudioF32);
    const buffer = context.createBuffer(1, channel.length, trace.sessionAudioSampleRateHz);
    buffer.copyToChannel(channel, 0);
    sessionBufferRef.current = buffer;
    sessionBufferKeyRef.current = key;
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
    if (
      activeControl?.tokenStart === word.tokenStart &&
      activeControl.kind === kind
    ) {
      stopPlayback();
      return;
    }
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
    if (
      activeControl?.tokenStart === word.tokenStart &&
      activeControl.kind === "original"
    ) {
      stopPlayback();
      return;
    }
    try {
      setControlError(null);
      setActiveGlobalControl(null);
      setActiveControl({ tokenStart: word.tokenStart, kind: "original", phase: "loading" });
      const buffer = await ensureSessionBuffer();
      await playBufferRange(buffer, word.startSec, word.endSec, word.tokenStart, "original");
    } catch (error) {
      setControlError(error instanceof Error ? error.message : String(error));
      setActiveControl(null);
    }
  };

  const playOriginalRange = async (startSec: number, endSec: number) => {
    try {
      setControlError(null);
      setActiveGlobalControl(null);
      setActiveControl(null);
      const buffer = await ensureSessionBuffer();
      await playBufferRange(buffer, startSec, endSec, Number.MAX_SAFE_INTEGER, "original");
    } catch (error) {
      setControlError(error instanceof Error ? error.message : String(error));
      setActiveControl(null);
    }
  };

  const playLaneSequence = async (kind: "original" | "transcript" | "zipa") => {
    if (activeGlobalControl?.kind === kind) {
      stopPlayback();
      return;
    }
    try {
      setControlError(null);
      stopPlayback();
      const runId = playbackRunIdRef.current;
      setActiveGlobalControl({ kind, phase: "loading" });
      let originalBuffer: AudioBuffer | null = null;
      if (kind === "original") {
        originalBuffer = await ensureSessionBuffer();
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
            kind === "transcript" ? word.transcriptRaw : word.zipaRaw,
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

  const handleSimulateCut = async () => {
    if (!onSimulateCut || selectedCutTokenEnd == null) {
      return;
    }
    try {
      setSimulateCutError(null);
      setSimulatingCut(true);
      stopPlayback();
      await onSimulateCut(selectedCutTokenEnd);
    } catch (error) {
      setSimulateCutError(error instanceof Error ? error.message : String(error));
    } finally {
      setSimulatingCut(false);
    }
  };

  useEffect(() => {
    if (!wsUrl || wordAlignments.length === 0) return;
    void Promise.allSettled([
      ...wordAlignments.map((word) => getSynthBufferCached("transcript", word.transcriptRaw)),
      ...wordAlignments.map((word) => getSynthBufferCached("zipa", word.zipaRaw)),
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
              <div className="prototype-stack" style={{ gap: "0.45rem", marginTop: "0.55rem" }}>
                <div className="token-row muted" style={{ userSelect: "text" }}>
                  ASR token alternatives
                </div>
                <AsrAlternativesView tokens={trace.asrAlternatives} />
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
            {onSimulateCut ? (
              <button
                type="button"
                className="word-group-inline-play-button global-lane-play-button"
                onClick={() => void handleSimulateCut()}
                disabled={selectedCutTokenEnd == null || simulatingCut}
                title={
                  selectedCutTokenEnd == null
                    ? "Select a word cut (✂) first"
                    : `Run simulated cut at committed tokens=${selectedCutTokenEnd}`
                }
              >
                <span>{simulatingCut ? "…" : "✂"}</span>
                <span>
                  {selectedCutTokenEnd == null
                    ? "Simulate Cut"
                    : `Sim Cut @ ${selectedCutTokenEnd}`}
                </span>
              </button>
            ) : null}
          </div>

          {onSimulateCut ? (
            <div className="token-row muted" style={{ marginBottom: "0.45rem" }}>
              selected cut:{" "}
              {selectedCutTokenEnd == null
                ? "none"
                : `after token ${selectedCutTokenEnd}`}
            </div>
          ) : null}

          <PhoneticTimingTimeline
            wordAlignments={wordAlignments}
            phoneSpans={trace.utteranceZipaPhoneSpans}
            onPlayRange={(startSec, endSec) => void playOriginalRange(startSec, endSec)}
            zoom={timelineZoom}
            onZoomChange={setTimelineZoom}
          />

          {controlError && (
            <div className="notice-row">
              <span className="error-pill">{controlError}</span>
            </div>
          )}
          {simulateCutError && (
            <div className="notice-row">
              <span className="error-pill">{simulateCutError}</span>
            </div>
          )}
          <div className="word-group-strip">
            {wordGroups.map(({ word, label }) => (
              <WordGroup
                key={`${word.tokenStart}:${word.tokenEnd}:${word.wordText}`}
                word={word}
                label={label}
                onPlayOriginal={() => void playOriginalWord(word)}
                onPlayTranscript={() => void synthWord(word, "transcript", word.transcriptRaw)}
                onPlayZipa={() => void synthWord(word, "zipa", word.zipaRaw)}
                onSelectCut={onSimulateCut ? setSelectedCutTokenEnd : undefined}
                cutSelected={selectedCutTokenEnd === word.tokenEnd}
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
