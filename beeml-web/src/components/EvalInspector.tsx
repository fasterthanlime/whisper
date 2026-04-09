import { useRef, useState, useCallback, useEffect } from "react";
import type { EvalInspectorData } from "../types";
import { EvalPlaybackBar } from "./EvalPlaybackBar";
import { EvalTimeline } from "./EvalTimeline";
import { PhoneticRescuePanel } from "./PhoneticRescuePanel";
import { TranscriptComparison } from "./TranscriptComparison";

export function EvalInspector({
  data,
  audioUrl,
  wsUrl,
}: {
  data: EvalInspectorData;
  audioUrl?: string;
  wsUrl?: string;
}) {
  const ctxRef = useRef<AudioContext | null>(null);
  const bufferRef = useRef<AudioBuffer | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [zoom, setZoom] = useState(3);
  const rafRef = useRef<number>(0);
  const playStartRef = useRef<{ ctxTime: number; offset: number } | null>(null);
  const rangeEndRef = useRef<number | null>(null);
  const sessionAudio = data.phoneticTrace;

  useEffect(() => {
    const traceAudio = sessionAudio?.sessionAudioF32;
    const traceRate = sessionAudio?.sessionAudioSampleRateHz;
    if ((!audioUrl && !traceAudio?.length) || !traceRate && !audioUrl) return;
    const ctx = new AudioContext();
    ctxRef.current = ctx;
    if (traceAudio?.length && traceRate) {
      const channel = new Float32Array(traceAudio);
      const buf = ctx.createBuffer(1, channel.length, traceRate);
      buf.copyToChannel(channel, 0);
      bufferRef.current = buf;
      setDuration(buf.duration);
    } else if (audioUrl) {
      fetch(audioUrl)
        .then((r) => r.arrayBuffer())
        .then((ab) => ctx.decodeAudioData(ab))
        .then((buf) => {
          bufferRef.current = buf;
          setDuration(buf.duration);
        });
    }

    return () => {
      sourceRef.current?.stop();
      sourceRef.current = null;
      ctx.close();
      ctxRef.current = null;
      bufferRef.current = null;
    };
  }, [audioUrl]);

  const stopSource = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.onended = null;
      sourceRef.current.stop();
      sourceRef.current = null;
    }
    playStartRef.current = null;
    rangeEndRef.current = null;
  }, []);

  const playFrom = useCallback(
    (offset: number, end?: number) => {
      const ctx = ctxRef.current;
      const buf = bufferRef.current;
      if (!ctx || !buf) return;

      stopSource();

      const source = ctx.createBufferSource();
      source.buffer = buf;
      source.connect(ctx.destination);

      const dur = end != null ? end - offset : undefined;
      source.start(0, offset, dur);
      sourceRef.current = source;
      playStartRef.current = { ctxTime: ctx.currentTime, offset };
      rangeEndRef.current = end ?? null;

      source.onended = () => {
        sourceRef.current = null;
        playStartRef.current = null;
        rangeEndRef.current = null;
        setPlaying(false);
      };

      setPlaying(true);
      setCurrentTime(offset);
    },
    [stopSource],
  );

  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      const ctx = ctxRef.current;
      const ps = playStartRef.current;
      if (ctx && ps) {
        const t = ps.offset + (ctx.currentTime - ps.ctxTime);
        setCurrentTime(t);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing]);

  const handlePlayPause = useCallback(() => {
    if (playing) {
      stopSource();
      setPlaying(false);
    } else {
      const startFrom = currentTime >= duration - 0.05 ? 0 : currentTime;
      playFrom(startFrom);
    }
  }, [playing, currentTime, duration, playFrom, stopSource]);

  const handleSeek = useCallback(
    (time: number) => {
      setCurrentTime(time);
      if (playing) playFrom(time);
    },
    [playing, playFrom],
  );

  const handlePlayRange = useCallback(
    (start: number, end: number) => playFrom(start, end),
    [playFrom],
  );

  return (
    <div className="eval-inspector">
      {(audioUrl || sessionAudio?.sessionAudioF32.length) && (
        <EvalPlaybackBar
          playing={playing}
          currentTime={currentTime}
          duration={duration}
          zoom={zoom}
          tokens={data.qwenAlignment}
          onPlayPause={handlePlayPause}
          onSeek={handleSeek}
          onZoomChange={setZoom}
        />
      )}

      <EvalTimeline
        alignments={data.alignments}
        qwenAlignment={data.qwenAlignment}
        sentenceCandidates={data.prototype.sentenceCandidates}
        reranker={data.prototype.reranker}
        currentTime={currentTime}
        duration={duration}
        onSeek={handleSeek}
        onPlayRange={handlePlayRange}
        zoom={zoom}
      />

      <div className="eval-detail">
        {data.elapsedMs != null && (
          <div className="eval-elapsed">
            {(data.elapsedMs / 1000).toFixed(2)}s total
          </div>
        )}

        <TranscriptComparison
          transcriptLabel={data.transcriptLabel}
          transcript={data.transcript}
          expected={data.expected}
          corrected={data.prototype.corrected}
          accepted={data.prototype.accepted}
        />

        {data.phoneticTrace ? (
          <PhoneticRescuePanel
            trace={data.phoneticTrace}
            wsUrl={wsUrl}
          />
      ) : null}
      </div>
    </div>
  );
}
