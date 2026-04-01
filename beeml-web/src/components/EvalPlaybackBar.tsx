import { useEffect, useCallback } from "react";
import type { TimedToken } from "../types";

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

function currentWord(tokens: TimedToken[], time: number): string {
  for (const t of tokens) {
    if (time >= t.s && time < t.e) return t.w;
  }
  return "";
}

const ZOOM_LEVELS = [0.25, 0.5, 1, 1.6, 2.6, 3.8, 5.4] as const;

export function EvalPlaybackBar({
  playing,
  currentTime,
  duration,
  zoom,
  tokens,
  onPlayPause,
  onSeek,
  onZoomChange,
}: {
  playing: boolean;
  currentTime: number;
  duration: number;
  zoom: number;
  tokens: TimedToken[];
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onZoomChange: (z: number) => void;
}) {
  const step = useCallback(
    (delta: number) => {
      onSeek(Math.max(0, Math.min(currentTime + delta, duration)));
    },
    [currentTime, duration, onSeek],
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          onPlayPause();
          break;
        case "ArrowLeft":
          e.preventDefault();
          step(e.shiftKey ? -1 : -0.1);
          break;
        case "ArrowRight":
          e.preventDefault();
          step(e.shiftKey ? 1 : 0.1);
          break;
        case "Home":
          e.preventDefault();
          onSeek(0);
          break;
        case "End":
          e.preventDefault();
          onSeek(duration);
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onPlayPause, step, onSeek, duration]);

  const word = currentWord(tokens, currentTime);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        padding: "0.4rem 1rem",
        background: "var(--bg-surface)",
        borderBottom: "1px solid var(--border)",
        fontSize: "0.85rem",
      }}
    >
      <button onClick={onPlayPause} style={{ minWidth: 36 }}>
        {playing ? "⏸" : "▶"}
      </button>
      <span style={{ fontVariantNumeric: "tabular-nums", color: "var(--text-muted)" }}>
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
      <span
        style={{
          flex: 1,
          textAlign: "center",
          fontSize: "1.4rem",
          fontWeight: 700,
          color: word ? "var(--text)" : "transparent",
          letterSpacing: "0.02em",
        }}
      >
        {word || "\u00A0"}
      </span>
      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>zoom</span>
      {ZOOM_LEVELS.map((z) => (
        <button
          key={z}
          className={zoom === z ? "primary" : ""}
          onClick={() => onZoomChange(z)}
          style={{ padding: "0.2em 0.5em", fontSize: "0.75rem" }}
        >
          {z}x
        </button>
      ))}
    </div>
  );
}
