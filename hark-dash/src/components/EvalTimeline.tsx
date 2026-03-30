import { useRef, useState, useEffect, useCallback } from "react";
import type { PrototypeAlignments, TimedToken, SentenceCandidate, Reranker } from "../types";

type LaneToken = TimedToken & { dim?: boolean };

type Lane = {
  label: string;
  tokens: LaneToken[];
  color: string;
  bg: string;
};

/**
 * Build a full corrected lane: all transcript words with edits applied.
 * Unchanged words are marked dim; edited spans show the replacement.
 */
function buildCorrectedLane(
  transcriptTokens: TimedToken[],
  sentenceCandidates: SentenceCandidate[],
  reranker: Reranker | null | undefined,
): LaneToken[] {
  if (!transcriptTokens?.length) return [];

  // Find the chosen candidate's edits
  let edits: SentenceCandidate["edits"] = [];
  if (reranker?.chosenIndex != null && sentenceCandidates[reranker.chosenIndex]?.edits?.length) {
    edits = sentenceCandidates[reranker.chosenIndex].edits;
  }

  // Build a map: token index → edit (for the start of each edit span)
  const editByStart = new Map<number, (typeof edits)[0]>();
  const editCovered = new Set<number>();
  for (const edit of edits) {
    editByStart.set(edit.tokenStart, edit);
    for (let i = edit.tokenStart; i < edit.tokenEnd; i++) {
      editCovered.add(i);
    }
  }

  const tokens: LaneToken[] = [];
  let i = 0;
  while (i < transcriptTokens.length) {
    const edit = editByStart.get(i);
    if (edit) {
      // Edited span: show replacement
      const startToken = transcriptTokens[edit.tokenStart];
      const endIdx = Math.min(edit.tokenEnd - 1, transcriptTokens.length - 1);
      const endToken = transcriptTokens[endIdx];
      if (startToken && endToken) {
        tokens.push({
          w: edit.to,
          s: startToken.s,
          e: endToken.e,
          dim: false,
        });
      }
      i = edit.tokenEnd;
    } else {
      // Unchanged word
      tokens.push({
        ...transcriptTokens[i],
        dim: true,
      });
      i++;
    }
  }
  return tokens;
}

function buildLanes(
  alignments: PrototypeAlignments,
  parakeetAlignment: TimedToken[],
  sentenceCandidates?: SentenceCandidate[],
  reranker?: Reranker | null,
): Lane[] {
  const lanes: Lane[] = [];

  if (parakeetAlignment.length > 0) {
    lanes.push({
      label: "Parakeet",
      tokens: parakeetAlignment,
      color: "var(--lane-parakeet)",
      bg: "var(--lane-parakeet-bg)",
    });
  }

  if (alignments.espeak && alignments.espeak.length > 0) {
    lanes.push({
      label: "eSpeak",
      tokens: alignments.espeak,
      color: "var(--lane-espeak)",
      bg: "var(--lane-espeak-bg)",
    });
  }

  if (alignments.zipaEspeak && alignments.zipaEspeak.length > 0) {
    lanes.push({
      label: "ZIPA@eSpeak",
      tokens: alignments.zipaEspeak,
      color: "var(--lane-zipa-espeak)",
      bg: "var(--lane-zipa-espeak-bg)",
    });
  }

  if (alignments.zipa && alignments.zipa.length > 0) {
    lanes.push({
      label: "ZIPA",
      tokens: alignments.zipa,
      color: "var(--lane-zipa)",
      bg: "var(--lane-zipa-bg)",
    });
  }

  // Corrected lane: all transcript words with reranker edits applied
  const transcriptBase = alignments.transcript ?? alignments.espeak ?? [];
  const correctedTokens = buildCorrectedLane(
    transcriptBase,
    sentenceCandidates ?? [],
    reranker,
  );
  if (correctedTokens.length > 0) {
    lanes.push({
      label: "Corrected",
      tokens: correctedTokens,
      color: "var(--lane-reranker)",
      bg: "var(--lane-reranker-bg)",
    });
  }

  return lanes;
}

const LABEL_WIDTH = 90;
const LANE_HEIGHT = 36;
const RULER_HEIGHT = 24;

type Selection = { laneIdx: number; tokenIdx: number } | null;

export function EvalTimeline({
  alignments,
  parakeetAlignment,
  sentenceCandidates,
  reranker,
  currentTime,
  duration,
  onSeek,
  onPlayRange,
  zoom = 1,
}: {
  alignments: PrototypeAlignments;
  parakeetAlignment: TimedToken[];
  sentenceCandidates?: SentenceCandidate[];
  reranker?: Reranker | null;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  onPlayRange?: (start: number, end: number) => void;
  zoom?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const lanes = buildLanes(alignments, parakeetAlignment, sentenceCandidates, reranker);
  const lanesRef = useRef(lanes);
  lanesRef.current = lanes;
  const pxPerSec = 120 * zoom;

  const [selection, setSelection] = useState<Selection>(null);

  // Total width covers at least the audio duration and all alignment data
  let maxEnd = duration;
  for (const lane of lanes) {
    for (const t of lane.tokens) {
      if (t.e > maxEnd) maxEnd = t.e;
    }
  }
  const totalWidth = Math.max(maxEnd, 0.01) * pxPerSec;

  // Ruler: click + drag to scrub
  const rulerRef = useRef<HTMLDivElement>(null);
  const seekFromX = useCallback(
    (clientX: number) => {
      const el = rulerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = clientX - rect.left;
      const time = x / pxPerSec;
      onSeek(Math.max(0, Math.min(time, duration)));
    },
    [pxPerSec, duration, onSeek],
  );

  const handleRulerPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      setSelection(null);
      seekFromX(e.clientX);
    },
    [seekFromX],
  );

  const handleRulerPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (e.buttons === 0) return;
      seekFromX(e.clientX);
    },
    [seekFromX],
  );

  // Select a word and play it
  const selectAndPlay = useCallback(
    (laneIdx: number, tokenIdx: number) => {
      const lanes = lanesRef.current;
      if (laneIdx < 0 || laneIdx >= lanes.length) return;
      const tokens = lanes[laneIdx].tokens;
      if (tokenIdx < 0 || tokenIdx >= tokens.length) return;
      const token = tokens[tokenIdx];
      setSelection({ laneIdx, tokenIdx });
      if (onPlayRange) {
        onPlayRange(token.s, token.e);
      } else {
        onSeek(token.s);
      }
    },
    [onPlayRange, onSeek],
  );

  // Keyboard: left/right to navigate words on selected lane, up/down to switch lanes
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (!selection) return;

      const lanes = lanesRef.current;
      const { laneIdx, tokenIdx } = selection;

      if (e.code === "ArrowLeft") {
        e.preventDefault();
        e.stopPropagation();
        selectAndPlay(laneIdx, tokenIdx - 1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        e.stopPropagation();
        selectAndPlay(laneIdx, tokenIdx + 1);
      } else if (e.code === "ArrowUp") {
        e.preventDefault();
        if (laneIdx > 0) {
          // Find closest token on the lane above by time
          const curToken = lanes[laneIdx].tokens[tokenIdx];
          const aboveTokens = lanes[laneIdx - 1].tokens;
          const closest = findClosestToken(aboveTokens, curToken.s);
          selectAndPlay(laneIdx - 1, closest);
        }
      } else if (e.code === "ArrowDown") {
        e.preventDefault();
        if (laneIdx < lanes.length - 1) {
          const curToken = lanes[laneIdx].tokens[tokenIdx];
          const belowTokens = lanes[laneIdx + 1].tokens;
          const closest = findClosestToken(belowTokens, curToken.s);
          selectAndPlay(laneIdx + 1, closest);
        }
      } else if (e.code === "Escape") {
        setSelection(null);
      }
    };
    // Use capture so we get it before the playback bar's handler
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [selection, selectAndPlay]);

  // Auto-scroll: when playhead passes 3/4, jump so it's at 1/4
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const playheadX = LABEL_WIDTH + currentTime * pxPerSec;
    const viewWidth = el.clientWidth;
    const relX = playheadX - el.scrollLeft;
    if (relX > viewWidth * 0.75 || relX < LABEL_WIDTH) {
      el.scrollLeft = playheadX - viewWidth * 0.25;
    }
  }, [currentTime, pxPerSec]);

  if (lanes.length === 0) {
    return (
      <div
        style={{
          padding: "2rem",
          textAlign: "center",
          color: "var(--text-muted)",
          background: "var(--bg-surface)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        No alignment data available
      </div>
    );
  }

  const playheadX = LABEL_WIDTH + currentTime * pxPerSec;
  const contentHeight = RULER_HEIGHT + lanes.length * LANE_HEIGHT;

  // Generate ruler tick marks
  const rulerTicks: number[] = [];
  let tickInterval = 1;
  if (pxPerSec < 60) tickInterval = 2;
  if (pxPerSec < 30) tickInterval = 5;
  if (pxPerSec > 200) tickInterval = 0.5;
  if (pxPerSec > 500) tickInterval = 0.25;
  for (let t = 0; t <= maxEnd + tickInterval; t += tickInterval) {
    rulerTicks.push(t);
  }

  return (
    <div
      ref={containerRef}
      className="hide-scrollbar"
      style={{
        width: "100%",
        overflowX: "auto",
        overflowY: "hidden",
        background: "var(--bg-surface)",
        borderBottom: "1px solid var(--border)",
        position: "relative",
      }}
    >
      <div style={{ width: LABEL_WIDTH + totalWidth, minHeight: contentHeight, position: "relative" }}>
        {/* Ruler lane — click/drag to scrub */}
        <div style={{ display: "flex", height: RULER_HEIGHT }}>
          <div
            style={{
              width: LABEL_WIDTH,
              flexShrink: 0,
              position: "sticky",
              left: 0,
              zIndex: 2,
              background: "var(--bg-surface)",
            }}
          />
          <div
            ref={rulerRef}
            onPointerDown={handleRulerPointerDown}
            onPointerMove={handleRulerPointerMove}
            style={{
              position: "relative",
              width: totalWidth,
              height: RULER_HEIGHT,
              cursor: "crosshair",
              borderBottom: "1px solid var(--border)",
              touchAction: "none",
            }}
          >
            {rulerTicks.map((t) => (
              <div
                key={t}
                style={{
                  position: "absolute",
                  left: t * pxPerSec,
                  top: 0,
                  height: "100%",
                  borderLeft: "1px solid var(--border)",
                  opacity: 0.4,
                }}
              >
                <span
                  style={{
                    fontSize: "0.6rem",
                    color: "var(--text-dim)",
                    position: "absolute",
                    bottom: 2,
                    left: 3,
                    whiteSpace: "nowrap",
                  }}
                >
                  {t % 1 === 0 ? `${t}s` : `${t.toFixed(2)}s`}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Token lanes */}
        {lanes.map((lane, laneIdx) => (
          <div key={lane.label} style={{ display: "flex", alignItems: "center", height: LANE_HEIGHT, position: "relative" }}>
            <div
              style={{
                width: LABEL_WIDTH,
                paddingLeft: 12,
                fontSize: "0.75rem",
                fontWeight: 600,
                color: lane.color,
                flexShrink: 0,
                position: "sticky",
                left: 0,
                zIndex: 2,
                background: "var(--bg-surface)",
              }}
            >
              {lane.label}
            </div>
            <div style={{ position: "relative", width: totalWidth, height: 28 }}>
              {lane.tokens.map((token, ti) => {
                const left = token.s * pxPerSec;
                const width = Math.max((token.e - token.s) * pxPerSec, 2);
                const isPlaying = currentTime >= token.s && currentTime < token.e;
                const isSelected = selection?.laneIdx === laneIdx && selection?.tokenIdx === ti;
                return (
                  <div
                    key={ti}
                    title={`${token.w} (${token.s.toFixed(2)}s–${token.e.toFixed(2)}s${token.c != null ? `, conf ${token.c.toFixed(2)}` : ""})`}
                    onClick={(e) => {
                      e.stopPropagation();
                      selectAndPlay(laneIdx, ti);
                    }}
                    style={{
                      position: "absolute",
                      left,
                      width,
                      top: 2,
                      height: 24,
                      background: isSelected
                        ? lane.color + "60"
                        : isPlaying
                          ? lane.color + "40"
                          : lane.bg,
                      border: `2px solid ${isSelected ? lane.color : isPlaying ? lane.color + "80" : lane.color + "40"}`,
                      borderRadius: 3,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "0.7rem",
                      color: lane.color,
                      overflow: "hidden",
                      whiteSpace: "nowrap",
                      textOverflow: "ellipsis",
                      padding: "0 2px",
                      cursor: "pointer",
                      outline: isSelected ? `1px solid ${lane.color}` : "none",
                      outlineOffset: 1,
                    }}
                  >
                    {token.w}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Playhead */}
        <div
          style={{
            position: "absolute",
            left: playheadX,
            top: 0,
            bottom: 0,
            width: 2,
            background: "var(--accent)",
            zIndex: 10,
            pointerEvents: "none",
          }}
        />
      </div>
    </div>
  );
}

function findClosestToken(tokens: TimedToken[], time: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < tokens.length; i++) {
    const dist = Math.abs(tokens[i].s - time);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}
