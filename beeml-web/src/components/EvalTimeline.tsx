import { useRef, useState, useEffect, useCallback } from "react";
import type { PrototypeAlignments, TimedToken, SentenceCandidate, Reranker } from "../types";

type LaneToken = TimedToken & {
  dim?: boolean;
  editFrom?: string;
  editFromPhonemes?: string | null;
  editToPhonemes?: string | null;
  editVia?: string;
  editRank?: number;
  editSimilarity?: number | null;
};

type Lane = {
  label: string;
  tokens: LaneToken[];
  color: string;
  bg: string;
};

/**
 * Build a corrected token lane from transcript tokens + a set of edits.
 * Unchanged words are dim; edited spans show the replacement.
 */
function applyCandidateEdits(
  transcriptTokens: TimedToken[],
  edits: SentenceCandidate["edits"],
): LaneToken[] {
  if (!transcriptTokens?.length) return [];

  const editByStart = new Map<number, (typeof edits)[0]>();
  for (const edit of edits) {
    editByStart.set(edit.tokenStart, edit);
  }

  const tokens: LaneToken[] = [];
  let i = 0;
  while (i < transcriptTokens.length) {
    const edit = editByStart.get(i);
    if (edit) {
      const startToken = transcriptTokens[edit.tokenStart];
      const endIdx = Math.min(edit.tokenEnd - 1, transcriptTokens.length - 1);
      const endToken = transcriptTokens[endIdx];
      if (startToken && endToken) {
        tokens.push({
          w: edit.to,
          s: startToken.s,
          e: endToken.e,
          dim: false,
          editFrom: edit.from,
          editFromPhonemes: edit.fromPhonemes,
          editToPhonemes: edit.toPhonemes,
          editVia: edit.via,
          editRank: edit.score,
          editSimilarity: edit.phoneticScore,
        });
      }
      i = edit.tokenEnd;
    } else {
      tokens.push({ ...transcriptTokens[i], dim: true });
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
  const transcriptBase = alignments.transcript ?? alignments.espeak ?? [];
  const candidates = sentenceCandidates ?? [];
  const chosenIdx = reranker?.chosenIndex;

  // Candidate lanes: one per sentence candidate, chosen first
  if (candidates.length > 0 && transcriptBase.length > 0) {
    // Find reranker scores for each candidate
    const rerankerCandidates = reranker?.candidates;

    // Sort: chosen first, then by yesProb descending
    const indices = candidates.map((_, i) => i);
    indices.sort((a, b) => {
      if (a === chosenIdx) return -1;
      if (b === chosenIdx) return 1;
      const aProb = rerankerCandidates?.[a]?.yesProb ?? 0;
      const bProb = rerankerCandidates?.[b]?.yesProb ?? 0;
      return bProb - aProb;
    });

    for (const idx of indices) {
      const candidate = candidates[idx];
      const rc = rerankerCandidates?.[idx];
      const isChosen = idx === chosenIdx;
      const tokens = applyCandidateEdits(transcriptBase, candidate.edits);
      if (tokens.length === 0) continue;

      const pct = rc ? `${(rc.yesProb * 100).toFixed(0)}%` : "";
      const label = isChosen ? `✓ ${pct}` : `#${idx} ${pct}`;

      lanes.push({
        label,
        tokens,
        color: isChosen ? "var(--lane-reranker)" : "var(--text-dim)",
        bg: isChosen ? "var(--lane-reranker-bg)" : "transparent",
      });
    }
  }

  if (parakeetAlignment.length > 0) {
    lanes.push({
      label: "QWEN",
      tokens: parakeetAlignment,
      color: "var(--lane-qwen)",
      bg: "var(--lane-qwen-bg)",
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
  const [hover, setHover] = useState<{ laneIdx: number; tokenIdx: number; x: number; y: number } | null>(null);

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
        overflowY: "auto",
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
                const isDim = !!(token as LaneToken).dim;
                return (
                  <div
                    key={ti}
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setHover({ laneIdx, tokenIdx: ti, x: rect.left + rect.width / 2, y: rect.bottom + 4 });
                    }}
                    onMouseLeave={() => setHover(null)}
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
                          : isDim
                            ? "transparent"
                            : lane.bg,
                      border: isDim && !isSelected && !isPlaying
                        ? "1px dashed var(--border)"
                        : `2px solid ${isSelected ? lane.color : isPlaying ? lane.color + "80" : lane.color + "60"}`,
                      borderRadius: 3,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: isDim ? "0.65rem" : "0.7rem",
                      fontWeight: isDim ? 400 : 600,
                      color: isDim ? "var(--text-muted)" : lane.color,
                      overflow: "hidden",
                      whiteSpace: "nowrap",
                      textOverflow: "ellipsis",
                      padding: "0 2px",
                      cursor: "pointer",
                      opacity: isDim ? 0.7 : 1,
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

      {/* Hover popover */}
      {hover && (() => {
        const token = lanes[hover.laneIdx]?.tokens[hover.tokenIdx] as LaneToken | undefined;
        if (!token) return null;
        return (
          <div
            style={{
              position: "fixed",
              left: hover.x,
              top: hover.y,
              transform: "translateX(-50%)",
              zIndex: 100,
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "0.5rem 0.75rem",
              fontSize: "0.8rem",
              lineHeight: 1.6,
              boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
              pointerEvents: "none",
              maxWidth: 360,
              whiteSpace: "nowrap",
            }}
          >
            <div style={{ fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>
              {token.w} <span style={{ fontWeight: 400, color: "var(--text-muted)" }}>{token.s.toFixed(2)}s – {token.e.toFixed(2)}s</span>
            </div>
            {token.c != null && (
              <div style={{ color: "var(--text-muted)" }}>conf {token.c.toFixed(3)}</div>
            )}
            {token.editFrom && (
              <>
                <div style={{ marginTop: 4, borderTop: "1px solid var(--border)", paddingTop: 4 }}>
                  <table style={{ borderCollapse: "collapse", fontSize: "0.8rem", width: "100%" }}>
                    <tbody>
                      <tr>
                        <td style={{ color: "var(--text-muted)", paddingRight: 8, verticalAlign: "top" }}>from</td>
                        <td>
                          <span style={{ color: "var(--danger)", fontWeight: 600 }}>{token.editFrom}</span>
                          {token.editFromPhonemes && <span style={{ color: "var(--text-dim)", marginLeft: 6, fontStyle: "italic" }}>/{token.editFromPhonemes}/</span>}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ color: "var(--text-muted)", paddingRight: 8, verticalAlign: "top" }}>to</td>
                        <td>
                          <span style={{ color: "var(--success)", fontWeight: 600 }}>{token.w}</span>
                          {token.editToPhonemes && <span style={{ color: "var(--text-dim)", marginLeft: 6, fontStyle: "italic" }}>/{token.editToPhonemes}/</span>}
                        </td>
                      </tr>
                      {token.editVia && (
                        <tr>
                          <td style={{ color: "var(--text-muted)", paddingRight: 8 }}>via</td>
                          <td>{token.editVia}</td>
                        </tr>
                      )}
                      {token.editSimilarity != null && (
                        <tr>
                          <td style={{ color: "var(--text-muted)", paddingRight: 8 }}>similarity</td>
                          <td style={{ fontVariantNumeric: "tabular-nums" }}>{token.editSimilarity.toFixed(3)}</td>
                        </tr>
                      )}
                      {token.editRank != null && (
                        <tr>
                          <td style={{ color: "var(--text-muted)", paddingRight: 8 }}>rank</td>
                          <td style={{ fontVariantNumeric: "tabular-nums" }}>{token.editRank.toFixed(3)}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        );
      })()}
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
