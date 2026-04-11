import { type UIEvent, useCallback, useEffect, useRef } from "react";
import type { WordSpan } from "../cut-trace-types";

const LABEL_WIDTH = 72;
const LANE_HEIGHT = 36;
const RULER_HEIGHT = 24;
const BASE_PX_PER_SEC = 120;

const REGION_STYLE = {
  stable: {
    label: "stable",
    color: "var(--lane-qwen)",
    bg: "var(--lane-qwen-bg)",
  },
  carry: {
    label: "carry",
    color: "var(--lane-reranker)",
    bg: "var(--lane-reranker-bg)",
  },
  preview: {
    label: "preview",
    color: "var(--danger)",
    bg: "rgba(207,34,46,0.12)",
  },
} as const;

const REGION_ORDER = ["stable", "carry", "preview"] as const;

interface Lane {
  region: (typeof REGION_ORDER)[number];
  words: WordSpan[];
}

export function CutTimeline({
  wordSpans,
  cutSampleSecs,
  zoom,
  viewStartSec,
  onViewStartSecChange,
}: {
  wordSpans: WordSpan[];
  cutSampleSecs?: number | null;
  zoom: number;
  viewStartSec: number;
  onViewStartSecChange?: (startSec: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pxPerSec = BASE_PX_PER_SEC * zoom;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const target = Math.max(0, viewStartSec * pxPerSec);
    const maxLeft = Math.max(0, el.scrollWidth - el.clientWidth);
    const clamped = Math.min(maxLeft, target);
    if (Math.abs(el.scrollLeft - clamped) > 1) {
      el.scrollLeft = clamped;
    }
  }, [pxPerSec, viewStartSec]);

  const handleScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      if (!onViewStartSecChange) return;
      const el = event.currentTarget;
      onViewStartSecChange(el.scrollLeft / pxPerSec);
    },
    [onViewStartSecChange, pxPerSec],
  );

  // Separate into lanes, keeping only words that have timing
  const lanesMap = new Map<string, WordSpan[]>();
  for (const region of REGION_ORDER) lanesMap.set(region, []);
  for (const w of wordSpans) {
    if (w.start_secs != null && w.end_secs != null) {
      lanesMap.get(w.region)?.push(w);
    }
  }

  const lanes: Lane[] = REGION_ORDER.filter(
    (r) => (lanesMap.get(r)?.length ?? 0) > 0
  ).map((region) => ({ region, words: lanesMap.get(region)! }));

  // Compute time range across all timed words
  let maxSecs = 0;
  for (const w of wordSpans) {
    if (w.end_secs != null && w.end_secs > maxSecs) maxSecs = w.end_secs;
  }
  if (cutSampleSecs != null && cutSampleSecs > maxSecs) maxSecs = cutSampleSecs;
  maxSecs = Math.max(maxSecs, 0.5);

  const totalWidth = maxSecs * pxPerSec;
  const contentHeight = RULER_HEIGHT + lanes.length * LANE_HEIGHT;

  // Ruler ticks at 0.5s intervals, labels at 1s
  const ticks: number[] = [];
  for (let t = 0; t <= maxSecs + 0.5; t += 0.5) {
    ticks.push(parseFloat(t.toFixed(2)));
  }

  return (
    <div
      ref={containerRef}
      className="timeline hide-scrollbar"
      onScroll={handleScroll}
    >
      <div
        className="timeline-inner"
        style={{ width: LABEL_WIDTH + totalWidth, minHeight: contentHeight, position: "relative" }}
      >
        {/* Ruler */}
        <div className="timeline-ruler-row" style={{ height: RULER_HEIGHT }}>
          <div className="timeline-ruler-label" style={{ width: LABEL_WIDTH }} />
          <div className="timeline-ruler" style={{ width: totalWidth, height: RULER_HEIGHT }}>
            {ticks.map((t) => (
              <div key={t} className="timeline-tick" style={{ left: t * pxPerSec }}>
                {t % 1 === 0 && <span>{t}s</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Lanes */}
        {lanes.map(({ region, words }) => {
          const style = REGION_STYLE[region];
          return (
            <div key={region} className="timeline-lane" style={{ height: LANE_HEIGHT }}>
              <div
                className="timeline-lane-label"
                style={{ width: LABEL_WIDTH, color: style.color }}
              >
                {style.label}
              </div>
              <div
                className="timeline-lane-tokens"
                style={{ width: totalWidth, height: 28 }}
              >
                {words.map((w, i) => {
                  const left = w.start_secs! * pxPerSec;
                  const width = Math.max(
                    (w.end_secs! - w.start_secs!) * pxPerSec,
                    2
                  );
                  return (
                    <div
                      key={i}
                      className="timeline-token"
                      title={`tok ${w.start}–${w.end}  ${w.start_secs!.toFixed(3)}s–${w.end_secs!.toFixed(3)}s`}
                      style={{
                        left,
                        width,
                        background: style.bg,
                        border: `2px solid ${style.color}60`,
                        color: style.color,
                      }}
                    >
                      {w.text}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Cut boundary marker */}
        {cutSampleSecs != null && (
          <div
            style={{
              position: "absolute",
              top: 0,
              bottom: 0,
              left: LABEL_WIDTH + cutSampleSecs * pxPerSec,
              width: 2,
              background: "var(--danger)",
              opacity: 0.8,
              pointerEvents: "none",
            }}
            title={`cut at ${cutSampleSecs.toFixed(3)}s`}
          />
        )}
      </div>
    </div>
  );
}
