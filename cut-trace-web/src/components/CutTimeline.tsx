import { type UIEvent, useCallback, useEffect, useRef } from "react";
import type { WordSpan } from "../cut-trace-types";

const LABEL_WIDTH = 72;
const LANE_HEIGHT = 36;
const RULER_HEIGHT = 24;
const BASE_PX_PER_SEC = 120;

const REGION_STYLE = {
  stable: {
    label: "stable",
    color: "var(--lane-stable)",
    bg: "var(--lane-stable-bg)",
  },
  carry: {
    label: "carry",
    color: "var(--lane-carry)",
    bg: "var(--lane-carry-bg)",
  },
  preview: {
    label: "preview",
    color: "var(--lane-preview)",
    bg: "var(--lane-preview-bg)",
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
  selectedWordKey,
  onWordSelect,
}: {
  wordSpans: WordSpan[];
  cutSampleSecs?: number | null;
  zoom: number;
  viewStartSec: number;
  onViewStartSecChange?: (startSec: number) => void;
  selectedWordKey?: string | null;
  onWordSelect?: (word: WordSpan) => void;
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

  const lanesMap = new Map<string, WordSpan[]>();
  for (const region of REGION_ORDER) lanesMap.set(region, []);
  for (const word of wordSpans) {
    if (word.start_secs != null && word.end_secs != null) {
      lanesMap.get(word.region)?.push(word);
    }
  }

  const lanes: Lane[] = REGION_ORDER.filter(
    (region) => (lanesMap.get(region)?.length ?? 0) > 0,
  ).map((region) => ({ region, words: lanesMap.get(region)! }));

  let maxSecs = 0;
  for (const word of wordSpans) {
    if (word.end_secs != null && word.end_secs > maxSecs) maxSecs = word.end_secs;
  }
  if (cutSampleSecs != null && cutSampleSecs > maxSecs) maxSecs = cutSampleSecs;
  maxSecs = Math.max(maxSecs, 0.5);

  const totalWidth = maxSecs * pxPerSec;
  const contentHeight = RULER_HEIGHT + lanes.length * LANE_HEIGHT;

  const ticks: number[] = [];
  for (let t = 0; t <= maxSecs + 0.5; t += 0.5) {
    ticks.push(Number(t.toFixed(2)));
  }

  return (
    <div ref={containerRef} className="timeline hide-scrollbar" onScroll={handleScroll}>
      <div
        className="timeline-inner"
        style={{ width: LABEL_WIDTH + totalWidth, minHeight: contentHeight }}
      >
        <div className="timeline-ruler-row" style={{ height: RULER_HEIGHT }}>
          <div className="timeline-ruler-label" style={{ width: LABEL_WIDTH }} />
          <div className="timeline-ruler" style={{ width: totalWidth, height: RULER_HEIGHT }}>
            {ticks.map((tick) => (
              <div key={tick} className="timeline-tick" style={{ left: tick * pxPerSec }}>
                {tick % 1 === 0 && <span>{tick}s</span>}
              </div>
            ))}
          </div>
        </div>

        {lanes.map(({ region, words }) => {
          const style = REGION_STYLE[region];
          return (
            <div key={region} className="timeline-lane" style={{ height: LANE_HEIGHT }}>
              <div className="timeline-lane-label" style={{ width: LABEL_WIDTH, color: style.color }}>
                {style.label}
              </div>
              <div className="timeline-lane-tokens" style={{ width: totalWidth, height: 28 }}>
                {words.map((word, index) => {
                  const left = word.start_secs! * pxPerSec;
                  const width = Math.max((word.end_secs! - word.start_secs!) * pxPerSec, 2);
                  const wordKey = `${word.start}-${word.end}-${word.text}`;
                  return (
                    <button
                      key={wordKey}
                      className="timeline-token"
                      title={`tok ${word.start}-${word.end}  ${word.start_secs!.toFixed(3)}s-${word.end_secs!.toFixed(3)}s`}
                      type="button"
                      onClick={() => onWordSelect?.(word)}
                      data-selected={selectedWordKey === wordKey ? "yes" : "no"}
                      style={{
                        left,
                        width,
                        background: style.bg,
                        border: `1px solid ${style.color}`,
                        color: style.color,
                      }}
                    >
                      {word.text}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}

        {cutSampleSecs != null && (
          <div
            className="timeline-cut-line"
            style={{ left: LABEL_WIDTH + cutSampleSecs * pxPerSec }}
            title={`cut at ${cutSampleSecs.toFixed(3)}s`}
          />
        )}
      </div>
    </div>
  );
}
