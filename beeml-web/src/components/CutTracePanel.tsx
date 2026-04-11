import { useEffect, useRef, useState } from "react";
import {
  type FeedGroup,
  type TraceEvent,
  groupByFeed,
} from "../cut-trace-types";
import { CutTimeline } from "./CutTimeline";

function useCutTrace() {
  const [feeds, setFeeds] = useState<Map<number, FeedGroup>>(new Map());
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource("/cut-trace-api/events");
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e) => {
      const events: TraceEvent[] = JSON.parse(e.data);
      setFeeds(groupByFeed(events));
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  return { feeds, connected };
}

function FeedListItem({
  group,
  selected,
  prevGroup,
  onClick,
}: {
  group: FeedGroup;
  selected: boolean;
  prevGroup: FeedGroup | null;
  onClick: () => void;
}) {
  const fe = group.feedEnd;
  const ca = group.cutApplied;
  const fs = group.feedStart;

  const stableChanged =
    prevGroup?.feedEnd?.stable_through !== fe?.stable_through;
  const loopy = fe?.has_repeated_ngrams;
  const cutApplied = ca?.applied === true;
  const audioSecs = fs?.audio_end_secs ?? fe?.tape_end ?? null;

  const transcript = fe?.transcript ?? "";
  const excerpt = transcript.length > 48 ? transcript.slice(0, 48) + "…" : transcript;

  return (
    <button
      data-feed={group.feedIndex}
      onClick={onClick}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "6px 10px",
        borderRadius: 0,
        borderLeft: selected ? "3px solid var(--accent)" : "3px solid transparent",
        borderTop: "none",
        borderRight: "none",
        borderBottom: "1px solid var(--border)",
        background: selected ? "var(--bg-surface-hover)" : "transparent",
        fontFamily: "monospace",
        fontSize: 12,
        lineHeight: 1.5,
        cursor: "pointer",
      }}
    >
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <span style={{ color: "var(--text-muted)", minWidth: 36 }}>
          #{group.feedIndex}
        </span>
        {audioSecs != null && (
          <span style={{ color: "var(--text-dim)" }}>
            {audioSecs.toFixed(2)}s
          </span>
        )}
        {cutApplied && (
          <span style={{ color: "var(--success)", fontWeight: 700 }}>CUT</span>
        )}
        {stableChanged && !cutApplied && (
          <span style={{ color: "var(--warning)", fontSize: 10 }}>Δstable</span>
        )}
        {loopy && (
          <span style={{ color: "var(--danger)", fontSize: 10 }}>LOOP</span>
        )}
      </div>
      {excerpt && (
        <div
          style={{
            color: "var(--text-muted)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            marginTop: 2,
          }}
        >
          {excerpt}
        </div>
      )}
    </button>
  );
}

function EventRow({ ev }: { ev: TraceEvent }) {
  const { event, feed_index: _fi, ...rest } = ev as unknown as Record<string, unknown>;
  const eventColor: Record<string, string> = {
    feed_start: "var(--text-dim)",
    plan_preview_decode: "var(--accent)",
    rewrite_preview: "var(--accent)",
    update_preview_from: "var(--warning)",
    cut_candidate: "var(--lane-reranker)",
    cut_applied: "var(--success)",
    feed_end: "var(--text-dim)",
  };

  return (
    <tr>
      <td
        style={{
          color: eventColor[event as string] ?? "var(--text)",
          fontWeight: 600,
          paddingRight: 16,
          whiteSpace: "nowrap",
          verticalAlign: "top",
        }}
      >
        {event as string}
      </td>
      <td style={{ color: "var(--text-muted)", fontFamily: "monospace", fontSize: 11 }}>
        {Object.entries(rest)
          .filter(([k]) => k !== "transcript" && k !== "word_spans" && k !== "context_debug")
          .map(([k, v]) => (
            <span key={k} style={{ marginRight: 12 }}>
              <span style={{ color: "var(--text-dim)" }}>{k}=</span>
              <span style={{ color: "var(--text)" }}>
                {v === null ? "null" : v === true ? "✓" : v === false ? "✗" : String(v)}
              </span>
            </span>
          ))}
      </td>
    </tr>
  );
}

function FeedDetail({ group }: { group: FeedGroup }) {
  const fe = group.feedEnd;
  const ca = group.cutApplied;
  const fs = group.feedStart;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, padding: 16 }}>
      {/* Header */}
      <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
        <h2 style={{ fontSize: 16, fontWeight: 700 }}>
          Feed #{group.feedIndex}
        </h2>
        {fs && (
          <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>
            audio → {fs.audio_end_secs.toFixed(3)}s
          </span>
        )}
        {fe && (
          <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>
            stable={fe.stable_through} preview_from={fe.preview_from} tape_end={fe.tape_end}
          </span>
        )}
        {ca?.applied && (
          <span style={{ color: "var(--success)", fontWeight: 700 }}>
            CUT → tok {ca.new_stable} @ {ca.cut_sample_secs?.toFixed(3)}s
          </span>
        )}
        {ca && !ca.applied && (
          <span style={{ color: "var(--text-dim)" }}>no cut</span>
        )}
        {fe?.has_repeated_ngrams && (
          <span style={{ color: "var(--danger)", fontWeight: 700 }}>
            ⚠ repeated n-grams
          </span>
        )}
      </div>

      {/* Transcript */}
      {fe?.transcript && (
        <div
          style={{
            fontFamily: "monospace",
            fontSize: 13,
            padding: "8px 12px",
            background: "var(--bg-surface)",
            borderRadius: 4,
            border: "1px solid var(--border)",
          }}
        >
          {fe.transcript}
        </div>
      )}

      {/* Timeline */}
      {fe?.word_spans && fe.word_spans.length > 0 && (
        <CutTimeline
          wordSpans={fe.word_spans}
          cutSampleSecs={ca?.applied ? ca.cut_sample_secs : null}
        />
      )}

      {/* Events table */}
      <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
        <tbody>
          {group.events.map((ev, i) => (
            <EventRow key={i} ev={ev} />
          ))}
        </tbody>
      </table>

      {/* context_debug from cut_applied, if any */}
      {ca?.context_debug && (
        <details>
          <summary style={{ color: "var(--text-muted)", cursor: "pointer", fontSize: 12 }}>
            context_debug
          </summary>
          <pre
            style={{
              fontSize: 11,
              color: "var(--text-muted)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
              marginTop: 4,
            }}
          >
            {ca.context_debug}
          </pre>
        </details>
      )}
    </div>
  );
}

export function CutTracePanel() {
  const { feeds, connected } = useCutTrace();
  const [selectedFeed, setSelectedFeed] = useState<number | null>(null);

  const sortedIndices = Array.from(feeds.keys()).sort((a, b) => a - b);

  // Auto-select latest feed when new data arrives
  const lastAutoRef = useRef<number | null>(null);
  useEffect(() => {
    if (sortedIndices.length > 0) {
      const latest = sortedIndices[sortedIndices.length - 1];
      if (lastAutoRef.current !== latest && selectedFeed === null) {
        setSelectedFeed(latest);
        lastAutoRef.current = latest;
      }
    }
  }, [sortedIndices.length, selectedFeed]);

  const selectedGroup = selectedFeed != null ? feeds.get(selectedFeed) ?? null : null;

  // Scroll selected feed into view
  const feedListRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selectedFeed == null || !feedListRef.current) return;
    const el = feedListRef.current.querySelector(`[data-feed="${selectedFeed}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [selectedFeed]);

  // j/k navigation
  const sortedIndicesRef = useRef(sortedIndices);
  sortedIndicesRef.current = sortedIndices;
  const selectedFeedRef = useRef(selectedFeed);
  selectedFeedRef.current = selectedFeed;
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key !== "j" && e.key !== "k") return;
      const indices = sortedIndicesRef.current;
      if (indices.length === 0) return;
      const cur = selectedFeedRef.current;
      const pos = cur != null ? indices.indexOf(cur) : -1;
      const next = e.key === "j"
        ? Math.min(pos + 1, indices.length - 1)
        : Math.max(pos - 1, 0);
      setSelectedFeed(indices[next]);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* Feed list */}
      <div
        ref={feedListRef}
        style={{
          width: 220,
          flexShrink: 0,
          borderRight: "1px solid var(--border)",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            padding: "8px 10px",
            fontSize: 11,
            color: "var(--text-dim)",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>{sortedIndices.length} feeds</span>
          <span style={{ color: connected ? "var(--success)" : "var(--danger)" }}>
            {connected ? "live" : "offline"}
          </span>
        </div>
        {sortedIndices.map((fi, i) => {
          const group = feeds.get(fi)!;
          const prev = i > 0 ? (feeds.get(sortedIndices[i - 1]) ?? null) : null;
          return (
            <FeedListItem
              key={fi}
              group={group}
              selected={selectedFeed === fi}
              prevGroup={prev}
              onClick={() => setSelectedFeed(fi)}
            />
          );
        })}
      </div>

      {/* Detail panel */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {selectedGroup ? (
          <FeedDetail group={selectedGroup} />
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "var(--text-dim)",
            }}
          >
            {sortedIndices.length === 0
              ? "Waiting for trace data… (is cut-trace-server.js running?)"
              : "Select a feed"}
          </div>
        )}
      </div>
    </div>
  );
}
