import { useEffect, useRef, useState } from "react";
import { type FeedGroup, type TraceEvent, groupByFeed } from "../cut-trace-types";
import { CutTimeline } from "./CutTimeline";

function useCutTrace() {
  const [feeds, setFeeds] = useState<Map<number, FeedGroup>>(new Map());
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const source = new EventSource("/cut-trace-api/events");

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (event) => {
      const traceEvents: TraceEvent[] = JSON.parse(event.data);
      setFeeds(groupByFeed(traceEvents));
    };

    return () => source.close();
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
  const feedEnd = group.feedEnd;
  const cutApplied = group.cutApplied;
  const feedStart = group.feedStart;
  const stableChanged = prevGroup?.feedEnd?.stable_through !== feedEnd?.stable_through;
  const audioSecs = feedStart?.audio_end_secs ?? feedEnd?.tape_end ?? null;
  const transcript = feedEnd?.transcript ?? "";
  const excerpt = transcript.length > 56 ? `${transcript.slice(0, 56)}...` : transcript;

  return (
    <button
      className={`feed-list-item${selected ? " selected" : ""}`}
      data-feed={group.feedIndex}
      onClick={onClick}
      type="button"
    >
      <div className="feed-list-item-top">
        <span className="feed-list-index">#{group.feedIndex}</span>
        {audioSecs != null && <span className="feed-list-secs">{audioSecs.toFixed(2)}s</span>}
      </div>
      <div className="feed-list-badges">
        {cutApplied?.applied && <span className="badge success">cut</span>}
        {!cutApplied?.applied && stableChanged && <span className="badge warning">stable</span>}
        {feedEnd?.has_repeated_ngrams && <span className="badge danger">loop</span>}
      </div>
      {excerpt && <div className="feed-list-excerpt">{excerpt}</div>}
    </button>
  );
}

function EventRow({ event }: { event: TraceEvent }) {
  const { event: eventName, feed_index: _feedIndex, ...fields } = event as unknown as Record<
    string,
    unknown
  >;
  const eventColor: Record<string, string> = {
    feed_start: "var(--text-muted)",
    plan_preview_decode: "var(--accent)",
    rewrite_preview: "var(--accent-strong)",
    update_preview_from: "var(--warning)",
    cut_candidate: "var(--lane-carry)",
    cut_applied: "var(--success)",
    feed_end: "var(--text-muted)",
  };

  return (
    <tr>
      <td className="event-name" style={{ color: eventColor[eventName as string] ?? "var(--text)" }}>
        {eventName as string}
      </td>
      <td className="event-fields">
        {Object.entries(fields)
          .filter(([key]) => key !== "transcript" && key !== "word_spans" && key !== "context_debug")
          .map(([key, value]) => (
            <span key={key} className="event-field">
              <span className="event-field-key">{key}=</span>
              <span>{value === null ? "null" : value === true ? "yes" : value === false ? "no" : String(value)}</span>
            </span>
          ))}
      </td>
    </tr>
  );
}

function FeedDetail({
  group,
  timelineZoom,
  timelineViewStartSec,
  onTimelineZoomChange,
  onTimelineViewStartSecChange,
}: {
  group: FeedGroup;
  timelineZoom: number;
  timelineViewStartSec: number;
  onTimelineZoomChange: (zoom: number) => void;
  onTimelineViewStartSecChange: (startSec: number) => void;
}) {
  const feedEnd = group.feedEnd;
  const cutApplied = group.cutApplied;
  const feedStart = group.feedStart;
  const zoomOptions = [0.5, 0.75, 1, 1.5, 2, 3, 4];

  return (
    <section className="detail-panel">
      <div className="detail-header">
        <div>
          <p className="detail-kicker">feed</p>
          <h2>#{group.feedIndex}</h2>
        </div>
        <div className="detail-stats">
          {feedStart && <span>audio {feedStart.audio_end_secs.toFixed(3)}s</span>}
          {feedEnd && (
            <span>
              stable {feedEnd.stable_through} / preview {feedEnd.preview_from} / tape {feedEnd.tape_end}
            </span>
          )}
          {cutApplied?.applied && (
            <span className="detail-cut">
              cut to tok {cutApplied.new_stable} at {cutApplied.cut_sample_secs?.toFixed(3)}s
            </span>
          )}
          {cutApplied && !cutApplied.applied && <span className="detail-muted">no cut</span>}
        </div>
      </div>

      {feedEnd?.transcript && <div className="transcript-card">{feedEnd.transcript}</div>}

      {feedEnd && (
        <section className="panel-card">
          <div className="panel-card-header">
            <strong>Timeline</strong>
            <div className="zoom-row">
              {zoomOptions.map((option) => (
                <button
                  key={option}
                  className={timelineZoom === option ? "primary" : ""}
                  onClick={() => onTimelineZoomChange(option)}
                  type="button"
                >
                  {option}x
                </button>
              ))}
            </div>
          </div>
          <CutTimeline
            wordSpans={feedEnd.word_spans}
            cutSampleSecs={cutApplied?.applied ? cutApplied.cut_sample_secs : null}
            zoom={timelineZoom}
            viewStartSec={timelineViewStartSec}
            onViewStartSecChange={onTimelineViewStartSecChange}
          />
        </section>
      )}

      <section className="panel-card">
        <div className="panel-card-header">
          <strong>Events</strong>
        </div>
        <div className="event-table-wrap">
          <table className="event-table">
            <tbody>
              {group.events.map((event, index) => (
                <EventRow key={index} event={event} />
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {cutApplied?.context_debug && (
        <details className="panel-card">
          <summary>context_debug</summary>
          <pre className="context-debug">{cutApplied.context_debug}</pre>
        </details>
      )}
    </section>
  );
}

export function CutTracePanel() {
  const { feeds, connected } = useCutTrace();
  const [selectedFeed, setSelectedFeed] = useState<number | null>(null);
  const [timelineZoom, setTimelineZoom] = useState(1);
  const [timelineViewStartSec, setTimelineViewStartSec] = useState(0);
  const feedListRef = useRef<HTMLDivElement>(null);

  const sortedIndices = Array.from(feeds.keys()).sort((a, b) => a - b);
  const selectedGroup = selectedFeed != null ? feeds.get(selectedFeed) ?? null : null;

  const totalCuts = Array.from(feeds.values()).filter((group) => group.cutApplied?.applied).length;
  const totalLoops = Array.from(feeds.values()).filter((group) => group.feedEnd?.has_repeated_ngrams).length;

  const lastAutoRef = useRef<number | null>(null);
  useEffect(() => {
    if (sortedIndices.length === 0) return;
    const latest = sortedIndices[sortedIndices.length - 1];
    if (selectedFeed === null && lastAutoRef.current !== latest) {
      setSelectedFeed(latest);
      lastAutoRef.current = latest;
    }
  }, [selectedFeed, sortedIndices]);

  useEffect(() => {
    if (selectedFeed == null || !feedListRef.current) return;
    const element = feedListRef.current.querySelector(`[data-feed="${selectedFeed}"]`);
    element?.scrollIntoView({ block: "nearest" });
  }, [selectedFeed]);

  const sortedIndicesRef = useRef(sortedIndices);
  sortedIndicesRef.current = sortedIndices;
  const selectedFeedRef = useRef(selectedFeed);
  selectedFeedRef.current = selectedFeed;

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key !== "j" && event.key !== "k") return;
      const indices = sortedIndicesRef.current;
      if (indices.length === 0) return;
      const current = selectedFeedRef.current;
      const position = current != null ? indices.indexOf(current) : -1;
      const nextPosition =
        event.key === "j" ? Math.min(position + 1, indices.length - 1) : Math.max(position - 1, 0);
      setSelectedFeed(indices[nextPosition]);
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="trace-layout">
      <aside className="trace-sidebar">
        <div className="sidebar-header">
          <div>
            <p className="sidebar-kicker">cut trace</p>
            <h1>Live feed viewer</h1>
          </div>
          <span className={`connection-pill ${connected ? "online" : "offline"}`}>
            {connected ? "live" : "offline"}
          </span>
        </div>

        <div className="sidebar-stats">
          <div>
            <strong>{sortedIndices.length}</strong>
            <span>feeds</span>
          </div>
          <div>
            <strong>{totalCuts}</strong>
            <span>cuts</span>
          </div>
          <div>
            <strong>{totalLoops}</strong>
            <span>loops</span>
          </div>
        </div>

        <div className="sidebar-help">
          <span>`j` / `k` moves through feeds</span>
          <span>SSE source: `/cut-trace-api/events`</span>
        </div>

        <div ref={feedListRef} className="feed-list">
          {sortedIndices.map((feedIndex, index) => {
            const group = feeds.get(feedIndex)!;
            const prevGroup = index > 0 ? (feeds.get(sortedIndices[index - 1]) ?? null) : null;
            return (
              <FeedListItem
                key={feedIndex}
                group={group}
                selected={selectedFeed === feedIndex}
                prevGroup={prevGroup}
                onClick={() => setSelectedFeed(feedIndex)}
              />
            );
          })}
        </div>
      </aside>

      <main className="trace-main">
        {selectedGroup ? (
          <FeedDetail
            group={selectedGroup}
            timelineZoom={timelineZoom}
            timelineViewStartSec={timelineViewStartSec}
            onTimelineZoomChange={setTimelineZoom}
            onTimelineViewStartSecChange={setTimelineViewStartSec}
          />
        ) : (
          <section className="empty-state">
            <h2>No feed selected</h2>
            <p>
              {sortedIndices.length === 0
                ? "Waiting for trace data. Start debug/cut-trace-server.js and point bee-roll at a trace file."
                : "Select a feed from the left column."}
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
