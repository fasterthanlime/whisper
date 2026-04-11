#!/usr/bin/env node
/**
 * debug/cut-trace-server.js
 *
 * Serves bee-roll cut-trace events over SSE at http://localhost:7701/cut-trace-api/events
 * The cut-trace-web Vite dev server proxies /cut-trace-api → here.
 *
 * Usage:
 *   BEE_ROLL_CUT_TRACE=/tmp/cuts.jsonl node debug/cut-trace-server.js
 */

"use strict";

const http = require("http");
const fs = require("fs");

const PORT = 7701;
const TRACE_FILE = process.env.BEE_ROLL_CUT_TRACE;

if (!TRACE_FILE) {
  console.error("Error: BEE_ROLL_CUT_TRACE is not set");
  console.error(
    "  BEE_ROLL_CUT_TRACE=/tmp/cuts.jsonl node debug/cut-trace-server.js"
  );
  process.exit(1);
}

function readTrace() {
  try {
    const raw = fs.readFileSync(TRACE_FILE, "utf8");
    return raw
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

// Active SSE clients
const clients = new Set();

function broadcast(events) {
  const payload = `data: ${JSON.stringify(events)}\n\n`;
  for (const res of clients) {
    try {
      res.write(payload);
    } catch {}
  }
}

// Watch the trace file; debounce rapid writes
let debounceTimer = null;
let watcher = null;

function startWatch() {
  if (watcher) return;
  try {
    watcher = fs.watch(TRACE_FILE, () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => broadcast(readTrace()), 150);
    });
    watcher.on("error", () => {
      watcher = null;
      // File may have been recreated (truncated); re-arm after a moment
      setTimeout(startWatch, 500);
    });
  } catch {
    // File doesn't exist yet; retry
    setTimeout(startWatch, 500);
  }
}

startWatch();

const server = http.createServer((req, res) => {
  if (req.url === "/cut-trace-api/events") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    });
    clients.add(res);
    // Send current snapshot immediately on connect
    res.write(`data: ${JSON.stringify(readTrace())}\n\n`);
    req.on("close", () => clients.delete(res));
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`\nCut-trace SSE server listening on port ${PORT}`);
  console.log(`Trace file: ${TRACE_FILE}`);
  console.log(
    `\nOpen the viewer: http://localhost:5174  (requires cut-trace-web dev server)\n`
  );
});
