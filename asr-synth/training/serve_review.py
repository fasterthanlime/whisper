#!/usr/bin/env python3
"""Serve the review HTML with embedded sentence data."""
import http.server
import json
import os
import sys
import webbrowser

sentences_path = sys.argv[1] if len(sys.argv) > 1 else "data/focused_sentences.jsonl"

with open(sentences_path) as f:
    sentences = [json.loads(line) for line in f if line.strip()]

with open("data/review.html") as f:
    html = f.read()

html = html.replace("'__SENTENCES_JSON__'", json.dumps(sentences))

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())
    def log_message(self, *args):
        pass

port = 8642
print(f"Serving review at http://localhost:{port}")
print(f"{len(sentences)} sentences from {sentences_path}")
print("Press Ctrl+C to stop")
webbrowser.open(f"http://localhost:{port}")
http.server.HTTPServer(("localhost", port), Handler).serve_forever()
