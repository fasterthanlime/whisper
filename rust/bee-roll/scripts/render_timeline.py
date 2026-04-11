#!/usr/bin/env python3
"""
Render deterministic ASCII timeline diagrams from a small JSON spec.

The goal is to stop hand-aligning timeline diagrams in markdown.

Usage:
    python3 rust/bee-roll/scripts/render_timeline.py SPEC.json

Optional:
    python3 rust/bee-roll/scripts/render_timeline.py SPEC.json --out OUTPUT.txt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Span:
    start_ms: int
    end_ms: int
    fill: str = " "
    text: str | None = None


@dataclass(frozen=True)
class Row:
    label: str
    spans: list[Span]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to the JSON spec")
    parser.add_argument("--out", type=Path, help="Write output to a file instead of stdout")
    return parser.parse_args()


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def ms_to_col(ms: int, ms_per_col: int) -> int:
    return ms // ms_per_col


def ms_to_end_col(ms: int, ms_per_col: int) -> int:
    return ceil_div(ms, ms_per_col)


def parse_span(raw: dict[str, Any]) -> Span:
    return Span(
        start_ms=int(raw["start_ms"]),
        end_ms=int(raw["end_ms"]),
        fill=str(raw.get("fill", " "))[:1],
        text=raw.get("text"),
    )


def parse_row(raw: dict[str, Any]) -> Row:
    return Row(
        label=str(raw["label"]),
        spans=[parse_span(span) for span in raw.get("spans", [])],
    )


def overlay_centered(buffer: list[str], start: int, end: int, text: str) -> None:
    if not text:
        return
    width = end - start
    if width <= 0:
        return
    if len(text) > width:
        if width == 1:
            text = text[:1]
        elif width == 2:
            text = text[:2]
        else:
            text = text[: width - 1] + ">"
    text_start = start + max(0, (width - len(text)) // 2)
    for idx, ch in enumerate(text):
        pos = text_start + idx
        if start <= pos < end:
            buffer[pos] = ch


def render_row(width: int, ms_per_col: int, row: Row) -> str:
    cells = [" "] * width
    for span in row.spans:
        start = clamp(ms_to_col(span.start_ms, ms_per_col), 0, width)
        end = clamp(ms_to_end_col(span.end_ms, ms_per_col), 0, width)
        if end <= start and span.end_ms > span.start_ms:
            end = min(width, start + 1)
        for idx in range(start, end):
            cells[idx] = span.fill
        if span.text:
            overlay_centered(cells, start, end, span.text)
    return "".join(cells).rstrip()


def render_tick_row(width: int, ms_per_col: int, tick_ms: int) -> str:
    cells = [" "] * width
    tick = 0
    total_ms = width * ms_per_col
    while tick <= total_ms:
        col = clamp(ms_to_col(tick, ms_per_col), 0, width - 1)
        cells[col] = "|"
        tick += tick_ms
    return "".join(cells).rstrip()


def place_text_if_free(buffer: list[str], pos: int, text: str) -> None:
    if not text:
        return
    if pos < 0:
        text = text[-pos:]
        pos = 0
    if pos >= len(buffer):
        return
    end = min(len(buffer), pos + len(text))
    for idx in range(pos, end):
        if buffer[idx] != " ":
            return
    for idx, ch in enumerate(text[: end - pos]):
        buffer[pos + idx] = ch


def render_time_labels(width: int, ms_per_col: int, label_ms: int) -> str:
    cells = [" "] * width
    tick = 0
    total_ms = width * ms_per_col
    while tick <= total_ms:
        label = f"{tick / 1000:.1f}s"
        col = ms_to_col(tick, ms_per_col)
        place_text_if_free(cells, col, label)
        tick += label_ms
    return "".join(cells).rstrip()


def render_step(step: dict[str, Any], *, ms_per_col: int, tick_ms: int, label_ms: int) -> str:
    title = str(step["title"])
    timeline_end_ms = int(step["timeline_end_ms"])
    width = ms_to_end_col(timeline_end_ms, ms_per_col)
    rows = [parse_row(raw_row) for raw_row in step.get("rows", [])]
    label_width = max(
        [len("time"), len("ticks"), *(len(row.label) for row in rows)],
        default=0,
    )

    lines = [title]
    lines.append(f"{'time'.ljust(label_width)} : {render_time_labels(width, ms_per_col, label_ms)}")
    lines.append(f"{'ticks'.ljust(label_width)} : {render_tick_row(width, ms_per_col, tick_ms)}")
    for row in rows:
        lines.append(f"{row.label.ljust(label_width)} : {render_row(width, ms_per_col, row)}")
    return "\n".join(lines).rstrip()


def validate_spec(spec: dict[str, Any]) -> None:
    if "steps" not in spec or not isinstance(spec["steps"], list) or not spec["steps"]:
        raise ValueError("spec must contain a non-empty 'steps' list")
    if int(spec["ms_per_col"]) <= 0:
        raise ValueError("ms_per_col must be positive")
    if int(spec["tick_ms"]) <= 0:
        raise ValueError("tick_ms must be positive")
    if int(spec["label_ms"]) <= 0:
        raise ValueError("label_ms must be positive")


def render_document(spec: dict[str, Any]) -> str:
    validate_spec(spec)
    ms_per_col = int(spec["ms_per_col"])
    tick_ms = int(spec["tick_ms"])
    label_ms = int(spec["label_ms"])
    parts = [
        render_step(step, ms_per_col=ms_per_col, tick_ms=tick_ms, label_ms=label_ms)
        for step in spec["steps"]
    ]
    return "\n\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    rendered = render_document(spec)
    if args.out:
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
