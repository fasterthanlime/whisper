#!/usr/bin/env python3
"""
Render a horizontal phonetic timeline from a small JSON spec.

This is meant for diagrams where the sequence and timing matter more than
stacked field/value dumps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawSpan:
    token: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class NormalizedToken:
    token: str
    source_start: int
    source_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to the JSON spec")
    parser.add_argument("--out", type=Path, help="Optional output file")
    return parser.parse_args()


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def ms_to_col(ms: int, ms_per_col: int) -> int:
    return ms // ms_per_col


def ms_to_end_col(ms: int, ms_per_col: int) -> int:
    return ceil_div(ms, ms_per_col)


def overlay_centered(buffer: list[str], start: int, end: int, text: str) -> None:
    if end <= start or not text:
        return
    width = end - start
    if len(text) > width:
        if width == 1:
            text = text[:1]
        elif width == 2:
            text = text[:2]
        else:
            text = text[: width - 1] + ">"
    offset = max(0, (width - len(text)) // 2)
    for idx, ch in enumerate(text):
        pos = start + offset + idx
        if start <= pos < end:
            buffer[pos] = ch


def overlay_fit(buffer: list[str], desired_start: int, text: str) -> None:
    if not text:
        return
    width = len(buffer)
    if width <= 0:
        return
    start = max(0, min(desired_start, width - len(text)))
    end = start + len(text)
    if any(buffer[idx] != " " for idx in range(start, end)):
        return
    for idx, ch in enumerate(text):
        buffer[start + idx] = ch


def build_box_row(width: int, segments: list[tuple[int, int, str]]) -> str:
    cells = [" "] * width
    for start, end, label in segments:
        if end <= start:
            continue
        cells[start] = "["
        for idx in range(start + 1, end - 1):
            cells[idx] = "-"
        if end - 1 < width:
            cells[end - 1] = "]"
        overlay_centered(cells, start + 1, max(start + 1, end - 1), label)
    return "".join(cells).rstrip()


def build_time_labels(width: int, boundary_cols: list[int], boundary_labels: list[str]) -> str:
    cells = [" "] * width
    for col, label in zip(boundary_cols, boundary_labels):
        overlay_fit(cells, col, label)
    return "".join(cells).rstrip()


def build_tick_row(width: int, boundary_cols: list[int]) -> str:
    cells = [" "] * width
    for idx, col in enumerate(boundary_cols):
        if col >= width:
            col = width - 1
        cells[col] = "|"
        if idx + 1 >= len(boundary_cols):
            continue
        next_col = min(width - 1, boundary_cols[idx + 1])
        for mid in range(col + 1, next_col):
            cells[mid] = "-"
    return "".join(cells).rstrip()


def render(spec: dict) -> str:
    ms_per_col = int(spec["ms_per_col"])
    raw_spans = [
        RawSpan(
            token=str(item["token"]),
            start_ms=int(item["start_ms"]),
            end_ms=int(item["end_ms"]),
        )
        for item in spec["raw_spans"]
    ]
    normalized = [
        NormalizedToken(
            token=str(item["token"]),
            source_start=int(item["source_start"]),
            source_end=int(item["source_end"]),
        )
        for item in spec["normalized_tokens"]
    ]

    start_ms = min(span.start_ms for span in raw_spans)
    end_ms = max(span.end_ms for span in raw_spans)
    width = max(1, ms_to_end_col(end_ms - start_ms, ms_per_col))

    def rel_col(ms: int) -> int:
        return ms_to_col(ms - start_ms, ms_per_col)

    def rel_end_col(ms: int) -> int:
        return ms_to_end_col(ms - start_ms, ms_per_col)

    raw_segments = [
        (rel_col(span.start_ms), rel_end_col(span.end_ms), span.token) for span in raw_spans
    ]

    grouped: dict[tuple[int, int], list[NormalizedToken]] = {}
    for token in normalized:
        grouped.setdefault((token.source_start, token.source_end), []).append(token)

    norm_segments: list[tuple[int, int, str]] = []
    source_segments: list[tuple[int, int, str]] = []

    def source_label(source_start: int, source_end: int, seg_width: int) -> str:
        candidates = [
            f"{source_start}..{source_end}",
            f"{source_start}-{source_end}",
            f"{source_start}",
        ]
        interior = max(0, seg_width - 2)
        for candidate in candidates:
            if len(candidate) <= interior:
                return candidate
        return candidates[-1][:interior] if interior > 0 else ""

    for (source_start, source_end), group in grouped.items():
        source_start_ms = raw_spans[source_start].start_ms
        source_end_ms = raw_spans[source_end - 1].end_ms
        group_start = rel_col(source_start_ms)
        group_end = rel_end_col(source_end_ms)
        group_width = max(1, group_end - group_start)
        part_width = max(1, group_width // len(group))
        cursor = group_start
        for idx, token in enumerate(group):
            seg_start = cursor
            seg_end = group_end if idx == len(group) - 1 else min(group_end, cursor + part_width)
            if seg_end <= seg_start:
                seg_end = min(width, seg_start + 1)
            norm_segments.append((seg_start, seg_end, token.token))
            source_segments.append(
                (seg_start, seg_end, source_label(source_start, source_end, seg_end - seg_start))
            )
            cursor = seg_end

    boundary_ms = sorted({span.start_ms for span in raw_spans} | {raw_spans[-1].end_ms})
    boundary_cols = [min(width - 1, rel_col(ms)) for ms in boundary_ms]
    boundary_labels = [f"{ms / 1000:.2f}" for ms in boundary_ms]

    label_width = max(len("time"), len("ticks"), len("raw ZIPA"), len("normalized"), len("source"))
    title = str(spec.get("title", "")).rstrip()
    lines = []
    if title:
        lines.append(title)
    lines.extend(
        [
            f"{'time'.ljust(label_width)} : {build_time_labels(width, boundary_cols, boundary_labels)}",
            f"{'ticks'.ljust(label_width)} : {build_tick_row(width, boundary_cols)}",
            f"{'raw ZIPA'.ljust(label_width)} : {build_box_row(width, raw_segments)}",
            f"{'normalized'.ljust(label_width)} : {build_box_row(width, norm_segments)}",
            f"{'source'.ljust(label_width)} : {build_box_row(width, source_segments)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    output = render(spec)
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
