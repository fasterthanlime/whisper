#!/usr/bin/env python3
"""
Render a token-aligned phonetic "layer cake" from real workspace data.

This script delegates the data collection to:
    cargo run -q -p bee-roll-readme --bin phonetic-layer-cake-data

The resulting diagram uses Qwen3 token columns as the shared grid and lays
text, audio, G2P, alignment, and ZIPA rows onto that same x-axis.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


COL_GAP = 1


@dataclass
class Span:
    start: int
    end: int
    label: str


@dataclass
class Row:
    name: str
    spans: list[Span]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="serde", help="Recording example term")
    parser.add_argument(
        "--mode",
        choices=["proj", "ops"],
        default="proj",
        help="Show token-piece projection ranges or local phoneme alignment ops",
    )
    parser.add_argument(
        "--use-text",
        action="store_true",
        help="Use recording example text instead of the transcript",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_data(term: str, use_text: bool) -> dict:
    parts = [
        'eval "$(direnv export bash)"',
        "&&",
        "cargo run -q -p bee-roll-readme --bin phonetic-layer-cake-data --",
        f"--term {shlex.quote(term)}",
    ]
    if use_text:
        parts.append("--text")
    cmd = ["zsh", "-lc", " ".join(parts)]
    result = subprocess.run(
        cmd,
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def span_width(widths: list[int], start: int, end: int) -> int:
    if start >= end:
        return 0
    return sum(widths[start:end]) + COL_GAP * (end - start - 1)


def text_row(data: dict) -> Row:
    return Row(
        "text",
        [
            Span(segment["token_start"], segment["token_end"], segment["label"])
            for segment in data["text_segments"]
        ],
    )


def audio_row(data: dict, name: str) -> Row:
    return Row(
        name,
        [
            Span(piece["token_start"], piece["token_end"], piece["audio"]["label"])
            for piece in data["token_pieces"]
            if piece["audio"] is not None
        ],
    )


def token_row(data: dict) -> Row:
    return Row(
        "Qwen3 tok",
        [Span(token["index"], token["index"] + 1, token["label"]) for token in data["tokens"]],
    )


def token_piece_row(data: dict, name: str, field: str) -> Row:
    return Row(
        name,
        [
            Span(piece["token_start"], piece["token_end"], " ".join(piece[field]))
            for piece in data["token_pieces"]
            if piece["token_start"] < piece["token_end"] and piece[field]
        ],
    )


def middle_row(data: dict, mode: str) -> Row:
    field = mode
    row_name = "proj" if mode == "proj" else "ops"
    return Row(
        row_name,
        [
            Span(piece["token_start"], piece["token_end"], piece[field])
            for piece in data["token_pieces"]
            if piece["token_start"] < piece["token_end"] and piece[field]
        ],
    )


def build_rows(data: dict, mode: str) -> list[Row]:
    return [
        text_row(data),
        audio_row(data, "audio top"),
        token_row(data),
        token_piece_row(data, "G2P IPA", "g2p_raw"),
        token_piece_row(data, "G2P norm", "g2p_normalized"),
        middle_row(data, mode),
        token_piece_row(data, "ZIPA norm", "zipa_normalized"),
        token_piece_row(data, "ZIPA raw", "zipa_raw"),
        audio_row(data, "audio bot"),
    ]


def grow_widths(widths: list[int], rows: list[Row]) -> None:
    changed = True
    while changed:
        changed = False
        for row in rows:
            for span in row.spans:
                if span.start >= span.end or not span.label:
                    continue
                needed = len(span.label) + 2
                current = span_width(widths, span.start, span.end)
                if needed <= current:
                    continue
                extra = needed - current
                cols = span.end - span.start
                base, rem = divmod(extra, cols)
                for offset, index in enumerate(range(span.start, span.end)):
                    widths[index] += base + (1 if offset < rem else 0)
                changed = True


def render_span_line(widths: list[int], row: Row) -> str:
    positions = []
    cursor = 0
    for width in widths:
        positions.append(cursor)
        cursor += width + COL_GAP
    total = cursor - COL_GAP
    canvas = [" "] * total

    for span in row.spans:
        if span.start >= span.end:
            continue
        start = positions[span.start]
        end = positions[span.end - 1] + widths[span.end - 1]
        width = end - start
        if width < 2:
            continue
        inner = width - 2
        label = span.label
        if len(label) > inner:
            label = label[:inner]
        fill = "[" + label.center(inner) + "]"
        canvas[start:end] = list(fill)

    return "".join(canvas).rstrip()


def render(data: dict, mode: str) -> str:
    rows = build_rows(data, mode)
    widths = [max(4, len(token["label"])) for token in data["tokens"]]
    grow_widths(widths, rows)

    name_width = max(len(row.name) for row in rows)
    out = [
        "Two phonetic routes over the same Qwen3 token grid",
        f"sentence  {data['sentence']}",
        "",
    ]
    for row in rows:
        out.append(f"{row.name.ljust(name_width)}  {render_span_line(widths, row)}")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    data = load_data(args.term, args.use_text)
    print(render(data, args.mode), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
