#!/usr/bin/env python3
"""
Render a concrete G2P-vs-ZIPA comparison from the real workspace binary.

This script delegates to:
    cargo run -p bee-zipa-mlx --bin zipa-compare-espeak

Then it parses one result row, aligns the normalized token sequences, and
prints a human-readable ASCII report suitable for README snippets.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComparisonRow:
    term: str
    text: str
    zipa_raw: list[str]
    espeak_raw: list[str]
    zipa_normalized: list[str]
    espeak_normalized: list[str]
    raw_similarity: str
    normalized_similarity: str
    normalized_feature_similarity: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", required=True, help="Term to select from recording_examples")
    parser.add_argument(
        "--use-transcript",
        action="store_true",
        default=True,
        help="Use the transcript field from recording_examples (default: true)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=14,
        help="Alignment columns per block",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_compare(term: str, use_transcript: bool) -> str:
    cmd = [
        "zsh",
        "-lc",
        " ".join(
            [
                'eval "$(direnv export bash)"',
                "&&",
                "cargo run -q -p bee-zipa-mlx --bin zipa-compare-espeak --",
                f"--term {shlex.quote(term)}",
                "--limit 1",
                "--use-transcript" if use_transcript else "",
            ]
        ).strip(),
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_compare_output(text: str) -> ComparisonRow:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("summary."):
            continue
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        values[key] = value

    required = [
        "term",
        "text",
        "zipa_raw",
        "espeak_raw",
        "zipa_normalized",
        "espeak_normalized",
        "raw_similarity",
        "normalized_similarity",
        "normalized_feature_similarity",
    ]
    missing = [key for key in required if key not in values]
    if missing:
        raise SystemExit(f"missing expected fields from compare output: {', '.join(missing)}")

    return ComparisonRow(
        term=values["term"],
        text=values["text"],
        zipa_raw=values["zipa_raw"].split(),
        espeak_raw=values["espeak_raw"].split(),
        zipa_normalized=values["zipa_normalized"].split(),
        espeak_normalized=values["espeak_normalized"].split(),
        raw_similarity=values["raw_similarity"],
        normalized_similarity=values["normalized_similarity"],
        normalized_feature_similarity=values["normalized_feature_similarity"],
    )


def align_tokens(left: list[str], right: list[str]) -> list[tuple[str | None, str | None]]:
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    back: list[list[tuple[int, int] | None]] = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = (i - 1, 0)
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = (0, j - 1)

    for i in range(1, rows):
        for j in range(1, cols):
            sub_cost = 0 if left[i - 1] == right[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + sub_cost, (i - 1, j - 1)),
                (dp[i - 1][j] + 1, (i - 1, j)),
                (dp[i][j - 1] + 1, (i, j - 1)),
            ]
            best_score, best_prev = min(choices, key=lambda item: item[0])
            dp[i][j] = best_score
            back[i][j] = best_prev

    out: list[tuple[str | None, str | None]] = []
    i = len(left)
    j = len(right)
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            out.append((left[i - 1], right[j - 1]))
        elif pi == i - 1 and pj == j:
            out.append((left[i - 1], None))
        else:
            out.append((None, right[j - 1]))
        i, j = pi, pj

    out.reverse()
    return out


def diff_symbol(left: str | None, right: str | None) -> str:
    if left is None:
        return ">"
    if right is None:
        return "<"
    if left == right:
        return "|"
    return "x"


def format_token(token: str | None) -> str:
    return token if token is not None else "·"


def render_alignment(
    left_label: str,
    right_label: str,
    pairs: list[tuple[str | None, str | None]],
    cols_per_block: int,
) -> str:
    lines: list[str] = []
    diff_label = "diff"
    label_width = max(len(left_label), len(right_label), len(diff_label))
    for start in range(0, len(pairs), cols_per_block):
        chunk = pairs[start : start + cols_per_block]
        widths = [
            max(len(format_token(left)), len(format_token(right)), 1) for left, right in chunk
        ]

        def render_row(label: str, values: list[str]) -> str:
            body = " ".join(value.ljust(width) for value, width in zip(values, widths))
            return f"{label.ljust(label_width)} : {body}".rstrip()

        left_values = [format_token(left) for left, _ in chunk]
        right_values = [format_token(right) for _, right in chunk]
        diff_values = [diff_symbol(left, right) for left, right in chunk]

        lines.append(render_row(left_label, left_values))
        lines.append(render_row(right_label, right_values))
        lines.append(render_row(diff_label, diff_values))
        if start + cols_per_block < len(pairs):
            lines.append("")

    return "\n".join(lines)


def render_report(row: ComparisonRow, cols_per_block: int) -> str:
    aligned = align_tokens(row.espeak_normalized, row.zipa_normalized)
    parts = [
        f"term       : {row.term}",
        f"text       : {row.text}",
        "",
        "raw",
        f"g2p        : {' '.join(row.espeak_raw)}",
        f"zipa       : {' '.join(row.zipa_raw)}",
        "",
        "normalized + aligned",
        render_alignment("g2p", "zipa", aligned, cols_per_block),
        "",
        "scores",
        f"raw        : {row.raw_similarity}",
        f"normalized : {row.normalized_similarity}",
        f"feat norm  : {row.normalized_feature_similarity}",
        "",
        "diff legend",
        "| = exact token match",
        "x = substituted token",
        "< = token only on the g2p side",
        "> = token only on the zipa side",
    ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    output = run_compare(args.term, args.use_transcript)
    row = parse_compare_output(output)
    print(render_report(row, args.cols), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
