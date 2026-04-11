#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentencepiece",
#   "torch",
#   "transformers",
# ]
# ///
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
STRESS_CHARS = str.maketrans("", "", "ˈˌ")


@dataclass
class WordResult:
    word: str
    charsiu_raw: str
    charsiu_normalized: str
    espeak_raw: str
    espeak_normalized: str


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_ipa(text: str) -> str:
    normalized = text.translate(STRESS_CHARS)
    normalized = normalized.replace("\u200d", "").replace("‍", "")
    return " ".join(normalized.split())


def espeak_binary() -> str:
    for candidate in ("espeak-ng", "/opt/homebrew/bin/espeak-ng"):
        if shutil.which(candidate) or candidate.startswith("/"):
            if candidate.startswith("/") and not shutil.which(candidate):
                continue
            return candidate
    raise RuntimeError("could not find espeak-ng on PATH")


def espeak_ipa(word: str, binary: str) -> str:
    out = subprocess.check_output([binary, "-q", "--ipa=3", word], text=True)
    return out.strip()


def load_model(model_id: str, tokenizer_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def charsiu_ipa(
    words: list[str],
    *,
    lang_code: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> list[str]:
    prompts = [f"<{lang_code}>: {word}" for word in words]
    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    preds = model.generate(
        **encoded,
        num_beams=1,
        max_length=max_length,
        do_sample=False,
    )
    return tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)


def build_results(
    text: str,
    *,
    lang_code: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    espeak_bin: str,
) -> dict[str, object]:
    words = tokenize_words(text)
    if not words:
        return {
            "transcript": text,
            "lang_code": lang_code,
            "device": str(device),
            "words": [],
            "charsiu_joined": "",
            "charsiu_joined_normalized": "",
            "espeak_joined": "",
            "espeak_joined_normalized": "",
        }

    charsiu_raw = charsiu_ipa(
        words,
        lang_code=lang_code,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
    )
    results = []
    for word, c_raw in zip(words, charsiu_raw, strict=True):
        e_raw = espeak_ipa(word, espeak_bin)
        results.append(
            WordResult(
                word=word,
                charsiu_raw=c_raw,
                charsiu_normalized=normalize_ipa(c_raw),
                espeak_raw=e_raw,
                espeak_normalized=normalize_ipa(e_raw),
            )
        )

    return {
        "transcript": text,
        "lang_code": lang_code,
        "device": str(device),
        "words": [asdict(item) for item in results],
        "charsiu_joined": " | ".join(item.charsiu_raw for item in results),
        "charsiu_joined_normalized": " | ".join(item.charsiu_normalized for item in results),
        "espeak_joined": " | ".join(item.espeak_raw for item in results),
        "espeak_joined_normalized": " | ".join(item.espeak_normalized for item in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Charsiu G2P output with the current eSpeak-based pipeline."
    )
    parser.add_argument(
        "--model-id",
        default="charsiu/g2p_multilingual_byT5_tiny_16_layers",
        help="Hugging Face model id to load",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="google/byt5-small",
        help="Tokenizer id used by the Charsiu checkpoint",
    )
    parser.add_argument(
        "--lang-code",
        default="eng-us",
        help="Charsiu language code prefix without angle brackets",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table",
    )
    parser.add_argument(
        "--transcript",
        action="append",
        default=[],
        help="Transcript to evaluate; repeat for multiple items",
    )
    parser.add_argument(
        "--transcript-file",
        help="Read transcripts from a text file, one per line",
    )
    return parser.parse_args()


def load_transcripts(args: argparse.Namespace) -> list[str]:
    transcripts = [item.strip() for item in args.transcript if item.strip()]
    if args.transcript_file:
        with open(args.transcript_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    transcripts.append(line)
    if not transcripts:
        raise SystemExit("provide at least one --transcript or --transcript-file")
    return transcripts


def print_text_report(report: dict[str, object]) -> None:
    print(f"transcript: {report['transcript']}")
    print(f"lang_code: {report['lang_code']}")
    print(f"device: {report['device']}")
    print("words:")
    for word in report["words"]:
        print(f"  - {word['word']}")
        print(f"    charsiu: {word['charsiu_raw']}")
        print(f"    charsiu_norm: {word['charsiu_normalized']}")
        print(f"    espeak: {word['espeak_raw']}")
        print(f"    espeak_norm: {word['espeak_normalized']}")
    print(f"charsiu_joined: {report['charsiu_joined']}")
    print(f"charsiu_joined_normalized: {report['charsiu_joined_normalized']}")
    print(f"espeak_joined: {report['espeak_joined']}")
    print(f"espeak_joined_normalized: {report['espeak_joined_normalized']}")


def main() -> int:
    args = parse_args()
    transcripts = load_transcripts(args)
    device = pick_device(args.device)
    tokenizer, model = load_model(args.model_id, args.tokenizer_id, device)
    espeak_bin = espeak_binary()

    reports = [
        build_results(
            transcript,
            lang_code=args.lang_code,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            espeak_bin=espeak_bin,
        )
        for transcript in transcripts
    ]

    if args.json:
        json.dump(reports, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    for idx, report in enumerate(reports):
        if idx:
            print()
        print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
