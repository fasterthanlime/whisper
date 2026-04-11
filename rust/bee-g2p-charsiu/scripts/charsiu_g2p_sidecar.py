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

import json
import re
import sys
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers"
DEFAULT_TOKENIZER_ID = "google/byt5-small"
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Request:
    words: list[str]
    lang_code: str


class CharsiuSidecar:
    def __init__(self) -> None:
        self.model_id = DEFAULT_MODEL_ID
        self.tokenizer_id = DEFAULT_TOKENIZER_ID
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.cache: dict[tuple[str, str], str] = {}

    @torch.no_grad()
    def phonemize(self, words: list[str], lang_code: str) -> list[str]:
        outputs: list[str] = [""] * len(words)
        missing_indices: list[int] = []
        prompts: list[str] = []

        for index, raw_word in enumerate(words):
            word = raw_word.strip()
            if not word:
                continue
            key = (lang_code, word)
            cached = self.cache.get(key)
            if cached is not None:
                outputs[index] = cached
                continue
            if not WORD_RE.search(word):
                continue
            missing_indices.append(index)
            prompts.append(f"<{lang_code}>: {word}")

        if prompts:
            encoded = self.tokenizer(
                prompts,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate(
                **encoded,
                num_beams=1,
                max_length=64,
                do_sample=False,
            )
            decoded = self.tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
            for index, ipa in zip(missing_indices, decoded, strict=True):
                word = words[index].strip()
                outputs[index] = ipa
                self.cache[(lang_code, word)] = ipa

        return outputs


def main() -> int:
    sidecar = CharsiuSidecar()
    print(
        json.dumps(
            {
                "ready": True,
                "model": sidecar.model_id,
                "device": sidecar.device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            request = Request(words=list(payload["words"]), lang_code=str(payload["lang_code"]))
            print(
                json.dumps(
                    {
                        "word_ipas": sidecar.phonemize(request.words, request.lang_code),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
