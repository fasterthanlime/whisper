#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio


MODEL_ID = os.environ.get("COHERE_MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")
LANGUAGE = os.environ.get("COHERE_LANGUAGE", "en")


def resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
class CohereAsrSidecar:
    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN")
        self.device = resolve_device(os.environ.get("COHERE_DEVICE", "auto"))
        build_start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, token=token
        )
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            token=token,
        ).to(self.device)
        self.model.eval()
        build_end = time.perf_counter()
        self.ready_timing_ms = {
            "build_model": round((build_end - build_start) * 1000, 1),
        }

    def transcribe(self, audio_path: str) -> dict:
        path = Path(audio_path)
        load_start = time.perf_counter()
        audio = load_audio(str(path), sampling_rate=16000)
        load_end = time.perf_counter()
        proc_start = time.perf_counter()
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=LANGUAGE,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        proc_end = time.perf_counter()
        infer_start = time.perf_counter()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
        )
        infer_end = time.perf_counter()
        text = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            language=LANGUAGE,
        ).strip()
        return {
            "file": str(path),
            "model_id": MODEL_ID,
            "device": self.device,
            "language": LANGUAGE,
            "text": text,
            "timing_ms": {
                "load_audio": round((load_end - load_start) * 1000, 1),
                "processor": round((proc_end - proc_start) * 1000, 1),
                "infer": round((infer_end - infer_start) * 1000, 1),
                "total": round((infer_end - load_start) * 1000, 1),
            },
            "warm": True,
        }


def main() -> None:
    sidecar = CohereAsrSidecar()
    print(
        json.dumps(
            {
                "ready": True,
                "model_id": MODEL_ID,
                "device": sidecar.device,
                "language": LANGUAGE,
                "timing_ms": sidecar.ready_timing_ms,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            print(json.dumps(sidecar.transcribe(req["audio"]), ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
