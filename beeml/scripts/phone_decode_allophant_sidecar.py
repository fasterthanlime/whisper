#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

from allophant import predictions
from allophant.dataset_processing import Batch
from allophant.estimator import Estimator
import torch

from phone_decode_allophant import decode_segments, load_audio


CHECKPOINT = "kgnlp/allophant"
INVENTORY_LANG = "en"


class AllophantSidecar:
    def __init__(self) -> None:
        build_start = time.perf_counter()
        self.model, self.attribute_indexer = Estimator.restore(CHECKPOINT, device="cpu")
        self.inventory = self.attribute_indexer.phoneme_inventory(INVENTORY_LANG)
        self.inventory_indexer = self.attribute_indexer.attributes.subset(self.inventory)
        self.decoder = predictions.feature_decoders(
            self.inventory_indexer, feature_names=["phoneme"]
        )["phoneme"]
        self.composition = self.attribute_indexer.composition_feature_matrix(
            self.inventory
        ).to("cpu")
        build_end = time.perf_counter()
        self.ready_timing_ms = {
            "build_model": round((build_end - build_start) * 1000, 1),
        }

    def decode(self, audio_path: str) -> dict:
        path = Path(audio_path)
        feature_start = time.perf_counter()
        audio, duration_s = load_audio(path, self.model.sample_rate)
        feature_end = time.perf_counter()
        batch = Batch(
            audio,
            torch.tensor([audio.shape[1]], dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
        infer_start = time.perf_counter()
        outputs = self.model.predict(batch.to("cpu"), self.composition)
        hypotheses = self.decoder(
            outputs.outputs["phoneme"].transpose(1, 0), outputs.lengths
        )
        infer_end = time.perf_counter()
        hypothesis = hypotheses[0][0]
        frame_count = int(outputs.lengths[0].item())
        segments = decode_segments(
            outputs.outputs["phoneme"],
            frame_count,
            duration_s,
            hypothesis,
            self.inventory_indexer,
        )
        phones = " ".join(segment["phone"] for segment in segments)
        return {
            "file": str(path),
            "checkpoint": CHECKPOINT,
            "inventory_lang": INVENTORY_LANG,
            "phones": phones,
            "original_seconds": round(duration_s, 4),
            "output_frames": frame_count,
            "frame_seconds": round(duration_s / frame_count, 6) if frame_count else None,
            "timing_ms": {
                "feature_extract": round((feature_end - feature_start) * 1000, 1),
                "infer": round((infer_end - infer_start) * 1000, 1),
                "total": round((infer_end - feature_start) * 1000, 1),
            },
            "segments": segments,
            "warm": True,
        }


def main() -> None:
    sidecar = AllophantSidecar()
    print(
        json.dumps(
            {
                "ready": True,
                "checkpoint": CHECKPOINT,
                "inventory_lang": INVENTORY_LANG,
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
            print(json.dumps(sidecar.decode(req["audio"]), ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
