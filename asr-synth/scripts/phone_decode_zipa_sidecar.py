#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download

from phone_decode_zipa import compute_features_timed, greedy_segments, load_tokens


REPO_ID = os.environ.get("ZIPA_REPO_ID", "anyspeech/zipa-small-crctc-300k")
MODEL_NAME = os.environ.get("ZIPA_MODEL_NAME", "model.int8.onnx")


class ZipaSidecar:
    def __init__(self) -> None:
        dl_start = time.perf_counter()
        self.model_path = hf_hub_download(REPO_ID, MODEL_NAME)
        self.token_path = Path(hf_hub_download(REPO_ID, "tokens.txt"))
        dl_end = time.perf_counter()
        build_start = time.perf_counter()
        self.tokens = load_tokens(self.token_path)
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        build_end = time.perf_counter()
        self.ready_timing_ms = {
            "download_model": round((dl_end - dl_start) * 1000, 1),
            "build_session": round((build_end - build_start) * 1000, 1),
            "total": round((build_end - dl_start) * 1000, 1),
        }

    def decode(self, audio_path: str) -> dict:
        path = Path(audio_path)
        feature_start = time.perf_counter()
        feat, duration_s, feature_timing = compute_features_timed(path)
        feature_end = time.perf_counter()
        x = feat[None, :, :]
        import numpy as np

        x_lens = np.array([feat.shape[0]], dtype=np.int64)
        infer_start = time.perf_counter()
        log_probs, log_probs_len = self.session.run(None, {"x": x, "x_lens": x_lens})
        infer_end = time.perf_counter()
        used_frames = int(log_probs_len[0])
        used = log_probs[0, :used_frames, :]
        greedy_start = time.perf_counter()
        segments = greedy_segments(used, self.tokens, duration_s)
        greedy_end = time.perf_counter()
        phones = " ".join(seg["phone"] for seg in segments)
        return {
            "file": str(path),
            "repo_id": REPO_ID,
            "model": MODEL_NAME,
            "phones": phones,
            "original_seconds": round(duration_s, 4),
            "input_frames": int(feat.shape[0]),
            "output_frames": used_frames,
            "frame_seconds": round(duration_s / used_frames, 6) if used_frames else None,
            "timing_ms": {
                **feature_timing,
                "infer": round((infer_end - infer_start) * 1000, 1),
                "greedy_segments": round((greedy_end - greedy_start) * 1000, 1),
                "total": round((infer_end - feature_start) * 1000, 1),
            },
            "segments": segments,
            "warm": True,
        }


def main() -> None:
    sidecar = ZipaSidecar()
    print(
        json.dumps(
            {
                "ready": True,
                "repo_id": REPO_ID,
                "model": MODEL_NAME,
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
            audio_path = req["audio"]
            print(json.dumps(sidecar.decode(audio_path), ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
