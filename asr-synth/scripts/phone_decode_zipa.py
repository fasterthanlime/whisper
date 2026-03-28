#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


def load_tokens(path: Path) -> dict[int, str]:
    tokens = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        token, idx = line.rsplit(" ", 1)
        tokens[int(idx)] = token
    return tokens


def compute_features(path: Path) -> tuple[np.ndarray, float]:
    y, _sr = librosa.load(path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        power=1.0,
        center=True,
    )
    feat = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
    duration_s = len(y) / 16000.0
    return feat, duration_s


def greedy_segments(
    log_probs: np.ndarray, tokens: dict[int, str], duration_s: float
) -> list[dict]:
    frame_ids = np.argmax(log_probs, axis=-1).tolist()
    total_frames = len(frame_ids)
    if total_frames == 0:
        return []

    blank_id = 0
    frame_s = duration_s / total_frames
    segments = []
    start = 0
    current = frame_ids[0]

    def maybe_add(seg_id: int, seg_start: int, seg_end: int) -> None:
        if seg_id == blank_id or seg_end <= seg_start:
            return
        token = tokens.get(seg_id, f"<{seg_id}>")
        scores = log_probs[seg_start:seg_end, seg_id]
        segments.append(
            {
                "phone": token,
                "token_id": seg_id,
                "start_frame": seg_start,
                "end_frame": seg_end,
                "start_sec": round(seg_start * frame_s, 4),
                "end_sec": round(seg_end * frame_s, 4),
                "avg_logprob": round(float(scores.mean()), 5),
            }
        )

    for i, token_id in enumerate(frame_ids[1:], start=1):
        if token_id != current:
            maybe_add(current, start, i)
            start = i
            current = token_id
    maybe_add(current, start, total_frames)
    return segments


def main() -> None:
    p = argparse.ArgumentParser(description="Decode audio to timed phone segments with ZIPA-small ONNX.")
    p.add_argument("audio", nargs="+", help="Audio file(s), ideally 16kHz mono wav.")
    p.add_argument("--repo-id", default="anyspeech/zipa-small-crctc-300k")
    p.add_argument("--model", default="model.int8.onnx")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    total_start = time.perf_counter()
    dl_start = time.perf_counter()
    model_path = hf_hub_download(args.repo_id, args.model)
    token_path = Path(hf_hub_download(args.repo_id, "tokens.txt"))
    dl_end = time.perf_counter()
    tokens = load_tokens(token_path)

    build_start = time.perf_counter()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    build_end = time.perf_counter()

    for audio_path in args.audio:
        feature_start = time.perf_counter()
        feat, duration_s = compute_features(Path(audio_path))
        feature_end = time.perf_counter()

        x = feat[None, :, :]
        x_lens = np.array([feat.shape[0]], dtype=np.int64)
        infer_start = time.perf_counter()
        log_probs, log_probs_len = session.run(None, {"x": x, "x_lens": x_lens})
        infer_end = time.perf_counter()

        used_frames = int(log_probs_len[0])
        used = log_probs[0, :used_frames, :]
        segments = greedy_segments(used, tokens, duration_s)
        phones = " ".join(seg["phone"] for seg in segments)

        payload = {
            "file": str(audio_path),
            "repo_id": args.repo_id,
            "model": args.model,
            "phones": phones,
            "original_seconds": round(duration_s, 4),
            "input_frames": int(feat.shape[0]),
            "output_frames": used_frames,
            "frame_seconds": round(duration_s / used_frames, 6) if used_frames else None,
            "timing_ms": {
                "download_model": round((dl_end - dl_start) * 1000, 1),
                "build_session": round((build_end - build_start) * 1000, 1),
                "feature_extract": round((feature_end - feature_start) * 1000, 1),
                "infer": round((infer_end - infer_start) * 1000, 1),
                "total": round((infer_end - total_start) * 1000, 1),
            },
            "segments": segments,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"FILE={Path(audio_path).name}")
            print(phones)


if __name__ == "__main__":
    main()
