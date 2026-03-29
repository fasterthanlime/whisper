#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from allophant import predictions
from allophant.dataset_processing import Batch
from allophant.estimator import Estimator


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, float]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    audio = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != target_sr:
        import torchaudio.functional as F

        audio = F.resample(audio, sample_rate, target_sr)
    duration_s = float(audio.shape[1]) / float(target_sr)
    return audio, duration_s


def decode_segments(
    logits: torch.Tensor,
    frame_count: int,
    duration_s: float,
    hypothesis,
    inventory_indexer,
) -> list[dict]:
    log_probs = torch.log_softmax(logits[:frame_count, 0, :], dim=-1)
    frame_s = duration_s / frame_count if frame_count else 0.0
    tokens = hypothesis.tokens.tolist()
    timesteps = hypothesis.timesteps.tolist()
    phones = [
        str(phone)
        for phone in inventory_indexer.feature_values(
            "phoneme", hypothesis.tokens.cpu() - 1
        )
    ]
    segments = []

    for i, (phone, token_id, start_frame) in enumerate(zip(phones, tokens, timesteps)):
        end_frame = timesteps[i + 1] if i + 1 < len(timesteps) else frame_count
        if end_frame <= start_frame:
            continue
        class_id = int(token_id) - 1  # Decoder token IDs include the blank offset.
        scores = log_probs[start_frame:end_frame, class_id]
        segments.append(
            {
                "phone": phone,
                "token_id": int(token_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(start_frame * frame_s, 4),
                "end_sec": round(end_frame * frame_s, 4),
                "avg_logprob": round(float(scores.mean().item()), 5),
            }
        )

    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode audio to timed IPA phone segments with Allophant."
    )
    parser.add_argument("audio", nargs="+", help="Audio file(s), ideally 16kHz mono wav.")
    parser.add_argument("--checkpoint", default="kgnlp/allophant")
    parser.add_argument("--inventory-lang", default="en")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    total_start = time.perf_counter()
    build_start = time.perf_counter()
    model, attribute_indexer = Estimator.restore(args.checkpoint, device="cpu")
    inventory = attribute_indexer.phoneme_inventory(args.inventory_lang)
    inventory_indexer = attribute_indexer.attributes.subset(inventory)
    decoder = predictions.feature_decoders(
        inventory_indexer, feature_names=["phoneme"]
    )["phoneme"]
    composition = attribute_indexer.composition_feature_matrix(inventory).to("cpu")
    build_end = time.perf_counter()

    for audio_path in args.audio:
        feature_start = time.perf_counter()
        audio, duration_s = load_audio(Path(audio_path), model.sample_rate)
        feature_end = time.perf_counter()

        batch = Batch(
            audio,
            torch.tensor([audio.shape[1]], dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )

        infer_start = time.perf_counter()
        outputs = model.predict(batch.to("cpu"), composition)
        hypotheses = decoder(outputs.outputs["phoneme"].transpose(1, 0), outputs.lengths)
        infer_end = time.perf_counter()

        hypothesis = hypotheses[0][0]
        frame_count = int(outputs.lengths[0].item())
        segments = decode_segments(
            outputs.outputs["phoneme"],
            frame_count,
            duration_s,
            hypothesis,
            inventory_indexer,
        )
        phones = " ".join(segment["phone"] for segment in segments)

        payload = {
            "file": str(audio_path),
            "checkpoint": args.checkpoint,
            "inventory_lang": args.inventory_lang,
            "phones": phones,
            "original_seconds": round(duration_s, 4),
            "output_frames": frame_count,
            "frame_seconds": round(duration_s / frame_count, 6) if frame_count else None,
            "timing_ms": {
                "build_model": round((build_end - build_start) * 1000, 1),
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
