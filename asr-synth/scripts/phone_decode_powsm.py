#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
from espnet2.torch_utils.device_funcs import to_device
from huggingface_hub import snapshot_download


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern!r} under {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one match for {pattern!r} under {root}, got {len(matches)}"
        )
    return matches[0]


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_audio_16k_20s(path: Path, sample_rate: int, max_seconds: float) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(audio.T[:1])
    if source_rate != sample_rate:
        mono = F.resample(mono, source_rate, sample_rate)

    target_len = int(sample_rate * max_seconds)
    if mono.shape[1] > target_len:
        mono = mono[:, :target_len]
    elif mono.shape[1] < target_len:
        pad = torch.zeros((1, target_len - mono.shape[1]), dtype=mono.dtype)
        mono = torch.cat([mono, pad], dim=1)

    return mono.squeeze(0)


def prepare_audio(
    path: Path, sample_rate: int, max_seconds: float
) -> tuple[torch.Tensor, float]:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(audio.T[:1])
    if source_rate != sample_rate:
        mono = F.resample(mono, source_rate, sample_rate)

    mono = mono.squeeze(0)
    original_seconds = mono.numel() / sample_rate
    target_len = int(sample_rate * max_seconds)
    if mono.numel() > target_len:
        mono = mono[:target_len]
    elif mono.numel() < target_len:
        mono = torch.cat([mono, torch.zeros(target_len - mono.numel(), dtype=mono.dtype)])
    return mono, original_seconds


def build_decoder(
    snapshot_dir: Path,
    device: str,
    lang_sym: str,
    task_sym: str,
) -> Speech2TextGreedySearch:
    config = find_one(snapshot_dir, "exp/**/config.yaml")
    model = find_one(snapshot_dir, "exp/**/*.pth")
    bpe_model = find_one(snapshot_dir, "data/token_list/**/bpe.model")
    with pushd(snapshot_dir):
        return Speech2TextGreedySearch(
            s2t_train_config=str(config),
            s2t_model_file=str(model),
            bpemodel=str(bpe_model),
            device=device,
            lang_sym=lang_sym,
            task_sym=task_sym,
        )


def is_phone_token(token: str) -> bool:
    return token.startswith("/") and token.endswith("/") and len(token) > 2


def normalize_phone_token(token: str) -> str:
    if is_phone_token(token):
        return token[1:-1]
    return token


def decode_with_trace(
    decoder: Speech2TextGreedySearch,
    audio: torch.Tensor,
    original_seconds: float,
    sample_rate: int,
    buffer_seconds: float,
    lang_sym: str,
    task_sym: str,
) -> dict:
    lang_id = decoder.converter.token2id[lang_sym]
    task_id = decoder.converter.token2id[task_sym]
    text_prev = torch.tensor([[decoder.s2t_model.na]], dtype=torch.long)
    text_prev_lengths = torch.tensor([1], dtype=torch.long)
    prefix = torch.tensor([[lang_id, task_id]], dtype=torch.long)
    prefix_lengths = torch.tensor([2], dtype=torch.long)
    speech = audio.unsqueeze(0).to(getattr(torch, decoder.dtype))
    speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    batch = {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "text_prev": text_prev,
        "text_prev_lengths": text_prev_lengths,
        "prefix": prefix,
        "prefix_lengths": prefix_lengths,
    }
    batch = to_device(batch, device=decoder.device)

    started = time.perf_counter()
    enc, enc_olens = decoder.s2t_model.encode(**batch)
    if isinstance(enc, tuple):
        enc = enc[0]
    enc_ms = (time.perf_counter() - started) * 1000.0

    olen = int(enc_olens[0])
    started = time.perf_counter()
    log_probs = decoder.s2t_model.ctc.log_softmax(enc)[0, :olen].cpu()
    argmax_ids = torch.argmax(log_probs, dim=-1)
    logprob_ms = (time.perf_counter() - started) * 1000.0

    frame_seconds = buffer_seconds / max(olen, 1)
    blank_id = decoder.converter.token2id.get("<blank>", 0)

    segments = []
    current = None
    for frame_idx, token_id in enumerate(argmax_ids.tolist()):
        token = decoder.converter.ids2tokens([token_id])[0]
        if token_id == blank_id:
            current = None
            continue
        if not is_phone_token(token):
            current = None
            continue

        start_sec = frame_idx * frame_seconds
        end_sec = (frame_idx + 1) * frame_seconds
        frame_logprob = float(log_probs[frame_idx, token_id].item())

        if current and current["token_id"] == token_id:
            current["end_frame"] = frame_idx + 1
            current["end_sec"] = end_sec
            current["logprob_sum"] += frame_logprob
            current["frame_count"] += 1
        else:
            current = {
                "token_id": token_id,
                "token": token,
                "phone": normalize_phone_token(token),
                "start_frame": frame_idx,
                "end_frame": frame_idx + 1,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "logprob_sum": frame_logprob,
                "frame_count": 1,
            }
            segments.append(current)

    for seg in segments:
        seg["avg_logprob"] = seg["logprob_sum"] / seg["frame_count"]
        seg["within_original_audio"] = seg["start_sec"] < original_seconds
        del seg["logprob_sum"]

    phone_string = "".join(seg["token"] for seg in segments if seg["within_original_audio"])

    return {
        "phones": phone_string,
        "buffer_seconds": buffer_seconds,
        "original_seconds": original_seconds,
        "sample_rate": sample_rate,
        "encoder_frames": olen,
        "frame_seconds": frame_seconds,
        "timing_ms": {
            "encode": enc_ms,
            "log_softmax": logprob_ms,
            "total": enc_ms + logprob_ms,
        },
        "segments": segments,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode audio to phone strings with espnet/powsm_ctc."
    )
    p.add_argument("audio", nargs="+", help="Audio files to decode")
    p.add_argument("--repo-id", default="espnet/powsm_ctc")
    p.add_argument("--device", default="cpu")
    p.add_argument("--lang-sym", default="<unk>")
    p.add_argument("--task-sym", default="<pr>")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--json", action="store_true", help="Emit JSON lines")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overall_started = time.perf_counter()
    download_started = time.perf_counter()
    snapshot_dir = Path(
        snapshot_download(repo_id=args.repo_id, allow_patterns=["exp/*", "exp/**/*", "data/*", "data/**/*"])
    )
    download_ms = (time.perf_counter() - download_started) * 1000.0
    build_started = time.perf_counter()
    decoder = build_decoder(
        snapshot_dir=snapshot_dir,
        device=args.device,
        lang_sym=args.lang_sym,
        task_sym=args.task_sym,
    )
    build_ms = (time.perf_counter() - build_started) * 1000.0

    paths = [Path(p).expanduser().resolve() for p in args.audio]
    for path in paths:
        prepare_started = time.perf_counter()
        audio, original_seconds = prepare_audio(
            path, sample_rate=args.sample_rate, max_seconds=args.max_seconds
        )
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
        trace = decode_with_trace(
            decoder=decoder,
            audio=audio,
            original_seconds=original_seconds,
            sample_rate=args.sample_rate,
            buffer_seconds=args.max_seconds,
            lang_sym=args.lang_sym,
            task_sym=args.task_sym,
        )
        trace["file"] = str(path)
        trace["timing_ms"]["prepare_audio"] = prepare_ms
        trace["timing_ms"]["download_snapshot"] = download_ms
        trace["timing_ms"]["build_decoder"] = build_ms
        trace["timing_ms"]["overall_elapsed"] = (time.perf_counter() - overall_started) * 1000.0

        if args.json:
            print(json.dumps(trace, ensure_ascii=False))
        else:
            print(f"FILE={path.name}")
            print(trace["phones"])
            print(
                "timing_ms",
                json.dumps(trace["timing_ms"], ensure_ascii=False, sort_keys=True),
            )
            print(
                "trace",
                json.dumps(
                    {
                        "encoder_frames": trace["encoder_frames"],
                        "frame_seconds": trace["frame_seconds"],
                        "original_seconds": trace["original_seconds"],
                        "segments": trace["segments"][:20],
                    },
                    ensure_ascii=False,
                ),
            )


if __name__ == "__main__":
    main()
