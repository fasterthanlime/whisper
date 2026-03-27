#!/usr/bin/env python3
import argparse
import io
import json
import os
import subprocess
import re
import sqlite3
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def normalize_sentence(text: str) -> str:
    text = text.lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_'\s]+", " ", text)
    return " ".join(text.split())


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_\s]+", " ", text)
    return " ".join(text.split())


def transcript_has_variant(transcript: str, variants: list[str]) -> bool:
    norm_transcript = f" {normalize_phrase(transcript)} "
    return any(f" {normalize_phrase(variant)} " in norm_transcript for variant in variants)


def load_alt_spellings(con: sqlite3.Connection) -> dict[str, list[str]]:
    alts: dict[str, list[str]] = {}
    for term, alt in con.execute("select term, alt_spelling from vocab_alt_spellings"):
        alts.setdefault(term.lower(), []).append(alt)
    return alts


def load_recordings(con: sqlite3.Connection) -> list[dict]:
    rows = []
    for row in con.execute(
        """
        select id, term, sentence, take_no, wav_path
        from authored_sentence_recordings
        order by id
        """
    ):
        rows.append(
            {
                "recording_id": row[0],
                "term": row[1],
                "sentence": row[2],
                "take_no": row[3],
                "wav_path": row[4],
            }
        )
    return rows


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(str(audio_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)
    except Exception:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        audio, sr = sf.read(io.BytesIO(proc.stdout), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def call_dual_asr(api_base: str, audio: np.ndarray, sr: int) -> tuple[str, str]:
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/asr/dual",
        data=wav_bytes(audio, sr),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data.get("qwen", "") or "", data.get("parakeet", "") or ""


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="corpus.db")
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--api-base", default="http://127.0.0.1:3456")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    db_path = Path(args.db).resolve()
    audio_root = Path(args.audio_root).resolve()
    con = sqlite3.connect(str(db_path))
    alt_spellings = load_alt_spellings(con)
    recordings = load_recordings(con)
    if args.limit:
        recordings = recordings[: args.limit]

    print(f"Loaded {len(recordings)} recordings from {db_path}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=token,
    ).to(device)
    model.eval()

    skipped_audio = []
    usable_recordings = []
    for rec in recordings:
        try:
            audio, sr = load_audio(audio_root / rec["wav_path"])
        except Exception as exc:
            skipped_audio.append(
                {
                    "recording_id": rec["recording_id"],
                    "term": rec["term"],
                    "wav_path": rec["wav_path"],
                    "error": str(exc),
                }
            )
            continue
        rec["audio"] = audio
        rec["sr"] = sr
        usable_recordings.append(rec)
    recordings = usable_recordings
    print(f"Usable recordings: {len(recordings)}; skipped audio: {len(skipped_audio)}")

    print("Running Cohere transcription...")
    for batch in batched(recordings, args.batch_size):
        arrays = [item["audio"] for item in batch]
        srs = [item["sr"] for item in batch]
        texts = model.transcribe(
            processor=processor,
            audio_arrays=arrays,
            sample_rates=srs,
            language="en",
            batch_size=len(batch),
        )
        for item, text in zip(batch, texts):
            item["cohere"] = text or ""
        print(f"  Cohere {batch[-1]['recording_id']}/{recordings[-1]['recording_id']}")

    print("Running dashboard dual ASR for baseline comparison...")
    for idx, rec in enumerate(recordings, 1):
        qwen, parakeet = call_dual_asr(args.api_base, rec["audio"], rec["sr"])
        rec["qwen"] = qwen
        rec["parakeet"] = parakeet
        print(f"  Baseline {idx}/{len(recordings)}")

    systems = ["qwen", "parakeet", "cohere"]
    summary = {}
    for system in systems:
        wers = []
        term_hits = 0
        misses = []
        for rec in recordings:
            expected = rec["sentence"]
            hyp = rec.get(system, "")
            wers.append(wer(normalize_sentence(expected), normalize_sentence(hyp)))
            variants = [rec["term"], *alt_spellings.get(rec["term"].lower(), [])]
            hit = transcript_has_variant(hyp, variants)
            if hit:
                term_hits += 1
            else:
                misses.append(
                    {
                        "recording_id": rec["recording_id"],
                        "term": rec["term"],
                        "expected": expected,
                        "hypothesis": hyp,
                    }
                )
        summary[system] = {
            "count": len(recordings),
            "mean_wer": statistics.fmean(wers) if wers else 0.0,
            "median_wer": statistics.median(wers) if wers else 0.0,
            "term_hit_rate": term_hits / len(recordings) if recordings else 0.0,
            "term_hits": term_hits,
            "misses": misses,
        }

    improvements = []
    regressions = []
    for rec in recordings:
        variants = [rec["term"], *alt_spellings.get(rec["term"].lower(), [])]
        q_hit = transcript_has_variant(rec["qwen"], variants)
        c_hit = transcript_has_variant(rec["cohere"], variants)
        if c_hit and not q_hit:
            improvements.append((rec["term"], rec["qwen"], rec["cohere"], rec["sentence"]))
        elif q_hit and not c_hit:
            regressions.append((rec["term"], rec["qwen"], rec["cohere"], rec["sentence"]))

    print("\n=== SUMMARY ===")
    for system in systems:
        item = summary[system]
        print(
            f"{system:8s} term_hit={item['term_hits']}/{item['count']} ({item['term_hit_rate']*100:.1f}%)"
            f"  mean_wer={item['mean_wer']:.3f} median_wer={item['median_wer']:.3f}"
        )

    print(f"\nCohere beats Qwen on target term: {len(improvements)}")
    for term, qwen, cohere, sentence in improvements[:15]:
        print(f"  + {term}: qwen={qwen!r} | cohere={cohere!r} | ref={sentence!r}")

    print(f"\nCohere loses to Qwen on target term: {len(regressions)}")
    for term, qwen, cohere, sentence in regressions[:15]:
        print(f"  - {term}: qwen={qwen!r} | cohere={cohere!r} | ref={sentence!r}")

    miss_counts = Counter(m["term"] for m in summary["cohere"]["misses"])
    print("\nTop Cohere misses:")
    for term, count in miss_counts.most_common(15):
        print(f"  {term}: {count}")

    if args.out:
        out_path = Path(args.out)
        payload = {
            "model": MODEL_ID,
            "summary": summary,
            "skipped_audio": skipped_audio,
            "improvements_vs_qwen": improvements,
            "regressions_vs_qwen": regressions,
            "recordings": [
                {
                    k: v
                    for k, v in rec.items()
                    if k not in {"audio", "sr"}
                }
                for rec in recordings
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
