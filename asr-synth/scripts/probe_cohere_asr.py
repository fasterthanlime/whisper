#!/usr/bin/env python3
import argparse
import io
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def load_audio(audio_path: Path):
    try:
        audio, sr = sf.read(str(audio_path), always_2d=False)
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
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def pick_recording(db_path: Path, audio_root: Path, recording_id: int | None):
    con = sqlite3.connect(str(db_path))
    if recording_id is None:
        row = con.execute(
            """
            select id, term, sentence, wav_path
            from authored_sentence_recordings
            order by id
            limit 1
            """
        ).fetchone()
    else:
        row = con.execute(
            """
            select id, term, sentence, wav_path
            from authored_sentence_recordings
            where id = ?
            """,
            (recording_id,),
        ).fetchone()
    if row is None:
        raise SystemExit("No matching authored recording found")
    return {
        "recording_id": row[0],
        "term": row[1],
        "sentence": row[2],
        "wav_path": row[3],
        "audio_path": audio_root / row[3],
    }


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="corpus.db")
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--recording-id", type=int)
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    audio_root = Path(args.audio_root).resolve()
    rec = pick_recording(db_path, audio_root, args.recording_id)
    audio, sr = load_audio(rec["audio_path"])
    device = resolve_device(args.device)

    print(f"recording_id={rec['recording_id']}")
    print(f"term={rec['term']}")
    print(f"expected={rec['sentence']}")
    print(f"audio_path={rec['audio_path']}")
    print(f"sample_rate={sr}")
    print(f"num_samples={len(audio)}")
    print(f"device={device}")
    print(f"language={args.language}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=True,
    ).to(device)
    model.eval()

    def run_probe(label: str, **kwargs):
        print(f"\n[{label}]")
        try:
            result = model.transcribe(
                processor=processor,
                language=args.language,
                **kwargs,
            )
            print(result[0] if result else "")
        except Exception as exc:
            print(f"ERROR: {exc}")

    run_probe("file-ogg", audio_files=[str(rec["audio_path"])])

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr, format="WAV", subtype="PCM_16")
        run_probe("file-wav", audio_files=[tmp.name])

    run_probe("array", audio_arrays=[audio], sample_rates=[sr])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
