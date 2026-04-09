#!/usr/bin/env python3

import argparse
import json
import sys
import wave
from pathlib import Path


def resolve_site_packages() -> Path:
    home = Path.home()
    lib_root = home / ".local/share/uv/tools/kokoro-tts/lib"
    for child in sorted(lib_root.glob("python*/site-packages")):
        if (child / "kokoro_onnx").exists():
            return child
    raise RuntimeError(f"could not find kokoro_onnx site-packages under {lib_root}")


def write_wav(path: Path, audio, sample_rate: int) -> None:
    pcm = audio.clip(-1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--phonemes")
    parser.add_argument("--out")
    args = parser.parse_args()

    sys.path.insert(0, str(resolve_site_packages()))
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(args.model, args.voices)
    if args.server:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                request = json.loads(raw_line)
                phonemes = request["phonemes"]
                out_path = Path(request["out"])
                voice = request.get("voice") or args.voice
                speed = float(request.get("speed", args.speed))
                lang = request.get("lang", args.lang)
                audio, sample_rate = kokoro.create(
                    text="",
                    voice=voice,
                    speed=speed,
                    lang=lang,
                    phonemes=phonemes,
                )
                write_wav(out_path, audio, sample_rate)
                sys.stdout.write(
                    json.dumps(
                        {
                            "ok": True,
                            "resolved_voice": voice,
                            "sample_rate_hz": sample_rate,
                            "wav_path": str(out_path),
                        }
                    )
                    + "\n"
                )
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
                sys.stdout.flush()
        return 0

    if not args.phonemes or not args.out:
        parser.error("--phonemes and --out are required unless --server is set")

    audio, sample_rate = kokoro.create(
        text="",
        voice=args.voice,
        speed=args.speed,
        lang=args.lang,
        phonemes=args.phonemes,
    )
    write_wav(Path(args.out), audio, sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
