#!/usr/bin/env bash
# Convert all OGG files in the corpus audio directory to 16kHz mono WAV.
# Output goes to data/phonetic-seed/audio-wav/
set -euo pipefail

AUDIO_DIR="$(dirname "$0")/../data/phonetic-seed/audio"
WAV_DIR="$(dirname "$0")/../data/phonetic-seed/audio-wav"

mkdir -p "$WAV_DIR"

count=0
for ogg in "$AUDIO_DIR"/*.ogg; do
    [ -f "$ogg" ] || continue
    base="$(basename "${ogg%.ogg}.wav")"
    wav="$WAV_DIR/$base"
    if [ ! -f "$wav" ] || [ "$ogg" -nt "$wav" ]; then
        ffmpeg -i "$ogg" -ar 16000 -ac 1 -f wav "$wav" -y -loglevel error
        count=$((count + 1))
    fi
done

echo "Converted $count file(s) to $WAV_DIR"
