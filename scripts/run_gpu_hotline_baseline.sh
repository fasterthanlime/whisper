#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:-$repo_root/notes/gpu-hotline/000-corpus-candidates.tsv}"
settings="${2:-$repo_root/notes/gpu-hotline/000-baseline-settings.env}"
run_name="${3:-$(date +%Y%m%d-%H%M%S)}"
out_dir="${BEE_BASELINE_OUT_DIR:-$repo_root/.artifacts/gpu-hotline/runs/$run_name}"

if [[ ! -f "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

if [[ ! -f "$settings" ]]; then
  echo "missing settings: $settings" >&2
  exit 1
fi

if ! command -v direnv >/dev/null 2>&1; then
  echo "direnv is required" >&2
  exit 1
fi

mkdir -p "$out_dir/transcripts"

pushd "$repo_root" >/dev/null
eval "$(direnv export bash)"
set -a
source "$settings"
set +a

cargo build -q -p bee-transcribe --bin transcribe

cp "$manifest" "$out_dir/corpus.tsv"
cp "$settings" "$out_dir/settings.env"
git rev-parse HEAD > "$out_dir/git-rev.txt"

summary="$out_dir/SUMMARY.tsv"
printf 'id\tgoal\tcategory\tmax_feed_ms\tfinish_ms\ttotal_ms\tfirst_partial\tlast_partial\tfinal_text\n' > "$summary"

while IFS=$'\t' read -r id wav_path source category goal status; do
  if [[ "$id" == "id" ]]; then
    continue
  fi

  log="$out_dir/transcripts/$id.txt"
  target/debug/transcribe "$wav_path" > "$log" 2>&1

  max_feed_ms="$(
    awk '
      /^  chunk [0-9]+: [0-9]+ms/ {
        ms = $3
        sub(/ms$/, "", ms)
        if (ms + 0 > max) {
          max = ms + 0
        }
      }
      END { print max + 0 }
    ' "$log"
  )"
  read -r finish_ms total_ms <<<"$(
    awk '
      /^--- Final \([0-9]+ms, total [0-9]+ms\) ---$/ {
        finish = $3
        total = $5
        gsub(/[^0-9]/, "", finish)
        gsub(/[^0-9]/, "", total)
        print finish, total
      }
    ' "$log"
  )"
  first_partial="$(
    awk -F'\\| ' '
      /^  chunk [0-9]+: .* \| rev=/ {
        print $NF
        exit
      }
    ' "$log"
  )"
  last_partial="$(
    awk -F'\\| ' '
      /^  chunk [0-9]+: .* \| rev=/ {
        value = $NF
      }
      END {
        print value
      }
    ' "$log"
  )"
  final_text="$(
    sed -n 's/^  text: //p' "$log" | tail -n 1
  )"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" \
    "$goal" \
    "$category" \
    "${max_feed_ms:-0}" \
    "${finish_ms:-0}" \
    "${total_ms:-0}" \
    "$first_partial" \
    "$last_partial" \
    "$final_text" >> "$summary"
done < "$manifest"

printf 'wrote baseline run to %s\n' "$out_dir"
popd >/dev/null
