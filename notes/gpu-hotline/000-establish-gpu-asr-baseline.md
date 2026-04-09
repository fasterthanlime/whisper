# 000: Establish GPU ASR baseline

## Goal

Create a fixed baseline for MLX/GPU optimization work so each change can be
judged on correctness, latency, and memory instead of profile vibes.

## Baseline corpus

- Freeze a small repro corpus of WAVs that exercises:
  - hesitation at start / bad early partials
  - short English utterances
  - longer utterances with revision churn
- Include the current captured repro:
  - `/Users/amos/bearcove/bee/.artifacts/repros/captured/38662693.wav`

## Measurements

- Final transcript correctness on the fixed corpus
- Early partial traces for the same corpus
- Per-feed latency
- First partial latency
- Finalize latency
- Resident RAM
- AGX / Metal "In use system memory"
- mlx-worker CPU
- Instruments weight for:
  - `mlx::core::copy_gpu`
  - `mlx::core::concatenate_gpu`
  - `bee_qwen3_asr::generate::topk_confidence`
  - `mlx_clear_cache`

## Output

- A repeatable CLI + app profiling recipe
- A before/after table for every optimization item

## Current status

- Candidate snapshot:
  - `/Users/amos/bearcove/bee/notes/gpu-hotline/000-corpus-candidates.tsv`
- Clean baseline corpus:
  - `/Users/amos/bearcove/bee/notes/gpu-hotline/000-corpus-clean.tsv`
- Pinned baseline settings:
  - `/Users/amos/bearcove/bee/notes/gpu-hotline/000-baseline-settings.env`
- Repeatable runner:
  - `/Users/amos/bearcove/bee/scripts/run_gpu_hotline_baseline.sh`
- Correction is disabled for GPU-hotline baseline runs so raw ASR quality is measured separately from the correction pipeline.

## Validation

- Baseline numbers are captured before any optimization lands
- Same corpus and same settings are reused for all later tasks

## Depends on

- Existing frozen repro WAVs
- Current `transcribe` CLI
- Current Instruments setup
