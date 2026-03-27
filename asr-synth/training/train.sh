#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a small model for ASR correction using MLX-LM LoRA
#
# Uses Qwen2.5-0.5B (smallest Qwen model) to minimize memory usage.
# The model learns: given two noisy ASR transcripts + vocab → corrected text

export AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1

MODEL="Qwen/Qwen2.5-0.5B"
DATA_DIR="training/data"
ADAPTER_DIR="training/adapters"
ITERS="${1:-200}"

echo "=== ASR Correction Model Training ==="
echo "Model:    $MODEL"
echo "Data:     $DATA_DIR"
echo "Adapters: $ADAPTER_DIR"
echo "Iters:    $ITERS"
echo ""

uvx --refresh --from 'mlx-lm==0.31.1' mlx_lm.lora \
  --model "$MODEL" \
  --data "$DATA_DIR" \
  --train \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers 4 \
  --adapter-path "$ADAPTER_DIR" \
  --mask-prompt
