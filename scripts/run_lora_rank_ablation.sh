#!/usr/bin/env bash
set -euo pipefail

RANKS="${RANKS:-16 32 64}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
COMET_MODEL="${COMET_MODEL:-Unbabel/wmt22-comet-da}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

for rank in $RANKS; do
  config="configs/ablation_lora_r${rank}_8k.yaml"
  output_dir="outputs/ablation/lora-r${rank}-8k"
  checkpoint="${output_dir}/checkpoint-8000"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  train_log="${LOG_DIR}/train-lora-r${rank}-8k-${timestamp}.log"
  eval_log="${LOG_DIR}/eval-lora-r${rank}-8k-${timestamp}.log"

  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi

  if [[ -d "$output_dir" && ! -d "$checkpoint" && "${ALLOW_PARTIAL_OUTPUT_DIR:-0}" != "1" ]]; then
    echo "Found partial output directory: $output_dir" >&2
    echo "Move it aside before rerunning, or set ALLOW_PARTIAL_OUTPUT_DIR=1 if you know this is safe." >&2
    exit 1
  fi

  echo "=== LoRA rank ${rank}: training with ${config} ==="
  if [[ -d "$checkpoint" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    echo "Found ${checkpoint}; skipping training. Set FORCE_TRAIN=1 to rerun."
  else
    uv run python train.py --config "$config" 2>&1 | tee "$train_log"
  fi

  if [[ ! -d "$checkpoint" ]]; then
    checkpoint="${output_dir}/final"
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "No checkpoint or final adapter found for rank ${rank} under ${output_dir}" >&2
    exit 1
  fi

  echo "=== LoRA rank ${rank}: evaluating ${checkpoint} ==="
  uv run python eval.py \
    --config "$config" \
    --split validation \
    --adapter "$checkpoint" \
    --limit "$EVAL_LIMIT" \
    --comet-model "$COMET_MODEL" \
    --output-jsonl "${output_dir}/eval/validation_predictions_${EVAL_LIMIT}.jsonl" \
    --metrics-path "${output_dir}/eval/validation_metrics_${EVAL_LIMIT}.json" \
    2>&1 | tee "$eval_log"
done
