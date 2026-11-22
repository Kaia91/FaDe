#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a Hugging Face model id or local model path.}"
TRAIN_DATA="${TRAIN_DATA:-dataset/conflictbank/conflict_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fade}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-$OUTPUT_DIR/teacher_cache}"

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ "${OVERWRITE_TEACHER_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite_teacher_cache)
fi

"$PYTHON_BIN" train.py \
  --stage cache \
  --model_name "$MODEL_NAME_OR_PATH" \
  --data_path "$TRAIN_DATA" \
  --teacher_cache_dir "$TEACHER_CACHE_DIR" \
  "${EXTRA_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR"
