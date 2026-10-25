#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:?usage: $0 DATASET_PATH OUTPUT_DIR [EXTRA_ARGS...]}"
OUTPUT_PATH="${2:?usage: $0 DATASET_PATH OUTPUT_DIR [EXTRA_ARGS...]}"
shift 2

python -u main.py \
  --coco_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_PATH" \
  "$@"
