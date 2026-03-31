#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_ROOT="${SPIDER_ROOT:?Set SPIDER_ROOT=/path/to/spider}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR=/path/to/output_dir}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SPIDER_ROOT/evaluation.py}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

python "$ROOT_DIR/evaluate_only.py" \
  --spider-root "$SPIDER_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --eval-script "$EVAL_SCRIPT" \
  --timeout-sec "$TIMEOUT_SEC" \
  --skip-existing-metrics
