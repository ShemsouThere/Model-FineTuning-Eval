#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPIDER_ROOT="${SPIDER_ROOT:?Set SPIDER_ROOT=/path/to/spider}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?Set LLAMA_SERVER_BIN=/path/to/llama-server}"
FINE_TUNED_GGUF_PATH="${FINE_TUNED_GGUF_PATH:-/workspace/Model-FineTuning/artifacts/spider1_qwen25_7b/gguf/model-q4_k_m.gguf}"

MODELS_CONFIG="${MODELS_CONFIG:-$ROOT_DIR/models.server.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/server_eval}"
BASE_REPO="${BASE_REPO:-Qwen/Qwen2.5-Coder-7B-Instruct-GGUF}"
BASE_FILE="${BASE_FILE:-}"
BASE_NAME="${BASE_NAME:-qwen25-coder-7b-base-q4_k_m}"
FINE_TUNED_NAME="${FINE_TUNED_NAME:-qwen25-coder-7b-finetuned-q4_k_m}"
GPU_LAYERS="${GPU_LAYERS:-999}"
STRATEGIES="${STRATEGIES:-advanced_reasoning}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-7200}"
LIMIT="${LIMIT:-}"

python "$ROOT_DIR/prepare_server_models.py" \
  --fine-tuned-gguf-path "$FINE_TUNED_GGUF_PATH" \
  --fine-tuned-name "$FINE_TUNED_NAME" \
  --base-repo "$BASE_REPO" \
  ${BASE_FILE:+--base-file "$BASE_FILE"} \
  --base-name "$BASE_NAME" \
  --models-config-out "$MODELS_CONFIG"

read -r -a STRATEGY_ARGS <<< "$STRATEGIES"

CMD=(
  python "$ROOT_DIR/run_all.py"
  --spider-root "$SPIDER_ROOT"
  --models-config "$MODELS_CONFIG"
  --llama-server "$LLAMA_SERVER_BIN"
  --output-dir "$OUTPUT_DIR"
  --strategies "${STRATEGY_ARGS[@]}"
  --gpu-layers "$GPU_LAYERS"
  --eval-timeout-sec "$EVAL_TIMEOUT_SEC"
  --log-level "$LOG_LEVEL"
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
