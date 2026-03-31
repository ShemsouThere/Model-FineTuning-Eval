# Server GGUF Evaluation

This repo is the server-focused Spider 1.0 evaluation bundle for comparing:

- the base `Qwen/Qwen2.5-Coder-7B-Instruct` GGUF model
- your fine-tuned `Q4_K_M` GGUF model

The workflow uses `llama.cpp` / `llama-server` directly and does not require the merged Hugging Face model unless you choose to use it separately as a fallback.

## Backup

Before any edits, the evaluation-related files were copied to:

```text
backup/eval_backup_20260401_001314/
```

## Fine-tuned model path

The default fine-tuned GGUF path is:

```text
/workspace/Model-FineTuning/artifacts/spider1_qwen25_7b/gguf/model-q4_k_m.gguf
```

You can override it with `FINE_TUNED_GGUF_PATH=...` when running the wrapper.

## What changed

- Added `prepare_server_models.py` to generate `models.server.json` for base-vs-fine-tuned GGUF evaluation.
- Added `run_server_eval.sh` as the main server entry point.
- Added `evaluate_only.py` and `eval_only.sh` for scoring existing `.sql` files without rerunning inference.
- Added `--eval-timeout-sec` to `run_all.py`.
- Added timeout handling in `evaluator.py` so a single slow Spider execution query does not block forever.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('punkt')
PY
chmod +x run_server_eval.sh eval_only.sh
```

This bundle assumes these already exist on the server:

- Spider 1.0 dataset directory with `dev.json`, `tables.json`, `database/`, `evaluation.py`, and `process_sql.py`
- `llama.cpp` built with `llama-server`

## Smoke test

Run 10 Spider dev examples with `advanced_reasoning` only:

```bash
source .venv/bin/activate
SPIDER_ROOT=/workspace/datasets/spider \
LLAMA_SERVER_BIN=/workspace/llama.cpp/build/bin/llama-server \
FINE_TUNED_GGUF_PATH=/workspace/Model-FineTuning/artifacts/spider1_qwen25_7b/gguf/model-q4_k_m.gguf \
OUTPUT_DIR=$PWD/outputs/smoke \
LIMIT=10 \
LOG_LEVEL=DEBUG \
bash run_server_eval.sh
```

## Full evaluation

This is the main server command:

```bash
source .venv/bin/activate
SPIDER_ROOT=/workspace/datasets/spider \
LLAMA_SERVER_BIN=/workspace/llama.cpp/build/bin/llama-server \
FINE_TUNED_GGUF_PATH=/workspace/Model-FineTuning/artifacts/spider1_qwen25_7b/gguf/model-q4_k_m.gguf \
OUTPUT_DIR=$PWD/outputs/server_eval \
GPU_LAYERS=999 \
STRATEGIES="advanced_reasoning" \
bash run_server_eval.sh
```

## Eval-only rerun

If prediction files already exist:

```bash
source .venv/bin/activate
SPIDER_ROOT=/workspace/datasets/spider \
OUTPUT_DIR=$PWD/outputs/server_eval \
bash eval_only.sh
```

## Optional overrides

Common environment overrides for `run_server_eval.sh`:

```bash
BASE_REPO=Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
BASE_FILE=<explicit base gguf filename>
BASE_NAME=qwen25-coder-7b-base-q4_k_m
FINE_TUNED_NAME=qwen25-coder-7b-finetuned-q4_k_m
MODELS_CONFIG=$PWD/models.server.json
EVAL_TIMEOUT_SEC=7200
LOG_LEVEL=INFO
LIMIT=200
```

If the base Hugging Face GGUF repo exposes more than one `Q4_K_M` file, set `BASE_FILE` explicitly.

## Validation notes

- The command path and defaults are wired to the server GGUF workflow.
- `prepare_server_models.py` validates that the fine-tuned GGUF path exists before evaluation begins.
- The actual smoke test still needs to be run on the rented server because the `/workspace/...` fine-tuned model path only exists there.
