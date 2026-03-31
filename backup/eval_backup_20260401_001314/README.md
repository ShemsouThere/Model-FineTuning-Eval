# Text-to-SQL Benchmark Pipeline

This workspace contains a CPU-friendly Spider benchmarking framework for local GGUF models served by `llama.cpp`.

## Folder structure

```text
U:/Model_Evaluation
├── benchmark_runner.py
├── evaluator.py
├── model_runner.py
├── prompt_builder.py
├── run_all.py
├── spider_utils.py
└── models.example.json
```

Runtime outputs are written to:

```text
outputs/
├── gold/
├── metrics/
├── predictions/
├── server_logs/
├── summary/
└── traces/
```

## What it does

- Starts one persistent `llama-server` process per model.
- Runs every configured strategy against the full Spider dev set.
- Writes Spider-format prediction files: `SQL<TAB>db_id`.
- Calls the official Spider `evaluation.py`.
- Stores per-experiment metrics plus a final markdown/CSV comparison table.

## Strategies

- `baseline`: question only.
- `metadata_aware`: question plus tables, columns, and relationships.
- `advanced_reasoning`: two-step plan-then-SQL prompting.
- `candidate_ranked`: generate multiple SQL candidates, then rank them with syntax and execution heuristics.

## Expected layout

The zero-config path is:

```text
./spider/
  dev.json
  tables.json
  database/
  evaluation.py

./models/
  *.gguf
```

If your files live elsewhere, pass explicit flags or create `models.json` from `models.example.json`.

## Commands

Run the full benchmark:

```bash
python run_all.py
```

Run with explicit Spider root and model config:

```bash
python run_all.py \
  --spider-root /data/spider \
  --models-config /data/models.json \
  --llama-server /opt/llama.cpp/build/bin/llama-server
```

Smoke test on 10 examples:

```bash
python run_all.py --limit 10 --log-level DEBUG
```

Run a subset of strategies:

```bash
python run_all.py --strategies baseline metadata_aware advanced_reasoning
```

## Reproducibility notes

- The runner keeps a fixed seed by default.
- Predictions are resumed by default if a `.sql` file already exists.
- Each experiment also writes a JSONL trace file with raw model output and errors.
- Candidate ranking uses the same Spider SQLite databases for basic execution guidance.

## Outputs

After a full run, the main artifacts are:

- `outputs/predictions/<model>_<strategy>.sql`
- `outputs/metrics/<model>_<strategy>.metrics.json`
- `outputs/summary/summary.md`
- `outputs/summary/summary.csv`

## Notes on Spider evaluation dependencies

`run_all.py` calls the official Spider `evaluation.py` through your active Python interpreter. Make sure the evaluator's Python dependencies are installed in the same environment you use to launch the benchmark.
