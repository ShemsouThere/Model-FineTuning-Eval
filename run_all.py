from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from benchmark_runner import BenchmarkRunner, RunnerConfig
from evaluator import SpiderEvaluator, locate_evaluation_script, write_gold_file
from model_runner import ModelConfig
from prompt_builder import PromptBuilder
from spider_utils import SchemaRepository, load_spider_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a full local Text-to-SQL benchmark on Spider using llama.cpp GGUF models."
    )
    parser.add_argument("--spider-root", type=Path, default=Path(os.environ.get("SPIDER_ROOT", "spider")))
    parser.add_argument("--dev-json", type=Path)
    parser.add_argument("--tables-json", type=Path)
    parser.add_argument("--db-dir", type=Path)
    parser.add_argument("--eval-script", type=Path)
    parser.add_argument("--models-config", type=Path, default=Path("models.json"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER_BIN", "llama-server"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8081)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--candidate-temperature", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=PromptBuilder.default_strategies(),
        choices=PromptBuilder.default_strategies(),
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    spider_root = args.spider_root.resolve()
    dev_json = _resolve_file(args.dev_json, spider_root, ["dev.json"])
    tables_json = _resolve_file(args.tables_json, spider_root, ["tables.json"])
    db_dir = _resolve_dir(args.db_dir, spider_root, ["database"])

    models = load_models(
        models_config_path=args.models_config,
        models_dir=args.models_dir,
        threads=args.threads,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        gpu_layers=args.gpu_layers,
    )
    if not models:
        raise FileNotFoundError(
            "No GGUF models found. Put .gguf files under ./models, or create models.json "
            "from models.example.json."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_spider_examples(dev_json, limit=args.limit)
    schema_repository = SchemaRepository(tables_json, db_dir)
    prompt_builder = PromptBuilder(schema_repository)

    evaluator = None
    if not args.skip_eval:
        evaluation_script = locate_evaluation_script(spider_root, explicit_path=args.eval_script)
        gold_path = write_gold_file(examples, output_dir / "gold" / "spider_dev_gold.sql")
        evaluator = SpiderEvaluator(
            evaluation_script=evaluation_script,
            gold_path=gold_path,
            database_dir=db_dir,
            tables_json_path=tables_json,
            output_dir=output_dir / "metrics",
            python_executable=sys.executable,
            timeout_sec=args.eval_timeout_sec,
        )

    runner_config = RunnerConfig(
        llama_server_binary=args.llama_server,
        host=args.host,
        base_port=args.base_port,
        output_dir=output_dir,
        strategies=args.strategies,
        candidate_count=args.candidate_count,
        candidate_temperature=args.candidate_temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        resume=not args.no_resume,
        skip_eval=args.skip_eval,
    )

    runner = BenchmarkRunner(
        config=runner_config,
        models=models,
        examples=examples,
        schema_repository=schema_repository,
        prompt_builder=prompt_builder,
        evaluator=evaluator,
    )
    results = runner.run_all()

    logging.info("Finished %s experiments", len(results))
    logging.info("Summary table: %s", output_dir / "summary" / "summary.md")
    return 0


def load_models(
    models_config_path: Path,
    models_dir: Path,
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    gpu_layers: int,
) -> list[ModelConfig]:
    if models_config_path.exists():
        payload = json.loads(models_config_path.read_text(encoding="utf-8"))
        items = payload["models"] if isinstance(payload, dict) else payload
        return [build_model_config(item, threads, ctx_size, batch_size, ubatch_size, gpu_layers) for item in items]

    discovered = sorted(models_dir.rglob("*.gguf")) if models_dir.exists() else []
    return [
        ModelConfig(
            name=model_path.stem,
            model_path=model_path.resolve(),
            threads=threads,
            ctx_size=ctx_size,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            gpu_layers=gpu_layers,
        )
        for model_path in discovered
    ]


def build_model_config(
    item: dict[str, Any],
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    gpu_layers: int,
) -> ModelConfig:
    model_path = Path(item["path"]).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Configured model path does not exist: {model_path}")
    return ModelConfig(
        name=item.get("name", model_path.stem),
        model_path=model_path,
        threads=int(item.get("threads", threads)),
        ctx_size=int(item.get("ctx_size", ctx_size)),
        batch_size=int(item.get("batch_size", batch_size)),
        ubatch_size=int(item.get("ubatch_size", ubatch_size)),
        gpu_layers=int(item.get("gpu_layers", gpu_layers)),
        extra_args=list(item.get("extra_args", [])),
    )


def _resolve_file(explicit: Path | None, spider_root: Path, names: list[str]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    for name in names:
        direct = spider_root / name
        if direct.exists():
            return direct.resolve()

    for name in names:
        matches = list(spider_root.rglob(name))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(f"Could not find any of {names} under {spider_root}")


def _resolve_dir(explicit: Path | None, spider_root: Path, names: list[str]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    for name in names:
        direct = spider_root / name
        if direct.exists() and direct.is_dir():
            return direct.resolve()

    for path in spider_root.rglob("*"):
        if path.is_dir() and path.name in names:
            return path.resolve()

    raise FileNotFoundError(f"Could not find any of {names} under {spider_root}")


if __name__ == "__main__":
    raise SystemExit(main())
