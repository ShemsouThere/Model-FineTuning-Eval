from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator import EvaluationResult, SpiderEvaluator
from model_runner import LlamaCppServer, ModelConfig, SamplingConfig
from prompt_builder import PromptBuilder
from spider_utils import SpiderExample, SchemaRepository, extract_sql, safe_file_stem


LOGGER = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    llama_server_binary: str
    host: str
    base_port: int
    output_dir: Path
    strategies: list[str]
    candidate_count: int
    candidate_temperature: float
    max_tokens: int
    seed: int
    resume: bool = True
    skip_eval: bool = False


@dataclass
class ExperimentResult:
    model: str
    strategy: str
    exact_match: float | None
    execution: float | None
    prediction_path: Path


class BenchmarkRunner:
    def __init__(
        self,
        config: RunnerConfig,
        models: list[ModelConfig],
        examples: list[SpiderExample],
        schema_repository: SchemaRepository,
        prompt_builder: PromptBuilder,
        evaluator: SpiderEvaluator | None,
    ) -> None:
        self.config = config
        self.models = models
        self.examples = examples
        self.schema_repository = schema_repository
        self.prompt_builder = prompt_builder
        self.evaluator = evaluator

        self.predictions_dir = self.config.output_dir / "predictions"
        self.traces_dir = self.config.output_dir / "traces"
        self.metrics_dir = self.config.output_dir / "metrics"
        self.summary_dir = self.config.output_dir / "summary"
        for directory in (
            self.predictions_dir,
            self.traces_dir,
            self.metrics_dir,
            self.summary_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for model_index, model in enumerate(self.models):
            port = self.config.base_port + model_index
            server_log_dir = self.config.output_dir / "server_logs"
            with LlamaCppServer(
                binary_path=self.config.llama_server_binary,
                model_config=model,
                host=self.config.host,
                port=port,
                log_dir=server_log_dir,
            ) as server:
                for strategy in self.config.strategies:
                    results.append(self._run_experiment(model, strategy, server))

        self._write_summary(results)
        return results

    def _run_experiment(
        self,
        model: ModelConfig,
        strategy: str,
        server: LlamaCppServer,
    ) -> ExperimentResult:
        file_stem = f"{model.safe_name}_{safe_file_stem(strategy)}"
        prediction_path = self.predictions_dir / f"{file_stem}.sql"
        trace_path = self.traces_dir / f"{file_stem}.jsonl"

        completed = self._prediction_count(prediction_path) if self.config.resume else 0
        if completed > len(self.examples):
            raise RuntimeError(
                f"{prediction_path} contains more rows than the loaded Spider dev set."
            )

        write_mode = "a" if completed else "w"
        LOGGER.info(
            "Running %s / %s (%s completed, %s total)",
            model.name,
            strategy,
            completed,
            len(self.examples),
        )

        with prediction_path.open(write_mode, encoding="utf-8") as prediction_handle, trace_path.open(
            write_mode,
            encoding="utf-8",
        ) as trace_handle:
            for example_offset, example in enumerate(self.examples[completed:], start=completed):
                record: dict[str, Any] = {
                    "index": example.index,
                    "db_id": example.db_id,
                    "question": example.question,
                    "strategy": strategy,
                    "model": model.name,
                }
                try:
                    sql, details = self._generate_sql(server, strategy, example)
                    record.update(details)
                except Exception as exc:  # pragma: no cover
                    LOGGER.exception(
                        "Generation failed for model=%s strategy=%s example=%s",
                        model.name,
                        strategy,
                        example.index,
                    )
                    sql = "SELECT 1"
                    record["error"] = repr(exc)
                    record.setdefault("raw_text", "")

                sql = re.sub(r"\s+", " ", sql).strip() or "SELECT 1"
                prediction_handle.write(f"{sql}\t{example.db_id}\n")
                prediction_handle.flush()

                record["final_sql"] = sql
                trace_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                trace_handle.flush()

                if (
                    example_offset == completed
                    or (example_offset + 1) % 25 == 0
                    or example_offset + 1 == len(self.examples)
                ):
                    LOGGER.info(
                        "Progress %s / %s / %s: %s/%s",
                        model.name,
                        strategy,
                        example.db_id,
                        example_offset + 1,
                        len(self.examples),
                    )

        metrics = self._evaluate(prediction_path)
        return ExperimentResult(
            model=model.name,
            strategy=strategy,
            exact_match=metrics.exact_match if metrics else None,
            execution=metrics.execution_accuracy if metrics else None,
            prediction_path=prediction_path,
        )

    def _generate_sql(
        self,
        server: LlamaCppServer,
        strategy: str,
        example: SpiderExample,
    ) -> tuple[str, dict[str, Any]]:
        if strategy == PromptBuilder.BASELINE:
            messages = self.prompt_builder.build_single_step_messages(strategy, example)
            result = server.generate(
                messages,
                SamplingConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed + example.index,
                ),
            )
            return extract_sql(result.text), {
                "latency_sec": result.latency_sec,
                "raw_text": result.text,
            }

        if strategy == PromptBuilder.METADATA_AWARE:
            messages = self.prompt_builder.build_single_step_messages(strategy, example)
            result = server.generate(
                messages,
                SamplingConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed + example.index,
                ),
            )
            return extract_sql(result.text), {
                "latency_sec": result.latency_sec,
                "raw_text": result.text,
            }

        if strategy == PromptBuilder.ADVANCED_REASONING:
            plan_messages = self.prompt_builder.build_reasoning_plan_messages(example)
            plan_result = server.generate(
                plan_messages,
                SamplingConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=min(256, self.config.max_tokens),
                    seed=self.config.seed + example.index,
                ),
            )
            sql_messages = self.prompt_builder.build_reasoning_sql_messages(
                example,
                plan_result.text,
            )
            sql_result = server.generate(
                sql_messages,
                SamplingConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed + example.index,
                ),
            )
            return extract_sql(sql_result.text), {
                "latency_sec": plan_result.latency_sec + sql_result.latency_sec,
                "reasoning_plan": plan_result.text,
                "raw_text": sql_result.text,
            }

        if strategy == PromptBuilder.CANDIDATE_RANKED:
            messages = self.prompt_builder.build_single_step_messages(strategy, example)
            candidates: list[dict[str, Any]] = []
            for candidate_index in range(self.config.candidate_count):
                result = server.generate(
                    messages,
                    SamplingConfig(
                        temperature=self.config.candidate_temperature,
                        top_p=0.95,
                        max_tokens=self.config.max_tokens,
                        seed=self.config.seed + example.index * 100 + candidate_index,
                    ),
                )
                sql = extract_sql(result.text)
                score, execution_ok, error = self._score_candidate(sql, example.db_id)
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "sql": sql,
                        "score": score,
                        "execution_ok": execution_ok,
                        "execution_error": error,
                        "latency_sec": result.latency_sec,
                        "raw_text": result.text,
                    }
                )

            best = max(candidates, key=lambda item: item["score"])
            total_latency = sum(candidate["latency_sec"] for candidate in candidates)
            return str(best["sql"]), {
                "latency_sec": total_latency,
                "raw_text": best["raw_text"],
                "candidates": candidates,
            }

        raise ValueError(f"Unsupported strategy: {strategy}")

    def _score_candidate(self, sql: str, db_id: str) -> tuple[float, bool, str | None]:
        score = 0.0
        text = sql.strip()
        if not text:
            return -1000.0, False, "empty SQL"

        if text.upper().startswith(("SELECT", "WITH")):
            score += 20.0
        else:
            score -= 50.0

        if text.count("(") != text.count(")"):
            score -= 10.0

        table_references = [
            match.group(1).strip("`\"[]")
            for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w$]*)", text, flags=re.IGNORECASE)
        ]
        known_tables = self.schema_repository.known_table_names(db_id)
        unknown_tables = [table for table in table_references if table.lower() not in known_tables]
        score -= 10.0 * len(unknown_tables)

        execution_ok, execution_error = self._execute_sql(db_id, text)
        if execution_ok:
            score += 100.0
        else:
            score -= 25.0

        score -= len(text) / 1000.0
        return score, execution_ok, execution_error

    def _execute_sql(self, db_id: str, sql: str) -> tuple[bool, str | None]:
        db_path = self.schema_repository.database_path(db_id)
        if not db_path.exists():
            return False, f"database not found: {db_path}"

        step_budget = 250_000
        progress_calls = 0

        def progress_handler() -> int:
            nonlocal progress_calls
            progress_calls += 1
            if progress_calls > step_budget:
                return 1
            return 0

        try:
            with sqlite3.connect(db_path) as connection:
                connection.set_progress_handler(progress_handler, 1000)
                cursor = connection.execute(sql)
                cursor.fetchmany(1)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _evaluate(self, prediction_path: Path) -> EvaluationResult | None:
        if self.config.skip_eval or self.evaluator is None:
            return None
        return self.evaluator.evaluate(prediction_path)

    def _write_summary(self, results: list[ExperimentResult]) -> None:
        csv_path = self.summary_dir / "summary.csv"
        md_path = self.summary_dir / "summary.md"

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Model", "Strategy", "Exact Match", "Execution", "Prediction File"])
            for result in results:
                writer.writerow(
                    [
                        result.model,
                        result.strategy,
                        "" if result.exact_match is None else f"{result.exact_match:.4f}",
                        "" if result.execution is None else f"{result.execution:.4f}",
                        str(result.prediction_path),
                    ]
                )

        lines = [
            "| Model | Strategy | Exact Match | Execution |",
            "| --- | --- | ---: | ---: |",
        ]
        for result in results:
            exact = "-" if result.exact_match is None else f"{result.exact_match:.4f}"
            execution = "-" if result.execution is None else f"{result.execution:.4f}"
            lines.append(f"| {result.model} | {result.strategy} | {exact} | {execution} |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _prediction_count(prediction_path: Path) -> int:
        if not prediction_path.exists():
            return 0
        with prediction_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
