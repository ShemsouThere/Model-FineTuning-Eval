from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from evaluator import SpiderEvaluator, write_gold_file
from spider_utils import load_spider_examples


STRATEGIES = [
    "candidate_ranked",
    "advanced_reasoning",
    "metadata_aware",
    "baseline",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing Spider-format prediction files without rerunning inference."
    )
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--skip-existing-metrics", action="store_true")
    return parser


def split_stem(stem: str) -> tuple[str, str]:
    for strategy in STRATEGIES:
        suffix = "_" + strategy
        if stem.endswith(suffix):
            return stem[: -len(suffix)], strategy
    return stem, "unknown"


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    args = build_parser().parse_args()

    spider_root = args.spider_root.resolve()
    output_dir = args.output_dir.resolve()
    pred_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    examples = load_spider_examples(spider_root / "dev.json")
    gold_path = write_gold_file(examples, output_dir / "gold" / "spider_dev_gold.sql")
    evaluator = SpiderEvaluator(
        evaluation_script=(args.eval_script or spider_root / "evaluation.py").resolve(),
        gold_path=gold_path,
        database_dir=(spider_root / "database").resolve(),
        tables_json_path=(spider_root / "tables.json").resolve(),
        output_dir=metrics_dir,
        python_executable=sys.executable,
        timeout_sec=args.timeout_sec,
    )

    rows: list[list[str]] = []
    for prediction_path in sorted(pred_dir.glob("*.sql")):
        line_count = count_nonempty_lines(prediction_path)
        if line_count != len(examples):
            print(
                f"SKIP incomplete prediction: {prediction_path.name} "
                f"({line_count}/{len(examples)})"
            )
            continue

        metrics_path = metrics_dir / f"{prediction_path.stem}.metrics.json"
        if args.skip_existing_metrics and metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            exact_match = float(payload["exact_match"])
            execution_accuracy = float(payload["execution_accuracy"])
        else:
            result = evaluator.evaluate(prediction_path)
            exact_match = result.exact_match
            execution_accuracy = result.execution_accuracy

        model, strategy = split_stem(prediction_path.stem)
        rows.append(
            [
                model,
                strategy,
                f"{exact_match:.4f}",
                f"{execution_accuracy:.4f}",
                str(prediction_path),
            ]
        )
        print(
            f"EVAL {prediction_path.name}: "
            f"exact={exact_match:.4f} exec={execution_accuracy:.4f}"
        )

    csv_path = summary_dir / "summary_eval_only.csv"
    md_path = summary_dir / "summary_eval_only.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "Strategy", "Exact Match", "Execution", "Prediction File"])
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Model | Strategy | Exact Match | Execution |\n")
        handle.write("| --- | --- | ---: | ---: |\n")
        for model, strategy, exact_match, execution_accuracy, _ in rows:
            handle.write(
                f"| {model} | {strategy} | {exact_match} | {execution_accuracy} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
