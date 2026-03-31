from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from spider_utils import SpiderExample, normalize_sql_whitespace


LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    exact_match: float
    execution_accuracy: float
    stdout: str


def write_gold_file(examples: list[SpiderExample], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f"{normalize_sql_whitespace(example.query)}\t{example.db_id}\n")
    return output_path


def locate_evaluation_script(spider_root: Path, explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            spider_root / "evaluation.py",
            spider_root / "spider" / "evaluation.py",
            spider_root / "eval" / "evaluation.py",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Spider evaluation.py. "
        "Pass --eval-script explicitly or place evaluation.py under the Spider root."
    )


class SpiderEvaluator:
    def __init__(
        self,
        evaluation_script: Path,
        gold_path: Path,
        database_dir: Path,
        tables_json_path: Path,
        output_dir: Path,
        python_executable: str | None = None,
        timeout_sec: int | None = 7200,
    ) -> None:
        self.evaluation_script = evaluation_script
        self.gold_path = gold_path
        self.database_dir = database_dir
        self.tables_json_path = tables_json_path
        self.output_dir = output_dir
        self.python_executable = python_executable or sys.executable
        self.timeout_sec = timeout_sec

    def evaluate(self, prediction_path: Path) -> EvaluationResult:
        command = [
            self.python_executable,
            str(self.evaluation_script),
            "--gold",
            str(self.gold_path),
            "--pred",
            str(prediction_path),
            "--db",
            str(self.database_dir),
            "--table",
            str(self.tables_json_path),
            "--etype",
            "all",
        ]
        LOGGER.info("Evaluating %s", prediction_path.name)
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.evaluation_script.parent),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Spider evaluation timed out for {prediction_path.name} after "
                f"{self.timeout_sec} seconds."
            ) from exc
        stdout = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Spider evaluation failed for {prediction_path.name}:\n{stdout}"
            )

        result = EvaluationResult(
            exact_match=self._extract_metric(stdout, "exact match"),
            execution_accuracy=self._extract_metric(stdout, "execution"),
            stdout=stdout,
        )

        metrics_path = self.output_dir / f"{prediction_path.stem}.metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(asdict(result), indent=2),
            encoding="utf-8",
        )

        raw_stdout_path = self.output_dir / f"{prediction_path.stem}.evaluation.txt"
        raw_stdout_path.write_text(stdout, encoding="utf-8")
        return result

    @staticmethod
    def _extract_metric(stdout: str, label: str) -> float:
        for line in stdout.splitlines():
            compact = re.sub(r"\s+", " ", line.strip().lower())
            if compact.startswith(label):
                values = re.findall(r"\d+\.\d+|\d+", line)
                if values:
                    return float(values[-1])

        fallback = re.search(
            rf"{re.escape(label)}[^0-9]*(\d+\.\d+|\d+)",
            stdout,
            flags=re.IGNORECASE,
        )
        if fallback:
            return float(fallback.group(1))
        raise ValueError(f"Could not parse '{label}' from Spider evaluation output")
