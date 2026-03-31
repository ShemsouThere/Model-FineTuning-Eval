from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpiderExample:
    index: int
    question: str
    query: str
    db_id: str
    raw: dict[str, Any]


@dataclass
class SpiderSchema:
    db_id: str
    table_names_original: list[str]
    table_names_normalized: list[str]
    column_names_original: list[tuple[int, str]]
    column_names_normalized: list[tuple[int, str]]
    primary_keys: list[int]
    foreign_keys: list[tuple[int, int]]


def load_spider_examples(dev_json_path: Path, limit: int | None = None) -> list[SpiderExample]:
    payload = json.loads(dev_json_path.read_text(encoding="utf-8"))
    examples: list[SpiderExample] = []
    for index, item in enumerate(payload):
        examples.append(
            SpiderExample(
                index=index,
                question=str(item["question"]).strip(),
                query=normalize_sql_whitespace(str(item["query"])),
                db_id=str(item["db_id"]).strip(),
                raw=item,
            )
        )
    return examples[:limit] if limit is not None else examples


class SchemaRepository:
    def __init__(self, tables_json_path: Path, database_dir: Path) -> None:
        self.tables_json_path = tables_json_path
        self.database_dir = database_dir
        self.schemas = self._load_tables()
        self._brief_cache: dict[str, str] = {}
        self._detailed_cache: dict[str, str] = {}

    def brief_schema(self, db_id: str, question: str | None = None) -> str:
        if db_id not in self._brief_cache:
            self._brief_cache[db_id] = self._build_brief_schema(db_id)
        schema = self._brief_cache[db_id]
        if question:
            return self._prioritize_tables(schema, db_id, question)
        return schema

    def detailed_schema(self, db_id: str, question: str | None = None) -> str:
        if db_id not in self._detailed_cache:
            self._detailed_cache[db_id] = self._build_detailed_schema(db_id)
        return self._detailed_cache[db_id]

    def database_path(self, db_id: str) -> Path:
        return self.database_dir / db_id / f"{db_id}.sqlite"

    def known_table_names(self, db_id: str) -> set[str]:
        schema = self.schemas[db_id]
        return {table.lower() for table in schema.table_names_original}

    def known_column_names(self, db_id: str) -> set[str]:
        schema = self.schemas[db_id]
        return {
            column.lower()
            for table_index, column in schema.column_names_original
            if table_index >= 0
        }

    def _load_tables(self) -> dict[str, SpiderSchema]:
        payload = json.loads(self.tables_json_path.read_text(encoding="utf-8"))
        schemas: dict[str, SpiderSchema] = {}
        for item in payload:
            schemas[item["db_id"]] = SpiderSchema(
                db_id=item["db_id"],
                table_names_original=list(item["table_names_original"]),
                table_names_normalized=list(item["table_names"]),
                column_names_original=[tuple(column) for column in item["column_names_original"]],
                column_names_normalized=[tuple(column) for column in item["column_names"]],
                primary_keys=list(item["primary_keys"]),
                foreign_keys=[tuple(pair) for pair in item["foreign_keys"]],
            )
        return schemas

    def _build_brief_schema(self, db_id: str) -> str:
        schema = self.schemas[db_id]
        tables: list[str] = []
        for table_index, table_name in enumerate(schema.table_names_original):
            columns = [
                column_name
                for column_table_index, column_name in schema.column_names_original
                if column_table_index == table_index
            ]
            tables.append(f"{table_name}({', '.join(columns)})")
        relationships = self._relationship_lines(schema)
        parts = ["Tables:", *tables]
        if relationships:
            parts.extend(["Relationships:", *relationships])
        return "\n".join(parts)

    def _build_detailed_schema(self, db_id: str) -> str:
        schema = self.schemas[db_id]
        db_path = self.database_path(db_id)
        statements: list[str] = []
        if db_path.exists():
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT name, sql
                    FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
            for _, statement in rows:
                if statement:
                    normalized = re.sub(r"\s+", " ", statement).strip()
                    statements.append(f"{normalized};")

        if not statements:
            statements.append(self._build_brief_schema(db_id))

        relationships = self._relationship_lines(schema)
        primary_keys = self._primary_key_lines(schema)

        parts = ["Schema DDL:", *statements]
        if primary_keys:
            parts.extend(["Primary keys:", *primary_keys])
        if relationships:
            parts.extend(["Foreign keys:", *relationships])
        return "\n".join(parts)

    def _relationship_lines(self, schema: SpiderSchema) -> list[str]:
        lines: list[str] = []
        for from_index, to_index in schema.foreign_keys:
            from_table_idx, from_column = schema.column_names_original[from_index]
            to_table_idx, to_column = schema.column_names_original[to_index]
            from_table = schema.table_names_original[from_table_idx]
            to_table = schema.table_names_original[to_table_idx]
            lines.append(f"{from_table}.{from_column} -> {to_table}.{to_column}")
        return lines

    def _primary_key_lines(self, schema: SpiderSchema) -> list[str]:
        lines: list[str] = []
        for column_index in schema.primary_keys:
            table_index, column_name = schema.column_names_original[column_index]
            table_name = schema.table_names_original[table_index]
            lines.append(f"{table_name}.{column_name}")
        return lines

    def _prioritize_tables(self, schema_text: str, db_id: str, question: str) -> str:
        question_terms = set(re.findall(r"[A-Za-z_]+", question.lower()))
        schema = self.schemas[db_id]
        priority_tables = set()
        for table_name in schema.table_names_original:
            tokens = set(re.findall(r"[A-Za-z_]+", table_name.lower()))
            if tokens & question_terms:
                priority_tables.add(table_name)

        if not priority_tables:
            return schema_text

        lines = schema_text.splitlines()
        prioritized: list[str] = []
        remaining: list[str] = []
        for line in lines:
            if any(line.startswith(f"{table_name}(") or table_name in line for table_name in priority_tables):
                prioritized.append(line)
            else:
                remaining.append(line)
        return "\n".join(prioritized + remaining)


def normalize_sql_whitespace(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def extract_sql(text: str) -> str:
    if not text:
        return "SELECT 1"

    candidate = text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()

    candidate = re.sub(r"</?sql>", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"^assistant\s*:\s*", "", candidate, flags=re.IGNORECASE)

    for marker in ("final sql:", "sqlquery:", "sql:", "query:"):
        lowered = candidate.lower()
        marker_index = lowered.find(marker)
        if marker_index != -1:
            candidate = candidate[marker_index + len(marker) :].strip()
            break

    lines: list[str] = []
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith(("analysis:", "reasoning:", "plan:", "explanation:")):
            continue
        lines.append(line)
    candidate = " ".join(lines)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    select_match = re.search(r"\b(SELECT|WITH)\b.*", candidate, flags=re.IGNORECASE)
    if select_match:
        candidate = select_match.group(0)

    if ";" in candidate:
        candidate = candidate.split(";", 1)[0]

    candidate = candidate.strip().strip("`")
    return candidate or "SELECT 1"


def safe_file_stem(name: str) -> str:
    cleaned = [
        character.lower() if character.isalnum() else "_"
        for character in name.strip()
    ]
    collapsed = "".join(cleaned).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "artifact"
