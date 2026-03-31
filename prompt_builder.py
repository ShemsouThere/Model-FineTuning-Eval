from __future__ import annotations

from spider_utils import SpiderExample, SchemaRepository


BASE_SYSTEM_PROMPT = """You are a research Text-to-SQL model for the Spider benchmark.
Write one SQLite SQL query that answers the question.
Return only the SQL query.
Do not include markdown, comments, reasoning, or extra text."""


class PromptBuilder:
    BASELINE = "baseline"
    METADATA_AWARE = "metadata_aware"
    ADVANCED_REASONING = "advanced_reasoning"
    CANDIDATE_RANKED = "candidate_ranked"

    @classmethod
    def default_strategies(cls) -> list[str]:
        return [
            cls.BASELINE,
            cls.METADATA_AWARE,
            cls.ADVANCED_REASONING,
            cls.CANDIDATE_RANKED,
        ]

    def __init__(self, schema_repository: SchemaRepository) -> None:
        self.schema_repository = schema_repository

    def build_single_step_messages(
        self,
        strategy: str,
        example: SpiderExample,
    ) -> list[dict[str, str]]:
        if strategy == self.BASELINE:
            return self._baseline_messages(example)
        if strategy == self.METADATA_AWARE:
            return self._metadata_messages(example)
        if strategy == self.CANDIDATE_RANKED:
            return self._candidate_messages(example)
        raise ValueError(f"Unsupported single-step strategy: {strategy}")

    def build_reasoning_plan_messages(self, example: SpiderExample) -> list[dict[str, str]]:
        schema = self.schema_repository.detailed_schema(example.db_id, example.question)
        user_prompt = f"""Database schema:
{schema}

Question:
{example.question}

Think about the SQL construction in a structured way and respond with JSON only.
Use this schema:
{{
  "relevant_tables": ["..."],
  "join_path": ["..."],
  "filters": ["..."],
  "grouping": ["..."],
  "ordering": ["..."],
  "sql_sketch": "short natural language sketch"
}}

Do not write the final SQL yet."""
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def build_reasoning_sql_messages(
        self,
        example: SpiderExample,
        reasoning_plan: str,
    ) -> list[dict[str, str]]:
        schema = self.schema_repository.detailed_schema(example.db_id, example.question)
        user_prompt = f"""Database schema:
{schema}

Question:
{example.question}

Reasoning plan:
{reasoning_plan}

Now write the final SQLite SQL query.
Return only SQL."""
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _baseline_messages(self, example: SpiderExample) -> list[dict[str, str]]:
        user_prompt = f"""Question:
{example.question}

Return the SQLite SQL query only."""
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _metadata_messages(self, example: SpiderExample) -> list[dict[str, str]]:
        schema = self.schema_repository.brief_schema(example.db_id, example.question)
        user_prompt = f"""Database schema:
{schema}

Question:
{example.question}

Use exact table and column names from the schema.
Return only the SQLite SQL query."""
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _candidate_messages(self, example: SpiderExample) -> list[dict[str, str]]:
        schema = self.schema_repository.detailed_schema(example.db_id, example.question)
        user_prompt = f"""Database schema:
{schema}

Question:
{example.question}

Generate the best SQLite SQL query you can.
Prefer a valid executable query over a complicated query.
Return only SQL."""
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
