"""Scorers that need no LLM: result-set match, expected values, trajectory, efficiency."""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import Any

from ..dataset import Sample
from ..trace import Trace
from . import SCORERS, Score, Scorer


def _normalize(value: Any, precision: int) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), precision)
    if isinstance(value, str):
        text = value.strip()
        try:
            return round(float(text), precision)
        except ValueError:
            return text.lower()
    return value


def _rows(rows: list[tuple], precision: int) -> Counter:
    return Counter(tuple(_normalize(v, precision) for v in row) for row in rows)


@SCORERS.register("result_match")
class ResultMatch(Scorer):
    """Execution accuracy: run the reference SQL and the agent's last query, compare result sets.

    Row order and column names are ignored. With ``allow_extra_columns`` the
    agent may return more columns than the reference, as long as some subset of
    them (in any order) reproduces the reference rows exactly.
    """

    kind = "result_match"

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        if not sample.reference_sql:
            return [Score(name="result_match", value=None, explanation="no reference_sql")]
        precision = self.params.get("float_precision", 2)
        allow_extra = self.params.get("allow_extra_columns", True)
        queries = trace.sql_queries()
        if not queries:
            return [Score(name="result_match", value=0.0, explanation="agent ran no successful SQL query")]

        expected = self.ctx.db.execute(sample.reference_sql)
        expected_rows = _rows(expected.rows, precision)
        width = len(expected.columns)
        actual = self.ctx.db.execute(queries[-1])

        candidates = [tuple(range(len(actual.columns)))]
        if allow_extra and width <= len(actual.columns) <= 8:
            candidates = list(itertools.permutations(range(len(actual.columns)), width))
        for cols in candidates:
            if len(cols) != width:
                continue
            projected = _rows([tuple(row[i] for i in cols) for row in actual.rows], precision)
            if projected == expected_rows:
                return [Score(name="result_match", value=1.0, details={"query": queries[-1]})]
        return [
            Score(
                name="result_match",
                value=0.0,
                explanation=f"expected {_preview(expected.rows)} ({len(expected.rows)}x{width}), "
                f"got {_preview(actual.rows)} ({len(actual.rows)}x{len(actual.columns)})",
                details={"query": queries[-1], "expected_sample": expected.rows[:5], "actual_sample": actual.rows[:5]},
            )
        ]


def _preview(rows: list[tuple], limit: int = 3, max_chars: int = 80) -> str:
    text = ", ".join(str(tuple(r)) for r in rows[:limit]) + (", ..." if len(rows) > limit else "")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


_NUMBER = re.compile(r"-?\d[\d\s.,]*")


def _numbers(text: str) -> list[float]:
    found = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.strip().replace(" ", "").replace(" ", "")
        # Accept both "1,234.5" and "1 234,5": a comma followed by exactly 1-2 digits is a decimal mark.
        if re.fullmatch(r"-?[\d.]*,\d{1,2}", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            found.append(float(cleaned.rstrip(".")))
        except ValueError:
            pass
    return found


@SCORERS.register("answer_values")
class AnswerValues(Scorer):
    """Fraction of ``expected_values`` that appear in the final answer (numbers within a tolerance)."""

    kind = "answer_values"

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        if not sample.expected_values:
            return [Score(name="answer_values", value=None, explanation="no expected_values")]
        tolerance = self.params.get("relative_tolerance", 0.005)
        answer = trace.final_answer
        numbers = _numbers(answer)
        missing = []
        for expected in sample.expected_values:
            try:
                target = float(expected)
                hit = any(abs(n - target) <= tolerance * max(1.0, abs(target)) for n in numbers)
            except ValueError:
                hit = expected.lower() in answer.lower()
            if not hit:
                missing.append(expected)
        found = len(sample.expected_values) - len(missing)
        return [
            Score(
                name="answer_values",
                value=found / len(sample.expected_values),
                explanation=f"missing: {missing}" if missing else "",
            )
        ]


_TABLE_REF = re.compile(r'\b(?:from|join)\s+["`\[]?([a-zA-Z_][\w.]*)', re.IGNORECASE)


def touched_tables(trace: Trace, table_tools: tuple[str, ...] = ("describe_table",)) -> set[str]:
    tables = set()
    for call in trace.tool_calls:
        if call.name in table_tools and isinstance(call.input.get("table"), str):
            tables.add(call.input["table"].lower())
        if call.name == "run_sql" and not call.is_error:
            tables |= {t.split(".")[-1].lower() for t in _TABLE_REF.findall(call.input.get("sql", ""))}
    return tables


@SCORERS.register("tables")
class Tables(Scorer):
    """Trajectory: did the agent look at / query the expected tables, and only those?"""

    kind = "tables"

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        if not sample.expected_tables:
            return [Score(name="tables_recall", value=None), Score(name="tables_precision", value=None)]
        expected = {t.lower() for t in sample.expected_tables}
        touched = touched_tables(trace)
        hit = expected & touched
        return [
            Score(name="tables_recall", value=len(hit) / len(expected),
                  explanation=f"missed: {sorted(expected - touched)}" if expected - touched else ""),
            Score(name="tables_precision", value=len(hit) / len(touched) if touched else 0.0,
                  explanation=f"extra: {sorted(touched - expected)}" if touched - expected else ""),
        ]


@SCORERS.register("efficiency")
class Efficiency(Scorer):
    """Steps, tool errors, tokens, cost and latency. Lower is better for all of them."""

    kind = "efficiency"

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        u = trace.usage
        scores = [
            Score(name="tool_calls", value=len(trace.tool_calls)),
            Score(name="tool_errors", value=sum(c.is_error for c in trace.tool_calls)),
            Score(name="turns", value=trace.turns),
            Score(name="input_tokens", value=u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens),
            Score(name="output_tokens", value=u.output_tokens),
            Score(name="latency_s", value=round(trace.latency_s, 3)),
        ]
        price = self.ctx.pricing.get(trace.model_served or "")
        if price:
            cost = (
                u.input_tokens * price.get("input", 0)
                + u.output_tokens * price.get("output", 0)
                + u.cache_read_input_tokens * price.get("cache_read", 0)
                + u.cache_creation_input_tokens * price.get("cache_write", 0)
            ) / 1e6
            scores.append(Score(name="cost_usd", value=round(cost, 6)))
        return scores
