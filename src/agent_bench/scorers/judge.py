"""LLM-as-a-judge against a rubric of binary criteria.

Design choices:
* Binary pass/fail criteria, each with its own reasoning written *before* the
  verdict. Far less noisy than a 1-10 grade, and the per-criterion pass rates
  tell you *what* got better, not only that something did.
* The judge sees the reference answer: it compares instead of guessing.
* Determinism comes from the cache (each verdict is stored under the judge's
  fingerprint) plus optional majority voting, not from temperature: current
  Claude models do not accept sampling parameters.
* Trust it only after calibration against human labels (see calibration.py).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel

from ..dataset import Sample
from ..registry import stable_hash
from ..trace import Trace
from . import SCORERS, Score, Scorer

FALLBACK_BETA = "server-side-fallback-2026-07-01"

JUDGE_SYSTEM = """You grade answers produced by a data agent that answers business questions by querying a database.
You receive the question, a reference answer written by a domain expert, and the agent's answer.
Grade each criterion independently. For each one, write your reasoning first, then decide passed=true or passed=false.
Be strict: a criterion passes only if it is clearly met. Do not reward length, confidence or formatting.
Numbers that differ only by rounding or formatting count as equal."""


class Criterion(BaseModel):
    id: str
    question: str
    applies_to: Literal["all", "answerable", "unanswerable"] = "all"


class Rubric(BaseModel):
    name: str
    version: int = 1
    criteria: list[Criterion]


class CriterionVerdict(BaseModel):
    criterion_id: str
    reasoning: str
    passed: bool


class JudgeOutput(BaseModel):
    verdicts: list[CriterionVerdict]


class JudgeClient(Protocol):
    """Anything that turns a grading prompt into verdicts. Implement it to use another provider."""

    def judge(self, system: str, prompt: str) -> tuple[JudgeOutput, dict[str, Any]]: ...


class ClaudeJudgeClient:
    def __init__(self, model: str, effort: str = "medium", fallbacks: bool = True, client: Any = None):
        import anthropic

        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = client or anthropic.Anthropic()

    def judge(self, system: str, prompt: str) -> tuple[JudgeOutput, dict[str, Any]]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=JudgeOutput,
            output_config={"effort": self.effort},
        )
        if self.fallbacks:
            response = self.client.beta.messages.parse(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.parse(**kwargs)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
        # A fallback means another model graded this run: keep track of it.
        return response.parsed_output, {"judge_model_served": response.model}


@SCORERS.register("rubric_judge")
class RubricJudge(Scorer):
    kind = "rubric_judge"
    version = 1

    def __init__(self, ctx, rubric: str, model: str = "claude-opus-5-5", effort: str = "medium",
                 votes: int = 1, show_sql: bool = True, fallbacks: bool = True, **params: Any):
        super().__init__(ctx, rubric=rubric, model=model, effort=effort, votes=votes, show_sql=show_sql, **params)
        path = Path(rubric)
        path = path if path.is_absolute() else ctx.base_dir / path
        self._rubric_text = path.read_text()
        self.rubric = Rubric.model_validate(yaml.safe_load(self._rubric_text))
        self.votes = votes
        self.show_sql = show_sql
        self.client: JudgeClient = ctx.clients.get("judge") or ClaudeJudgeClient(model, effort, fallbacks)

    def fingerprint(self) -> str:
        return stable_hash({
            "kind": self.kind,
            "version": self.version,
            "params": self.params,
            "rubric": hashlib.sha256(self._rubric_text.encode()).hexdigest(),
            "system": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
        })

    def applicable(self, sample: Sample) -> list[Criterion]:
        scope = "answerable" if sample.answerable else "unanswerable"
        return [c for c in self.rubric.criteria if c.applies_to in ("all", scope)]

    def build_prompt(self, sample: Sample, trace: Trace, criteria: list[Criterion]) -> str:
        parts = [f"<question>\n{sample.question}\n</question>"]
        if not sample.answerable:
            parts.append("<note>This question cannot be answered from the database. "
                         "A good agent says so instead of inventing an answer.</note>")
        if sample.reference_answer:
            parts.append(f"<reference_answer>\n{sample.reference_answer}\n</reference_answer>")
        if sample.expected_values:
            parts.append(f"<expected_values>{', '.join(sample.expected_values)}</expected_values>")
        if self.show_sql and trace.sql_queries():
            queries = "\n---\n".join(trace.sql_queries())
            parts.append(f"<agent_sql_queries>\n{queries}\n</agent_sql_queries>")
        parts.append(f"<agent_answer>\n{trace.final_answer or '(empty answer)'}\n</agent_answer>")
        listed = "\n".join(f"- {c.id}: {c.question}" for c in criteria)
        parts.append(f"<criteria>\n{listed}\n</criteria>\nReturn exactly one verdict per criterion id.")
        return "\n\n".join(parts)

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        criteria = self.applicable(sample)
        if not criteria:
            return [Score(name="judge", value=None, explanation="no applicable criteria")]
        prompt = self.build_prompt(sample, trace, criteria)
        ballots: list[dict[str, CriterionVerdict]] = []
        meta: dict[str, Any] = {}
        for _ in range(self.votes):
            output, meta = self.client.judge(JUDGE_SYSTEM, prompt)
            by_id = {v.criterion_id: v for v in output.verdicts}
            missing = [c.id for c in criteria if c.id not in by_id]
            if missing:
                raise ValueError(f"judge skipped criteria {missing}")
            ballots.append(by_id)

        scores = []
        for c in criteria:
            votes = Counter(b[c.id].passed for b in ballots)
            passed = votes[True] > votes[False]
            reasoning = next(b[c.id].reasoning for b in ballots if b[c.id].passed == passed)
            scores.append(Score(
                name=f"judge.{c.id}",
                value=float(passed),
                explanation=reasoning,
                details={"votes": dict(votes), **meta},
            ))
        overall = sum(s.value for s in scores) / len(scores)
        scores.append(Score(name="judge", value=overall, details=meta))
        return scores
