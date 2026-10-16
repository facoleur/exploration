"""Stage 2 building blocks: scorers turn (sample, trace) into named scores.

A scorer returns several ``Score`` rows (e.g. recall *and* precision), or a
``None`` value when it does not apply to a sample (no reference SQL, ...).
Its ``fingerprint`` keys the score cache: change the rubric or the judge
model and only that scorer is recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..dataset import Sample
from ..db import Database
from ..registry import Registry, stable_hash
from ..trace import Trace


class Score(BaseModel):
    name: str
    value: float | None  # None = not applicable to this sample
    explanation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ScorerContext:
    db: Database
    base_dir: Path
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    clients: dict[str, Any] = field(default_factory=dict)  # injected LLM clients (tests, other providers)


class Scorer:
    kind: str = ""
    # Bump when the scoring logic changes, to invalidate cached scores.
    version: int = 1

    def __init__(self, ctx: ScorerContext, **params: Any):
        self.ctx = ctx
        self.params = params

    def fingerprint(self) -> str:
        return stable_hash({"kind": self.kind, "version": self.version, "params": self.params})

    def score(self, sample: Sample, trace: Trace) -> list[Score]:
        raise NotImplementedError


SCORERS: Registry[type[Scorer]] = Registry("scorer")


def build_scorer(kind: str, params: dict[str, Any], ctx: ScorerContext) -> Scorer:
    return SCORERS.get(kind)(ctx, **params)


from . import deterministic, judge  # noqa: E402,F401
