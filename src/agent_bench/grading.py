"""Stage 2: score every trace with every scorer, caching per (scorer fingerprint, run key).

Changing the judge rubric re-grades only with the judge; the agent is never re-run.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .runner import Bench
from .scorers import Score, Scorer, ScorerContext, build_scorer
from .trace import Trace

log = logging.getLogger(__name__)


class ScoreRow(BaseModel):
    run_key: str
    variant: str
    sample_id: str
    epoch: int
    category: str
    split: str
    scorer: str
    metric: str
    value: float | None
    explanation: str = ""


def build_scorers(bench: Bench, clients: dict[str, Any] | None = None) -> list[Scorer]:
    ctx = ScorerContext(
        db=bench.db,
        base_dir=bench.experiment.base_dir,
        pricing=bench.experiment.pricing,
        clients=clients or {},
    )
    return [build_scorer(spec.kind, spec.params, ctx) for spec in bench.experiment.scorers]


class ScoreCache:
    def __init__(self, run_dir: Path, scorer: Scorer):
        self.dir = run_dir / "scores" / f"{scorer.kind}-{scorer.fingerprint()}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, run_key: str) -> list[Score] | None:
        path = self.dir / f"{run_key}.json"
        if not path.exists():
            return None
        return [Score.model_validate(s) for s in json.loads(path.read_text())]

    def put(self, run_key: str, scores: list[Score]) -> None:
        (self.dir / f"{run_key}.json").write_text(json.dumps([s.model_dump() for s in scores], indent=2))


def grade(bench: Bench, traces: list[Trace], scorers: list[Scorer]) -> list[ScoreRow]:
    """Errored runs get no scores: they are infrastructure failures, reported separately."""
    jobs = [(scorer, ScoreCache(bench.experiment.run_dir, scorer), t) for scorer in scorers for t in traces if t.ok]
    with ThreadPoolExecutor(max_workers=bench.experiment.concurrency) as pool:
        results = list(pool.map(lambda job: _grade_one(bench, *job), jobs))

    rows = []
    for (scorer, _, trace), scores in zip(jobs, results):
        sample = bench.samples_by_id[trace.sample_id]
        for s in scores:
            rows.append(ScoreRow(
                run_key=trace.run_key, variant=trace.variant, sample_id=trace.sample_id, epoch=trace.epoch,
                category=sample.category, split=sample.split, scorer=scorer.kind, metric=s.name,
                value=s.value, explanation=s.explanation,
            ))
    return rows


def _grade_one(bench: Bench, scorer: Scorer, cache: ScoreCache, trace: Trace) -> list[Score]:
    cached = cache.get(trace.run_key)
    if cached is not None:
        return cached
    try:
        scores = scorer.score(bench.samples_by_id[trace.sample_id], trace)
    except Exception as exc:  # not cached: retried on the next grade
        log.warning("scorer %s failed on %s: %s", scorer.kind, trace.run_key, exc)
        return [Score(name=scorer.kind, value=None, explanation=f"scorer error: {type(exc).__name__}: {exc}")]
    cache.put(trace.run_key, scores)
    return scores
