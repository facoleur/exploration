"""Check the LLM judge against human labels before trusting it.

1. ``bench export-labels`` writes a CSV of runs with the judge's verdicts and
   empty ``human.<criterion>`` columns.
2. Fill the human columns (1/0) without looking at the judge columns.
3. ``bench calibrate`` reports agreement and Cohen's kappa per criterion, and
   lists the disagreements: read them to fix the rubric wording.

Rule of thumb: kappa >= 0.6 is usable, >= 0.8 is good. Below that, the judge's
numbers are mostly noise about your question.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .grading import ScoreRow
from .runner import Bench
from .trace import Trace

_TRUE = {"1", "true", "yes", "y", "oui", "pass", "ok"}
_FALSE = {"0", "false", "no", "n", "non", "fail", "ko"}


def export_labels(bench: Bench, traces: list[Trace], rows: list[ScoreRow], path: Path,
                  n: int = 40, seed: int = 0) -> int:
    verdicts: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.metric.startswith("judge.") and r.value is not None:
            verdicts[r.run_key][r.metric.removeprefix("judge.")] = r.value
    judged = [t for t in traces if t.run_key in verdicts]
    random.Random(seed).shuffle(judged)
    picked = judged[:n]
    criteria = sorted({c for v in verdicts.values() for c in v})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_key", "variant", "sample_id", "question", "reference_answer", "agent_answer",
                         *(f"human.{c}" for c in criteria), *(f"judge.{c}" for c in criteria)])
        for t in picked:
            s = bench.samples_by_id[t.sample_id]
            v = verdicts[t.run_key]
            writer.writerow([t.run_key, t.variant, t.sample_id, s.question, s.reference_answer or "",
                             t.final_answer, *([""] * len(criteria)),
                             *(("" if c not in v else int(v[c])) for c in criteria)])
    return len(picked)


@dataclass
class Agreement:
    criterion: str
    n: int
    accuracy: float
    kappa: float
    judge_pass_rate: float
    human_pass_rate: float
    disagreements: list[tuple[str, int, int]]  # (run_key, human, judge)


def calibrate(path: Path) -> list[Agreement]:
    with path.open() as f:
        records = list(csv.DictReader(f))
    criteria = sorted(c.removeprefix("human.") for c in (records[0].keys() if records else []) if c.startswith("human."))
    results = []
    for c in criteria:
        pairs = []
        for rec in records:
            human, judge = _bool(rec.get(f"human.{c}", "")), _bool(rec.get(f"judge.{c}", ""))
            if human is not None and judge is not None:
                pairs.append((rec["run_key"], int(human), int(judge)))
        if not pairs:
            continue
        results.append(Agreement(
            criterion=c,
            n=len(pairs),
            accuracy=sum(h == j for _, h, j in pairs) / len(pairs),
            kappa=cohen_kappa([h for _, h, _ in pairs], [j for _, _, j in pairs]),
            judge_pass_rate=sum(j for _, _, j in pairs) / len(pairs),
            human_pass_rate=sum(h for _, h, _ in pairs) / len(pairs),
            disagreements=[p for p in pairs if p[1] != p[2]],
        ))
    return results


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _bool(value: str) -> bool | None:
    v = value.strip().lower()
    return True if v in _TRUE else False if v in _FALSE else None
