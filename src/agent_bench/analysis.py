"""Stage 3: aggregate scores into comparable numbers with honest uncertainty.

* Epochs are averaged per sample first; the sample is the unit of resampling
  (a cluster bootstrap), since runs of one question are not independent.
* Variants are compared to the baseline *paired by sample*: the paired
  difference has far less variance than comparing two means.
* pass^k (tau-bench): probability that k independent runs all succeed, i.e.
  how reliable the agent is, not just how often it is right.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .grading import ScoreRow

N_BOOTSTRAP = 5000
SEED = 0


@dataclass
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def fmt(self, digits: int = 2, pct: bool = False) -> str:
        if self.n == 0 or math.isnan(self.mean):
            return "–"
        if pct:
            return f"{self.mean:.0%} [{self.low:.0%}, {self.high:.0%}]"
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


@dataclass
class Comparison:
    variant: str
    baseline: str
    diff: Estimate
    p_value: float
    wins: int
    losses: int
    ties: int


@dataclass
class Table:
    """metric -> variant -> sample_id -> values over epochs."""

    values: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    @classmethod
    def from_rows(cls, rows: Iterable[ScoreRow]) -> "Table":
        table = cls()
        for r in rows:
            if r.value is not None:
                table.values[r.metric][r.variant][r.sample_id].append(r.value)
        return table

    def sample_means(self, metric: str, variant: str) -> dict[str, float]:
        return {s: statistics.fmean(v) for s, v in self.values[metric][variant].items() if v}

    def metrics(self) -> list[str]:
        return list(self.values)


def bootstrap_mean(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05) -> Estimate:
    if not values:
        return Estimate(math.nan, math.nan, math.nan, 0)
    mean = statistics.fmean(values)
    if len(values) == 1:
        return Estimate(mean, mean, mean, 1)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=k)) for _ in range(n))
    return Estimate(mean, means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1], k)


def estimate(table: Table, metric: str, variant: str, samples: set[str] | None = None) -> Estimate:
    means = table.sample_means(metric, variant)
    return bootstrap_mean([v for s, v in means.items() if samples is None or s in samples])


def pass_hat_k(table: Table, metric: str, variant: str, k: int, threshold: float = 1.0) -> float:
    """Mean over samples of C(c, k) / C(n, k), with c successes out of n epochs."""
    per_sample = []
    for values in table.values[metric][variant].values():
        n, c = len(values), sum(v >= threshold for v in values)
        if n >= k:
            per_sample.append(math.comb(c, k) / math.comb(n, k))
    return statistics.fmean(per_sample) if per_sample else math.nan


def compare(table: Table, metric: str, variant: str, baseline: str,
            n: int = N_BOOTSTRAP, seed: int = SEED) -> Comparison:
    a, b = table.sample_means(metric, variant), table.sample_means(metric, baseline)
    shared = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in shared]
    wins = sum(d > 1e-9 for d in diffs)
    losses = sum(d < -1e-9 for d in diffs)
    return Comparison(
        variant=variant,
        baseline=baseline,
        diff=bootstrap_mean(diffs, n, seed),
        p_value=sign_flip_test(diffs, n, seed),
        wins=wins,
        losses=losses,
        ties=len(diffs) - wins - losses,
    )


def sign_flip_test(diffs: list[float], n: int = N_BOOTSTRAP, seed: int = SEED) -> float:
    """Two-sided paired permutation test of 'no difference'."""
    if not diffs or all(abs(d) < 1e-12 for d in diffs):
        return 1.0
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n)
    )
    return (extreme + 1) / (n + 1)
