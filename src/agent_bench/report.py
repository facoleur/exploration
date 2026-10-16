"""Markdown report + long-format CSV (one row per score, ready for pandas)."""

from __future__ import annotations

import csv
import datetime as dt
import math
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import Table, compare, estimate, pass_hat_k
from .grading import ScoreRow
from .runner import Bench
from .trace import Trace

# Metrics shown in the leaderboard when present, after the primary one.
LEADERBOARD_METRICS = ["judge", "answer_values", "tables_recall", "tool_calls", "cost_usd", "latency_s"]


def write_csv(rows: list[ScoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ScoreRow.model_fields))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.model_dump())


def build_report(bench: Bench, traces: list[Trace], rows: list[ScoreRow], split: str | None = None) -> str:
    exp = bench.experiment
    if split:
        rows = [r for r in rows if r.split == split]
        traces = [t for t in traces if bench.samples_by_id[t.sample_id].split == split]
    table = Table.from_rows(rows)
    primary = exp.primary_metric
    variants = [v.name for v in bench.variants if any(t.variant == v.name for t in traces)]
    baseline = exp.baseline.name
    k = exp.epochs

    runs = defaultdict(Counter)
    for t in traces:
        runs[t.variant]["ok" if t.ok else "error"] += 1
        runs[t.variant]["fallback"] += t.fallback_used

    out = [
        f"# {exp.name}",
        "",
        f"_{dt.datetime.now():%Y-%m-%d %H:%M}_ · {len({t.sample_id for t in traces})} questions"
        f" · {k} epochs · {len(variants)} variants · split: {split or 'all'}",
        "",
        f"Primary metric: **{primary}**. Brackets are 95% bootstrap CIs over questions. "
        f"Δ is paired by question against `{baseline}`; p from a sign-flip permutation test; "
        f"W/L/T = questions won / lost / tied. pass^{k} = share of questions solved on all {k} runs.",
        "",
        "## Leaderboard",
        "",
    ]
    extra = [m for m in LEADERBOARD_METRICS if m in table.metrics() and m != primary]
    header = ["variant", primary, f"pass^{k}", "Δ vs baseline", "p", "W/L/T", *extra, "runs ok/err", "fallback"]
    out += [_row(header), _row(["---"] * len(header))]
    for v in variants:
        cells = [f"`{v}`", estimate(table, primary, v).fmt(pct=True), _pct(pass_hat_k(table, primary, v, k))]
        if v == baseline:
            cells += ["–", "–", "–"]
        else:
            c = compare(table, primary, v, baseline)
            cells += [f"{c.diff.mean:+.0%} [{c.diff.low:+.0%}, {c.diff.high:+.0%}]",
                      f"{c.p_value:.3f}", f"{c.wins}/{c.losses}/{c.ties}"]
        for m in extra:
            e = estimate(table, m, v)
            cells.append("–" if e.n == 0 else (f"{e.mean:.0%}" if m in ("judge", "answer_values", "tables_recall")
                                                  else f"{e.mean:.3g}"))
        cells += [f"{runs[v]['ok']}/{runs[v]['error']}", str(runs[v]["fallback"])]
        out.append(_row(cells))

    judge_metrics = sorted(m for m in table.metrics() if m.startswith("judge."))
    if judge_metrics:
        out += ["", "## Judge criteria (pass rate)", "", _row(["variant", *judge_metrics]),
                _row(["---"] * (len(judge_metrics) + 1))]
        for v in variants:
            out.append(_row([f"`{v}`", *(_pct(estimate(table, m, v).mean) for m in judge_metrics)]))

    categories = sorted({bench.samples_by_id[t.sample_id].category for t in traces})
    out += ["", f"## {primary} by question category", "", _row(["variant", *categories]),
            _row(["---"] * (len(categories) + 1))]
    for v in variants:
        cells = []
        for cat in categories:
            ids = {s.id for s in bench.samples if s.category == cat}
            e = estimate(table, primary, v, ids)
            cells.append("–" if e.n == 0 else f"{e.mean:.0%} (n={e.n})")
        out.append(_row([f"`{v}`", *cells]))

    sample_ids = sorted({t.sample_id for t in traces})
    out += ["", f"## {primary} per question (mean over epochs)", "", _row(["question", "category", *variants]),
            _row(["---"] * (len(variants) + 2))]
    for sid in sample_ids:
        cells = []
        for v in variants:
            vals = table.values[primary][v].get(sid, [])
            cells.append("–" if not vals else f"{sum(vals) / len(vals):.2f}")
        out.append(_row([f"`{sid}`", bench.samples_by_id[sid].category, *cells]))

    failures = [r for r in rows if r.metric == primary and r.value is not None and r.value < 1]
    out +=["", "## Failures to read", "",
            "Read these traces and tag each failure with a cause (wrong table, bad join, misread term, "
            "hallucination...). The tags tell you what to change next.", ""]
    if not failures:
        out.append("None.")
    for r in sorted(failures, key=lambda r: (r.variant != baseline, r.variant, r.sample_id, r.epoch))[:60]:
        # report.md sits next to traces/, so the link is relative.
        out.append(f"- `{r.variant}` · `{r.sample_id}` #{r.epoch}: {r.explanation or 'wrong'} "
                   f"→ [trace](traces/{r.run_key}.json)")
    errors = [t for t in traces if not t.ok]
    if errors:
        out += ["", "## Errored runs (excluded from scores)", ""]
        out += [f"- `{t.variant}` · `{t.sample_id}` #{t.epoch}: {t.error.splitlines()[0]}" for t in errors[:30]]
    return "\n".join(out) + "\n"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _pct(x: float) -> str:
    return "–" if math.isnan(x) else f"{x:.0%}"
