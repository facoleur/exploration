"""Command line: bench {plan,run,grade,report,all,export-labels,calibrate} EXPERIMENT.yaml"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .calibration import calibrate, export_labels
from .config import load_experiment
from .grading import build_scorers, grade
from .report import build_report, write_csv
from .runner import Bench, RunItem, TraceStore, run
from .trace import Trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("experiment", type=Path)
        p.add_argument("--variants", help="comma-separated variant names (default: all)")
        p.add_argument("--split", choices=["dev", "holdout"], help="only questions from this split")
        p.add_argument("--samples", help="comma-separated sample ids")
        p.add_argument("--epochs", type=int, help="override the experiment's epoch count")
        return p

    add("plan", "show variants and how many runs are cached / to do")
    p_run = add("run", "run the agent (cached runs are skipped)")
    p_run.add_argument("--keep-errors", action="store_true", help="do not retry runs that errored")
    add("grade", "score existing traces")
    add("report", "write report.md and scores.csv")
    p_all = add("all", "run + grade + report")
    p_all.add_argument("--keep-errors", action="store_true")
    p_exp = add("export-labels", "write a CSV of judged runs to label by hand")
    p_exp.add_argument("--n", type=int, default=40)
    p_exp.add_argument("--out", type=Path)
    p_cal = sub.add_parser("calibrate", help="agreement between human labels and the judge")
    p_cal.add_argument("labels", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "calibrate":
        return _calibrate(args.labels)

    experiment = load_experiment(args.experiment)
    if args.epochs:
        experiment.epochs = args.epochs
    bench = Bench(experiment)
    items = bench.plan(
        variants=_csv(args.variants), split=args.split, sample_ids=_csv(args.samples)
    )
    if not items:
        print("Nothing planned: check --variants / --split / --samples.", file=sys.stderr)
        return 1

    if args.command == "plan":
        return _plan(bench, items)
    if args.command in ("run", "all"):
        traces = _run(bench, items, retry_errors=not args.keep_errors)
    else:
        traces = _load(bench, items)
    if args.command == "run":
        return 0

    rows = grade(bench, traces, build_scorers(bench))
    if args.command == "export-labels":
        out = args.out or experiment.run_dir / "labels.csv"
        count = export_labels(bench, traces, rows, out, n=args.n)
        print(f"Wrote {count} runs to {out}. Fill the human.* columns, then: bench calibrate {out}")
        return 0

    write_csv(rows, experiment.run_dir / "scores.csv")
    report = build_report(bench, traces, rows, split=args.split)
    (experiment.run_dir / "report.md").write_text(report)
    print(report)
    print(f"Wrote {experiment.run_dir / 'report.md'} and {experiment.run_dir / 'scores.csv'}")
    return 0


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _plan(bench: Bench, items: list[RunItem]) -> int:
    store = TraceStore(bench.experiment.run_dir)
    for v in bench.variants:
        mine = [i for i in items if i.variant == v.name]
        if not mine:
            continue
        cached = sum(store.get(i.run_key) is not None for i in mine)
        resolved = bench.resolve(v)
        print(f"{v.name:<32} runs={len(mine):<4} cached={cached:<4} "
              f"artifacts={sorted(resolved.fingerprint['artifacts'].values())} tools={v.tools}")
    return 0


def _run(bench: Bench, items: list[RunItem], retry_errors: bool) -> list[Trace]:
    done = {"n": 0}

    def progress(trace: Trace, cached: bool) -> None:
        done["n"] += 1
        status = "cached" if cached else ("ok" if trace.ok else "ERROR")
        print(f"[{done['n']}/{len(items)}] {trace.variant} · {trace.sample_id} #{trace.epoch}: {status}",
              file=sys.stderr)

    return run(bench, items, retry_errors=retry_errors, on_done=progress)


def _load(bench: Bench, items: list[RunItem]) -> list[Trace]:
    store = TraceStore(bench.experiment.run_dir)
    traces = [t for t in (store.get(i.run_key) for i in items) if t is not None]
    if len(traces) < len(items):
        print(f"{len(items) - len(traces)} planned runs have no trace yet (run `bench run` first).", file=sys.stderr)
    return traces


def _calibrate(path: Path) -> int:
    results = calibrate(path)
    if not results:
        print("No labelled rows found: fill the human.* columns with 1/0.", file=sys.stderr)
        return 1
    for a in results:
        print(f"{a.criterion:<24} n={a.n:<4} agreement={a.accuracy:.0%} kappa={a.kappa:.2f} "
              f"judge_pass={a.judge_pass_rate:.0%} human_pass={a.human_pass_rate:.0%}")
        for run_key, human, judge in a.disagreements[:10]:
            print(f"    disagreement {run_key}: human={human} judge={judge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
