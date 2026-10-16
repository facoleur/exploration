import csv

from agent_bench.calibration import calibrate, export_labels
from agent_bench.grading import ScoreRow
from agent_bench.runner import Bench, run


def test_export_then_calibrate(offline, tmp_path):
    bench = Bench(offline)
    traces = run(bench, bench.plan(variants=["careful"]))
    # Pretend the judge said "correct" everywhere.
    rows = [ScoreRow(run_key=t.run_key, variant=t.variant, sample_id=t.sample_id, epoch=t.epoch, category="c",
                     split="dev", scorer="rubric_judge", metric="judge.correct", value=1.0) for t in traces]
    path = tmp_path / "labels.csv"
    assert export_labels(bench, traces, rows, path, n=4) == 4

    with path.open() as f:
        records = list(csv.DictReader(f))
    for i, rec in enumerate(records):
        rec["human.correct"] = "oui" if i < 3 else "non"  # the human disagrees once
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (agreement,) = calibrate(path)
    assert agreement.criterion == "correct" and agreement.n == 4
    assert agreement.accuracy == 0.75 and len(agreement.disagreements) == 1
