from agent_bench.analysis import Table, estimate
from agent_bench.config import load_experiment
from agent_bench.grading import build_scorers, grade
from agent_bench.report import build_report
from agent_bench.runner import Bench, run


def _run_all(exp):
    """Returns the traces in plan order and, aligned with them, whether each came from the cache."""
    bench = Bench(exp)
    cached_by_key = {}
    traces = run(bench, bench.plan(), on_done=lambda t, cached: cached_by_key.__setitem__(t.run_key, cached))
    return bench, traces, [cached_by_key[t.run_key] for t in traces]


def test_offline_end_to_end(offline):
    bench, traces, events = _run_all(offline)
    assert len(traces) == 7 * 2 * 2 and all(t.ok for t in traces)
    assert not any(events)  # nothing cached on the first run

    rows = grade(bench, traces, build_scorers(bench))
    table = Table.from_rows(rows)
    assert estimate(table, "result_match", "careful").mean == 1.0
    naive = table.sample_means("result_match", "naive")
    assert naive["q01_clients_fr"] == 1.0 and naive["q02_ca_2025"] == 0.0
    assert "q06_marge" not in naive  # no reference SQL: not applicable, not a failure

    report = build_report(bench, traces, rows)
    assert "| `naive` |" in report and "Failures to read" in report
    assert "(4824.0,)" in report  # the wrong figure shows up in the failure list


def test_runs_are_cached_and_invalidated_by_artifact_content(offline, demo_dir):
    _run_all(offline)
    _, _, events = _run_all(load_experiment(demo_dir / "experiment.offline.yaml"))
    assert all(events)  # second run: everything comes from the cache

    # Edit the glossary *without* bumping its version: only variants that use it re-run.
    glossary = demo_dir / "artifacts" / "glossary" / "v2.yaml"
    glossary.write_text(glossary.read_text() + "\n- {term: nouveau, definition: x}\n")
    bench, traces, events = _run_all(load_experiment(demo_dir / "experiment.offline.yaml"))
    rerun = {t.variant for t, cached in zip(traces, events) if not cached}
    assert rerun == {"careful"}


def test_agent_crash_is_an_error_not_a_score(offline):
    bench = Bench(offline)

    class Boom:
        def run(self, *args):
            raise RuntimeError("rate limited")

    bench.resolve(bench.variants[0]).agent = Boom()
    traces = run(bench, bench.plan(variants=["careful"], sample_ids=["q01_clients_fr"]))
    assert all(not t.ok and "rate limited" in t.error for t in traces)
    assert grade(bench, traces, build_scorers(bench)) == []
