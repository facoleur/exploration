import math

from agent_bench.analysis import Table, bootstrap_mean, compare, pass_hat_k, sign_flip_test
from agent_bench.calibration import cohen_kappa
from agent_bench.grading import ScoreRow


def rows(variant: str, values: dict[str, list[float]]) -> list[ScoreRow]:
    return [
        ScoreRow(run_key=f"{variant}{s}{i}", variant=variant, sample_id=s, epoch=i, category="c", split="dev",
                 scorer="x", metric="m", value=v)
        for s, vs in values.items() for i, v in enumerate(vs)
    ]


def test_pass_hat_k():
    table = Table.from_rows(rows("a", {"q1": [1, 1, 1], "q2": [1, 0, 1], "q3": [0, 0, 0]}))
    assert pass_hat_k(table, "m", "a", 1) == (1 + 2 / 3 + 0) / 3
    assert pass_hat_k(table, "m", "a", 3) == 1 / 3


def test_paired_comparison():
    base = rows("base", {f"q{i}": [0, 0] for i in range(10)})
    better = rows("new", {f"q{i}": [1, 1] if i < 8 else [0, 0] for i in range(10)})
    c = compare(Table.from_rows(base + better), "m", "new", "base")
    assert (c.wins, c.losses, c.ties) == (8, 0, 2)
    assert math.isclose(c.diff.mean, 0.8) and c.diff.low > 0
    assert c.p_value < 0.01


def test_stats_edge_cases():
    assert sign_flip_test([0, 0, 0]) == 1.0
    assert bootstrap_mean([]).n == 0
    e = bootstrap_mean([0.5])
    assert e.low == e.high == 0.5


def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
