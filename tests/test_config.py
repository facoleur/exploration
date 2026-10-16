from agent_bench.config import Ablation, apply_ablation, expand_grid, load_experiment


def test_demo_variants(demo_dir):
    exp = load_experiment(demo_dir / "experiment.yaml")
    variants = {v.name: v for v in exp.variants()}
    assert list(variants)[0] == "baseline"
    assert variants["glossary_v1"].artifacts["glossary"] == "glossary@v1"
    assert "glossary" not in variants["no_glossary"].artifacts
    assert "lookup_glossary" not in variants["no_glossary"].tools
    assert variants["effort_low"].agent.params["effort"] == "low"
    # The baseline is never mutated by its ablations.
    assert variants["baseline"].agent.params["effort"] == "medium"
    assert "glossary" in variants["baseline"].artifacts


def test_add_tools_and_grid(demo_dir):
    base = load_experiment(demo_dir / "experiment.yaml").baseline
    v = apply_ablation(base, Ablation(name="x", remove_tools=["run_sql"], add_tools=["run_sql", "new_tool"]))
    assert v.tools[-2:] == ["run_sql", "new_tool"]

    grid = expand_grid(base, {"artifacts.glossary": ["glossary@v1", None], "agent.params.effort": ["low", "high"]})
    assert len(grid) == 4
    assert len({g.name for g in grid}) == 4
    assert sum("glossary" not in g.artifacts for g in grid) == 2
