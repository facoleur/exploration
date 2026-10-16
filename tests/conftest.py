import shutil
from pathlib import Path

import pytest

from agent_bench.config import load_experiment

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """A private copy of the demo, so tests can edit artifacts and write runs."""
    target = tmp_path / "demo"
    shutil.copytree(DEMO, target, ignore=shutil.ignore_patterns("runs"))
    return target


@pytest.fixture
def offline(demo_dir: Path):
    return load_experiment(demo_dir / "experiment.offline.yaml")
