"""Experiment and variant configuration.

An experiment is a baseline variant plus a list of ablations (one change each)
and/or a factorial grid. Every variant is plain data; nothing here knows how
tools or agents work.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

_DELETE = object()


class ComponentSpec(BaseModel):
    """A pluggable component: ``kind`` picks the registered class, the rest are its params."""

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        # Allow `{kind: claude, model: x}` as shorthand for `{kind: claude, params: {model: x}}`.
        if isinstance(data, dict) and "params" not in data:
            data = dict(data)
            kind = data.pop("kind")
            return {"kind": kind, "params": data}
        return data


class Variant(BaseModel):
    name: str
    agent: ComponentSpec
    prompt: str  # path to the system prompt, relative to the experiment file
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> "name@version"
    tools: list[str] = Field(default_factory=list)
    description: str = ""


class Ablation(BaseModel):
    name: str
    description: str = ""
    # Dotted-path overrides on the baseline, e.g. {"artifacts.glossary": "glossary@v1"}.
    # A null value deletes the key.
    set: dict[str, Any] = Field(default_factory=dict)
    add_tools: list[str] = Field(default_factory=list)
    remove_tools: list[str] = Field(default_factory=list)


class DatabaseSpec(BaseModel):
    kind: str = "sqlite"
    path: str  # .db file, or a .sql script loaded into an in-memory DB
    max_rows: int = 50


class Experiment(BaseModel):
    name: str
    dataset: str
    artifacts_dir: str = "artifacts"
    database: DatabaseSpec
    epochs: int = 3
    concurrency: int = 4
    primary_metric: str = "result_match"
    baseline: Variant
    ablations: list[Ablation] = Field(default_factory=list)
    # Optional factorial grid over dotted paths: {"artifacts.glossary": [...], "agent.params.effort": [...]}
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    scorers: list[ComponentSpec] = Field(default_factory=list)
    # $ per million tokens: {model: {input, output, cache_read, cache_write}}
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)
    output_dir: str = "runs"

    # Set by load_experiment; every relative path resolves against it.
    base_dir: Path = Path(".")

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.base_dir / p

    @property
    def run_dir(self) -> Path:
        return self.path(self.output_dir) / self.name

    def variants(self) -> list[Variant]:
        """Baseline, then each ablation, then each grid cell; names must be unique."""
        variants = [self.baseline]
        variants += [apply_ablation(self.baseline, a) for a in self.ablations]
        variants += expand_grid(self.baseline, self.grid)
        names = [v.name for v in variants]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate variant names: {duplicates}")
        return variants


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    experiment = Experiment.model_validate(data)
    experiment.base_dir = path.parent
    return experiment


def apply_ablation(base: Variant, ablation: Ablation) -> Variant:
    data = base.model_dump()
    for dotted, value in ablation.set.items():
        _set_path(data, dotted, _DELETE if value is None else value)
    tools = [t for t in data["tools"] if t not in ablation.remove_tools]
    tools += [t for t in ablation.add_tools if t not in tools]
    data["tools"] = tools
    data["name"] = ablation.name
    data["description"] = ablation.description
    return Variant.model_validate(data)


def expand_grid(base: Variant, grid: dict[str, list[Any]]) -> list[Variant]:
    if not grid:
        return []
    keys = list(grid)
    variants = []
    for values in itertools.product(*(grid[k] for k in keys)):
        data = base.model_dump()
        label = []
        for key, value in zip(keys, values):
            _set_path(data, key, _DELETE if value is None else value)
            label.append(f"{key.split('.')[-1]}={value}")
        data["name"] = "grid[" + ",".join(label) + "]"
        data["description"] = "factorial grid cell"
        variants.append(Variant.model_validate(data))
    return variants


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    if value is _DELETE:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = copy.deepcopy(value)
