"""Stage 1: run every (variant, sample, epoch) and store one trace per run.

Traces are content-addressed: the run key hashes the variant's full
fingerprint (agent params, prompt text, artifact *contents*, toolset), the
question and the epoch. Re-running only executes what changed, and stale
traces are never picked up because the report recomputes keys from the config.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import Agent, build_agent
from .artifacts import ArtifactBundle, ArtifactStore
from .config import Experiment, Variant
from .dataset import Sample, load_samples
from .db import Database, open_database
from .registry import stable_hash
from .tools import Tool, ToolContext, build_toolset
from .trace import Trace

log = logging.getLogger(__name__)


@dataclass
class ResolvedVariant:
    variant: Variant
    system_prompt: str
    tools: list[Tool]
    agent: Agent
    fingerprint: dict[str, Any]


@dataclass(frozen=True)
class RunItem:
    variant: str
    sample: Sample
    epoch: int
    run_key: str


class TraceStore:
    def __init__(self, run_dir: Path):
        self.dir = run_dir / "traces"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_key: str) -> Path:
        return self.dir / f"{run_key}.json"

    def get(self, run_key: str) -> Trace | None:
        path = self.path(run_key)
        return Trace.model_validate_json(path.read_text()) if path.exists() else None

    def put(self, trace: Trace) -> None:
        tmp = self.path(trace.run_key).with_suffix(".tmp")
        tmp.write_text(trace.model_dump_json(indent=2))
        tmp.replace(self.path(trace.run_key))


class Bench:
    """Everything resolved from an experiment file, shared by all stages."""

    def __init__(self, experiment: Experiment, db: Database | None = None, agent_overrides: dict[str, Any] | None = None):
        self.experiment = experiment
        self.store = ArtifactStore(experiment.path(experiment.artifacts_dir))
        self.db = db or open_database(experiment.database, experiment.base_dir)
        self.samples = load_samples(experiment.path(experiment.dataset))
        self.samples_by_id = {s.id: s for s in self.samples}
        self.variants = experiment.variants()
        # Extra constructor args injected into every agent (e.g. a fake client in tests).
        self._agent_overrides = agent_overrides or {}
        self._resolved: dict[str, ResolvedVariant] = {}

    def resolve(self, variant: Variant) -> ResolvedVariant:
        if variant.name in self._resolved:
            return self._resolved[variant.name]
        bundle = ArtifactBundle({name: self.store.load(ref) for name, ref in variant.artifacts.items()})
        prompt_text = self.experiment.path(variant.prompt).read_text()
        tools = build_toolset(variant.tools, ToolContext(artifacts=bundle, db=self.db))
        fingerprint = {
            "agent": variant.agent.model_dump(),
            "prompt": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
            "artifacts": bundle.fingerprint(),
            # Tool descriptions are part of the prompt the model sees.
            "tools": {t.name: stable_hash([t.description, t.input_schema]) for t in tools},
        }
        agent = build_agent(
            variant.agent.kind, {**variant.agent.params, **self._agent_overrides}, self.experiment.base_dir
        )
        resolved = ResolvedVariant(variant, prompt_text, tools, agent, fingerprint)
        self._resolved[variant.name] = resolved
        return resolved

    def plan(
        self,
        variants: list[str] | None = None,
        split: str | None = None,
        sample_ids: list[str] | None = None,
    ) -> list[RunItem]:
        items = []
        for variant in self.variants:
            if variants and variant.name not in variants:
                continue
            fp = self.resolve(variant).fingerprint
            for sample in self.samples:
                if split and sample.split != split:
                    continue
                if sample_ids and sample.id not in sample_ids:
                    continue
                for epoch in range(self.experiment.epochs):
                    key = stable_hash({"variant": fp, "sample": sample.fingerprint(), "epoch": epoch}, 16)
                    items.append(RunItem(variant.name, sample, epoch, key))
        return items


def run(
    bench: Bench,
    items: list[RunItem],
    retry_errors: bool = True,
    on_done: Callable[[Trace, bool], None] | None = None,
) -> list[Trace]:
    """Execute the planned runs that are not cached yet. Returns every trace, cached or new."""
    store = TraceStore(bench.experiment.run_dir)
    traces: dict[str, Trace] = {}
    todo = []
    for item in items:
        cached = store.get(item.run_key)
        if cached and (cached.ok or not retry_errors):
            traces[item.run_key] = cached
            if on_done:
                on_done(cached, True)
        else:
            todo.append(item)

    variants = {v.name: v for v in bench.variants}
    with ThreadPoolExecutor(max_workers=bench.experiment.concurrency) as pool:
        futures = {pool.submit(_run_one, bench.resolve(variants[i.variant]), i): i for i in todo}
        for future in as_completed(futures):
            trace = future.result()
            store.put(trace)
            traces[trace.run_key] = trace
            if on_done:
                on_done(trace, False)
    return [traces[i.run_key] for i in items]


def _run_one(resolved: ResolvedVariant, item: RunItem) -> Trace:
    start = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    identity = dict(
        run_key=item.run_key,
        variant=item.variant,
        sample_id=item.sample.id,
        epoch=item.epoch,
        variant_fingerprint=resolved.fingerprint,
        started_at=started_at,
    )
    try:
        result = resolved.agent.run(item.sample.question, resolved.system_prompt, resolved.tools)
        return Trace(**result.model_dump(), **identity, latency_s=time.perf_counter() - start)
    except Exception as exc:  # an infra failure is recorded, never silently scored as a wrong answer
        log.warning("run %s/%s#%d failed: %s", item.variant, item.sample.id, item.epoch, exc)
        return Trace(
            **identity,
            latency_s=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        )
