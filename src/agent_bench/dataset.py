"""Benchmark questions with their references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .registry import stable_hash


class Sample(BaseModel):
    id: str
    question: str
    # What a good answer says, in plain words. Used by the LLM judge.
    reference_answer: str | None = None
    # Executed against the frozen DB and compared with the agent's last query.
    reference_sql: str | None = None
    # Values (numbers, names) the final answer must mention.
    expected_values: list[str] = Field(default_factory=list)
    # Tables the agent should look at; drives trajectory metrics.
    expected_tables: list[str] = Field(default_factory=list)
    # lookup | aggregation | join | business_term | ambiguous | unanswerable ...
    category: str = "general"
    answerable: bool = True
    # dev: free to look at while improving the agent; holdout: report only.
    split: Literal["dev", "holdout"] = "dev"
    tags: list[str] = Field(default_factory=list)

    def fingerprint(self) -> str:
        # Only what the agent sees affects a run; references only affect grading.
        return stable_hash({"id": self.id, "question": self.question})


def load_samples(path: Path, split: str | None = None) -> list[Sample]:
    samples = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            samples.append(Sample.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_no}: invalid sample: {exc}") from exc
    ids = [s.id for s in samples]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate sample ids in {path}: {sorted(duplicates)}")
    if split:
        samples = [s for s in samples if s.split == split]
    return samples
