"""Versioned artifacts (glossary, enriched tables, flows, ...).

Layout on disk: ``<artifacts_dir>/<name>/<version>.(yaml|yml|json|md|txt)``.
A reference is written ``name@version`` (e.g. ``glossary@v2``). Every loaded
artifact carries a content hash, so editing a file without bumping its version
still invalidates cached runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_EXTENSIONS = (".yaml", ".yml", ".json", ".md", ".txt")


@dataclass(frozen=True)
class Artifact:
    name: str
    version: str
    data: Any
    sha256: str
    path: Path

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def parse_ref(ref: str) -> tuple[str, str]:
    name, sep, version = ref.partition("@")
    if not sep or not name or not version:
        raise ValueError(f"Artifact reference must look like 'name@version', got '{ref}'")
    return name, version


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, Artifact] = {}

    def load(self, ref: str) -> Artifact:
        if ref in self._cache:
            return self._cache[ref]
        name, version = parse_ref(ref)
        path = self._find(name, version)
        raw = path.read_bytes()
        artifact = Artifact(
            name=name,
            version=version,
            data=_parse(path, raw.decode()),
            sha256=hashlib.sha256(raw).hexdigest()[:12],
            path=path,
        )
        self._cache[ref] = artifact
        return artifact

    def versions(self, name: str) -> list[str]:
        folder = self.root / name
        return sorted(p.stem for p in folder.iterdir() if p.suffix in _EXTENSIONS)

    def _find(self, name: str, version: str) -> Path:
        for ext in _EXTENSIONS:
            candidate = self.root / name / f"{version}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No artifact file for '{name}@{version}' under {self.root / name}")


def _parse(path: Path, text: str) -> Any:
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    return text


class ArtifactBundle:
    """The artifacts resolved for one variant, keyed by name."""

    def __init__(self, artifacts: dict[str, Artifact]):
        self._artifacts = artifacts

    def __contains__(self, name: str) -> bool:
        return name in self._artifacts

    def __getitem__(self, name: str) -> Artifact:
        if name not in self._artifacts:
            raise KeyError(f"Artifact '{name}' is not part of this variant")
        return self._artifacts[name]

    def fingerprint(self) -> dict[str, str]:
        return {name: f"{a.ref}#{a.sha256}" for name, a in sorted(self._artifacts.items())}
