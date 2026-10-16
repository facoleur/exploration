"""Tiny name -> factory registry shared by tools, agents and scorers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(item: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = item
            return item

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
