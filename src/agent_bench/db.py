"""Read-only access to the frozen benchmark database.

Grading compares result sets, so the DB must not change between runs: point
this at a snapshot, never at a live database. Add another backend (Postgres,
BigQuery, ...) by implementing the ``Database`` protocol and registering it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import DatabaseSpec
from .registry import Registry


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


class Database(Protocol):
    max_rows: int

    def execute(self, sql: str) -> QueryResult: ...

    def table_names(self) -> list[str]: ...


DATABASES: Registry[Callable[[DatabaseSpec, Path], Database]] = Registry("database")


@DATABASES.register("sqlite")
class SqliteDatabase:
    """SQLite file or ``.sql`` script, opened read-only, with a query time limit."""

    def __init__(self, spec: DatabaseSpec, base_dir: Path, timeout_s: float = 10.0):
        self.max_rows = spec.max_rows
        self.timeout_s = timeout_s
        path = Path(spec.path)
        path = path if path.is_absolute() else base_dir / path
        if path.suffix == ".sql":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.executescript(path.read_text())
        else:
            self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.execute("PRAGMA query_only = ON")
        self._lock = threading.Lock()

    def execute(self, sql: str) -> QueryResult:
        deadline = time.monotonic() + self.timeout_s
        with self._lock:
            # Returning non-zero from the progress handler aborts the query.
            self._conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                cursor = self._conn.execute(sql)
                rows = cursor.fetchall()
            finally:
                self._conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description or []]
        return QueryResult(columns=columns, rows=[tuple(r) for r in rows])

    def table_names(self) -> list[str]:
        result = self.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        return [r[0] for r in result.rows]


def open_database(spec: DatabaseSpec, base_dir: Path) -> Database:
    return DATABASES.get(spec.kind)(spec, base_dir)
