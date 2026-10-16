"""Tools backed by the enriched artifacts.

Expected artifact shapes (adapt freely, these are the demo's):

* ``tables``:   {table: {description, grain?, columns: {col: description}, joins?: [str]}}
* ``glossary``: [{term, definition, synonyms?: [str], tables?: [str], sql_hint?}]
* ``flows``:    [{name, description, steps: [str], tables?: [str]}]
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from . import TOOLS, Tool, ToolContext, ToolError, object_schema


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1}


def _rank(query: str, items: Iterable[tuple[Any, str]], limit: int = 5) -> list[Any]:
    """Keyword-overlap ranking. Swap for embeddings/BM25 behind the same tool to test retrieval."""
    q = _tokens(query)
    scored = [(len(q & _tokens(text)), item) for item, text in items]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: -s[0])
    return [item for _, item in scored[:limit]]


def _enriched_tables(ctx: ToolContext) -> dict[str, Any]:
    return ctx.artifacts["tables"].data if "tables" in ctx.artifacts else {}


@TOOLS.register("list_tables")
def list_tables(ctx: ToolContext) -> Tool:
    def fn() -> list[dict]:
        enriched = _enriched_tables(ctx)
        return [
            {"table": t, "description": enriched.get(t, {}).get("description", "")}
            for t in ctx.db.table_names()
        ]

    return Tool(
        name="list_tables",
        description="List every table in the database with a one-line description when available.",
        input_schema=object_schema({}),
        fn=fn,
    )


@TOOLS.register("describe_table")
def describe_table(ctx: ToolContext) -> Tool:
    def fn(table: str) -> dict:
        if table not in ctx.db.table_names():
            raise ToolError(f"Unknown table '{table}'. Call list_tables to see valid names.")
        columns = ctx.db.execute(f'SELECT * FROM "{table}" LIMIT 0').columns
        enriched = _enriched_tables(ctx).get(table, {})
        column_docs = enriched.get("columns", {})
        return {
            "table": table,
            "description": enriched.get("description", ""),
            "grain": enriched.get("grain", ""),
            "columns": [{"name": c, "description": column_docs.get(c, "")} for c in columns],
            "joins": enriched.get("joins", []),
        }

    return Tool(
        name="describe_table",
        description="Columns of a table, with business descriptions, grain and join keys when available.",
        input_schema=object_schema({"table": {"type": "string"}}),
        fn=fn,
    )


@TOOLS.register("search_catalog")
def search_catalog(ctx: ToolContext) -> Tool:
    def fn(query: str) -> list[dict]:
        docs = []
        for table, meta in ctx.artifacts["tables"].data.items():
            text = " ".join([table, meta.get("description", ""), *meta.get("columns", {}).keys(),
                             *meta.get("columns", {}).values()])
            docs.append(({"table": table, "description": meta.get("description", "")}, text))
        return _rank(query, docs)

    return Tool(
        name="search_catalog",
        description="Find the tables most relevant to a question, searching enriched table and column descriptions.",
        input_schema=object_schema({"query": {"type": "string"}}),
        fn=fn,
        requires=["tables"],
    )


@TOOLS.register("lookup_glossary")
def lookup_glossary(ctx: ToolContext) -> Tool:
    def fn(term: str) -> list[dict]:
        entries = ctx.artifacts["glossary"].data
        docs = [(e, " ".join([e["term"], *e.get("synonyms", []), e.get("definition", "")])) for e in entries]
        return _rank(term, docs, limit=3) or [{"result": f"No glossary entry matches '{term}'."}]

    return Tool(
        name="lookup_glossary",
        description="Business definition of a term (KPI, status, acronym), its synonyms and the tables that hold it.",
        input_schema=object_schema({"term": {"type": "string"}}),
        fn=fn,
        requires=["glossary"],
    )


@TOOLS.register("lookup_flow")
def lookup_flow(ctx: ToolContext) -> Tool:
    def fn(query: str) -> list[dict]:
        flows = ctx.artifacts["flows"].data
        docs = [(f, " ".join([f["name"], f.get("description", ""), *f.get("steps", [])])) for f in flows]
        return _rank(query, docs, limit=3) or [{"result": f"No business flow matches '{query}'."}]

    return Tool(
        name="lookup_flow",
        description="Describe a business process (operations, data flows) step by step, with the tables involved.",
        input_schema=object_schema({"query": {"type": "string"}}),
        fn=fn,
        requires=["flows"],
    )
