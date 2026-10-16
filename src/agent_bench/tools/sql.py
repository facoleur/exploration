"""Tools that hit the database directly."""

from __future__ import annotations

from . import TOOLS, Tool, ToolContext, ToolError, object_schema


@TOOLS.register("run_sql")
def run_sql(ctx: ToolContext) -> Tool:
    def fn(sql: str) -> dict:
        try:
            result = ctx.db.execute(sql)
        except Exception as exc:  # DB errors are information for the agent, not crashes
            raise ToolError(f"SQL error: {exc}") from exc
        rows = result.rows[: ctx.db.max_rows]
        return {
            "columns": result.columns,
            "rows": [list(r) for r in rows],
            "row_count": len(result.rows),
            "truncated": len(result.rows) > len(rows),
        }

    return Tool(
        name="run_sql",
        description=(
            "Run one read-only SQL query against the database and return the result rows. "
            f"At most {ctx.db.max_rows} rows are returned; aggregate in SQL rather than reading raw rows."
        ),
        input_schema=object_schema({"sql": {"type": "string", "description": "A single SELECT statement."}}),
        fn=fn,
    )
