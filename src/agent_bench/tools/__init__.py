"""Agent tools.

A tool factory receives a ``ToolContext`` (the variant's artifacts + the DB)
and returns a ``Tool``. Tools never know which artifact *version* they get,
and artifacts never know which tools read them: that is what makes each
ablation a one-line config change.

    @TOOLS.register("lookup_glossary")
    def lookup_glossary(ctx: ToolContext) -> Tool: ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..artifacts import ArtifactBundle
from ..db import Database
from ..registry import Registry


@dataclass
class ToolContext:
    artifacts: ArtifactBundle
    db: Database


class ToolError(Exception):
    """Raised by a tool to return an error result the agent can recover from."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    # Artifacts this tool reads; checked when the toolset is built.
    requires: list[str] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> str:
        result = self.fn(**kwargs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


ToolFactory = Callable[[ToolContext], Tool]
TOOLS: Registry[ToolFactory] = Registry("tool")


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def build_toolset(names: list[str], ctx: ToolContext) -> list[Tool]:
    tools = []
    for name in names:
        tool = TOOLS.get(name)(ctx)
        missing = [a for a in tool.requires if a not in ctx.artifacts]
        if missing:
            raise ValueError(f"Tool '{name}' needs artifacts {missing}, which this variant does not provide")
        tools.append(tool)
    return tools


# Register the built-in tools.
from . import catalog, sql  # noqa: E402,F401
