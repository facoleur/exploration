"""Agents under test.

An agent takes a question, a system prompt and a toolset, and returns an
``AgentResult``. Register a new implementation (another provider, another
framework, a planner/executor design...) and select it with ``agent.kind``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

from ..registry import Registry
from ..tools import Tool, ToolError
from ..trace import AgentResult, ToolCall


class Agent(Protocol):
    def __init__(self, base_dir: Path, **params: Any): ...

    def run(self, question: str, system_prompt: str, tools: list[Tool]) -> AgentResult: ...


AGENTS: Registry[type] = Registry("agent")


def execute_tool(tools: dict[str, Tool], name: str, tool_input: dict[str, Any]) -> ToolCall:
    """Run one tool call; failures become error results so the agent can recover."""
    start = time.perf_counter()
    tool = tools.get(name)
    if tool is None:
        output, is_error = f"Unknown tool '{name}'.", True
    else:
        try:
            output, is_error = tool(**tool_input), False
        except ToolError as exc:
            output, is_error = str(exc), True
        except TypeError as exc:
            output, is_error = f"Invalid arguments for {name}: {exc}", True
    return ToolCall(
        name=name,
        input=tool_input,
        output=output,
        is_error=is_error,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def build_agent(kind: str, params: dict[str, Any], base_dir: Path) -> Agent:
    return AGENTS.get(kind)(base_dir=base_dir, **params)


from . import claude, scripted  # noqa: E402,F401
