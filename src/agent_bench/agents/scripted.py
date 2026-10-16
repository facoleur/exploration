"""Replays fixed tool calls and answers. For tests and for dry-running the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..tools import Tool
from ..trace import AgentResult
from . import AGENTS, execute_tool


@AGENTS.register("scripted")
class ScriptedAgent:
    """``script`` maps a question to ``{steps: [{tool, input}], answer}``.

    Unknown questions get an empty answer, which the scorers will mark as failed.
    """

    def __init__(self, base_dir: Path, script: str | dict[str, Any], **_: Any):
        if isinstance(script, str):
            path = Path(script)
            script = yaml.safe_load((path if path.is_absolute() else base_dir / path).read_text())
        self.script: dict[str, Any] = script

    def run(self, question: str, system_prompt: str, tools: list[Tool]) -> AgentResult:
        by_name = {t.name: t for t in tools}
        entry = self.script.get(question, {})
        result = AgentResult(model_served="scripted", stop_reason="end_turn")
        for step in entry.get("steps", []):
            result.tool_calls.append(execute_tool(by_name, step["tool"], step.get("input", {})))
            result.turns += 1
        result.turns += 1
        result.final_answer = entry.get("answer", "")
        return result
