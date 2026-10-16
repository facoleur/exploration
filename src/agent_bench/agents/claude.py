"""Claude tool-use agent with a hand-written loop.

The loop is written out (rather than using the SDK tool runner) so that every
tool call, token count and stop reason lands in the trace exactly as it happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic

from ..tools import Tool
from ..trace import AgentResult, Usage
from . import AGENTS, execute_tool

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@AGENTS.register("claude")
class ClaudeAgent:
    def __init__(
        self,
        base_dir: Path,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        max_turns: int = 20,
        fallbacks: bool = True,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        # Server-side fallback re-runs a refused request on another model. Kept on by
        # default; the trace records `fallback_used` so such runs can be filtered out.
        self.fallbacks = fallbacks
        self.client = client or anthropic.Anthropic()

    def run(self, question: str, system_prompt: str, tools: list[Tool]) -> AgentResult:
        by_name = {t.name: t for t in tools}
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
            for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        result = AgentResult()

        for _ in range(self.max_turns):
            response = self._create(system_prompt, tool_defs, messages)
            result.turns += 1
            result.stop_reason = response.stop_reason
            result.model_served = response.model
            result.usage.add(_usage(response))
            result.fallback_used |= _fallback_ran(response)
            # Append the content unchanged: thinking blocks must be replayed as-is.
            messages.append({"role": "assistant", "content": response.content})
            result.transcript.append({"role": "assistant", "content": [b.model_dump(mode="json") for b in response.content]})

            if response.stop_reason != "tool_use":
                result.final_answer = "\n".join(b.text for b in response.content if b.type == "text").strip()
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = execute_tool(by_name, block.name, block.input)
                result.tool_calls.append(call)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": call.output, "is_error": call.is_error}
                )
            # All results of a turn go back in a single user message.
            messages.append({"role": "user", "content": tool_results})
            result.transcript.append({"role": "user", "content": tool_results})
        else:
            result.stop_reason = "max_turns"

        return result

    def _create(self, system_prompt: str, tool_defs: list[dict], messages: list[dict]) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=tool_defs,
            messages=messages,
            output_config={"effort": self.effort},
            # Caches the growing prefix: each turn re-reads the previous ones at ~10% cost.
            cache_control={"type": "ephemeral"},
        )
        if self.fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)


def _usage(response: Any) -> Usage:
    u = response.usage
    return Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _fallback_ran(response: Any) -> bool:
    iterations = getattr(response.usage, "iterations", None) or []
    return any(getattr(entry, "type", None) == "fallback_message" for entry in iterations)
