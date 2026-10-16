"""What one agent run leaves behind. Everything grading needs lives here."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False
    latency_ms: float = 0.0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens


class AgentResult(BaseModel):
    """What an Agent implementation returns."""

    final_answer: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    turns: int = 0
    stop_reason: str | None = None
    model_served: str | None = None
    fallback_used: bool = False
    # Raw provider transcript, for reading traces during error analysis.
    transcript: list[dict[str, Any]] = Field(default_factory=list)


class Trace(AgentResult):
    """AgentResult plus the run's identity, timing and failure state."""

    run_key: str
    variant: str
    sample_id: str
    epoch: int
    variant_fingerprint: dict[str, Any]
    latency_s: float = 0.0
    error: str | None = None
    started_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def sql_queries(self, tool_name: str = "run_sql", successful_only: bool = True) -> list[str]:
        return [
            c.input.get("sql", "")
            for c in self.tool_calls
            if c.name == tool_name and not (successful_only and c.is_error)
        ]
