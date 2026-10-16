"""The Claude loop against a fake client: no network, checks the message protocol."""

from pathlib import Path
from types import SimpleNamespace

from agent_bench.agents.claude import FALLBACK_BETA, ClaudeAgent
from agent_bench.tools import Tool, object_schema


def block(type_, **fields):
    return SimpleNamespace(type=type_, model_dump=lambda mode=None: {"type": type_, **fields}, **fields)


def response(content, stop_reason, fallback=False):
    iterations = [SimpleNamespace(type="fallback_message")] if fallback else []
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=100,
                            cache_creation_input_tokens=0, iterations=iterations)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, model="claude-opus-5-5")


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def fake_client(responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), messages


def test_tool_loop_records_everything():
    tool = Tool(name="add", description="add", input_schema=object_schema({"a": {"type": "integer"}}),
                fn=lambda a: {"sum": a + 1})
    client, messages = fake_client([
        response([block("text", text="Let me compute."),
                  block("tool_use", id="t1", name="add", input={"a": 1}),
                  block("tool_use", id="t2", name="missing", input={})], "tool_use"),
        response([block("text", text="The answer is 2.")], "end_turn", fallback=True),
    ])
    agent = ClaudeAgent(Path("."), client=client)
    result = agent.run("1+1?", "system", [tool])

    assert result.final_answer == "The answer is 2."
    assert [c.name for c in result.tool_calls] == ["add", "missing"]
    assert result.tool_calls[0].output == '{"sum": 2}' and result.tool_calls[1].is_error
    assert result.turns == 2 and result.usage.cache_read_input_tokens == 200
    assert result.fallback_used

    first, second = messages.calls
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["tools"][0]["strict"] is True
    # Both tool results go back in ONE user message, after the unchanged assistant content.
    assistant, user = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and len(assistant["content"]) == 3
    assert [r["tool_use_id"] for r in user["content"]] == ["t1", "t2"]


def test_max_turns_and_no_fallback():
    looping = [response([block("tool_use", id=f"t{i}", name="x", input={})], "tool_use") for i in range(3)]
    client, messages = fake_client(looping)
    result = ClaudeAgent(Path("."), client=client, max_turns=3, fallbacks=False).run("q", "s", [])
    assert result.stop_reason == "max_turns" and result.turns == 3
    assert "fallbacks" not in messages.calls[0]
