"""One streamed agent turn, with timing. Shared by the server and the eval, so both
measure TTFT, thinking time, tool time and retries with exactly the same definitions."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

TOOL_LABELS = {
    "aggregate_events": "Counting events",
    "find_events": "Searching the mod log",
    "list_tables": "Discovering tables",
    "describe_table": "Reading the schema",
    "run_sql": "Running a query",
}


@dataclass
class TurnMetrics:
    first_thinking_ms: float | None = None   # first thinking token (adaptive thinking)
    first_tool_ms: float | None = None       # model starts emitting a tool call
    ttft_ms: float | None = None             # first token of the visible answer
    total_ms: float = 0.0
    model_requests: int = 0
    tool_calls: list[str] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    retries: int = 0                          # tool retries: SQL errors, bad args
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0                    # includes thinking tokens
    answer: str = ""
    error: str | None = None


async def stream_turn(
    agent: Agent,
    prompt: str,
    *,
    deps: Any,
    model: str | Model | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    usage_limits: UsageLimits | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Yield ("status"|"thinking"|"sql"|"delta"|"done", payload). "done" carries
    `metrics` (TurnMetrics) and `messages` (full history for the next turn)."""
    m = TurnMetrics()
    t0 = time.perf_counter()
    ms = lambda: round(1000 * (time.perf_counter() - t0), 1)  # noqa: E731
    chunks: list[str] = []
    messages: list[ModelMessage] = list(message_history or [])
    try:
        async with agent.run_stream_events(
            prompt, deps=deps, model=model, message_history=message_history, usage_limits=usage_limits
        ) as events:
            async for ev in events:
                if isinstance(ev, PartStartEvent) and isinstance(ev.part, ThinkingPart):
                    if m.first_thinking_ms is None:
                        m.first_thinking_ms = ms()
                        yield "thinking", {"label": "Thinking…"}
                elif isinstance(ev, PartStartEvent) and isinstance(ev.part, ToolCallPart):
                    # Fires as the model starts a tool call, before its args finish streaming.
                    m.first_tool_ms = m.first_tool_ms or ms()
                    yield "status", {"label": TOOL_LABELS.get(ev.part.tool_name, ev.part.tool_name) + "…"}
                elif isinstance(ev, FunctionToolCallEvent) and ev.part.tool_name == "run_sql":
                    yield "sql", {"sql": ev.part.args_as_dict().get("sql", "")}
                elif isinstance(ev, FunctionToolResultEvent):
                    yield "status", {"label": "Writing answer…"}
                elif isinstance(ev, PartStartEvent) and isinstance(ev.part, TextPart) and ev.part.content:
                    m.ttft_ms = m.ttft_ms or ms()
                    chunks.append(ev.part.content)
                    yield "delta", {"text": ev.part.content}
                elif isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                    m.ttft_ms = m.ttft_ms or ms()
                    chunks.append(ev.delta.content_delta)
                    yield "delta", {"text": ev.delta.content_delta}
                elif isinstance(ev, AgentRunResultEvent):
                    res = ev.result
                    m.answer = res.output
                    messages = res.all_messages()
                    u = res.usage
                    m.model_requests, m.input_tokens, m.output_tokens = u.requests, u.input_tokens, u.output_tokens
                    m.cache_read_tokens, m.cache_write_tokens = u.cache_read_tokens, u.cache_write_tokens
                    for msg in res.new_messages():
                        for p in msg.parts:
                            if isinstance(p, ToolCallPart):
                                m.tool_calls.append(p.tool_name)
                                if p.tool_name == "run_sql":
                                    m.sql.append(p.args_as_dict().get("sql", ""))
                            elif isinstance(p, RetryPromptPart):
                                m.retries += 1
    except Exception as e:  # usage limit hit, API error, ...
        m.error = f"{type(e).__name__}: {e}"[:300]
        m.answer = m.answer or "".join(chunks)
    m.total_ms = ms()
    yield "done", {"metrics": m, "messages": messages}


async def run_turn(agent: Agent, prompt: str, **kwargs) -> tuple[TurnMetrics, list[ModelMessage]]:
    """Non-streaming convenience wrapper (still streams under the hood, for TTFT)."""
    async for kind, payload in stream_turn(agent, prompt, **kwargs):
        if kind == "done":
            return payload["metrics"], payload["messages"]
    raise RuntimeError("stream ended without a result")
