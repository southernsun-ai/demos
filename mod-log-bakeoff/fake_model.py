"""Offline stand-in for the LLM, used to benchmark everything except the model itself.

It mimics the call pattern a real model produces: zero tool calls for overview questions,
or one tool call followed by a streamed answer. Model inference time is excluded, so the
numbers it yields are pure framework + tool + streaming overhead.
"""
from __future__ import annotations

import json

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel


def _last_user_text(messages: list[ModelMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if getattr(p, "part_kind", "") == "user-prompt":
                    return str(p.content).lower()
    return ""


async def _stream(messages: list[ModelMessage], info: AgentInfo):
    last = messages[-1]
    returns = [p for p in getattr(last, "parts", []) if isinstance(p, ToolReturnPart)]
    if returns:
        data = returns[0].model_response_object()
        text = f"Answer based on the tool result: {json.dumps(data)[:300]}"
        for i in range(0, len(text), 12):
            yield text[i:i + 12]
        return

    q = _last_user_text(messages)
    if "overview" in q or "summary" in q:
        for word in "There are 15,000 rows covering 2026-08-01 to 2026-10-17.".split(" "):
            yield word + " "
    elif "show" in q:
        yield {0: DeltaToolCall(name="find_events", json_args=json.dumps({"filters": {"actions": ["kick"]}, "limit": 5}), tool_call_id="c1")}
    else:
        yield {0: DeltaToolCall(name="aggregate_events", json_args=json.dumps({"group_by": "mod", "filters": {"actions": ["ban"]}}), tool_call_id="c1")}


fake_model = FunctionModel(stream_function=_stream, model_name="fake-offline")


def offline_model_for(mode: str, cases) -> FunctionModel:
    """Scripted models for the eval smoke test (tests the harness, not an LLM).

    sql / discover: an 'oracle' that walks the real tool sequence for that mode
    (discover: list_tables -> describe_table -> run_sql) using each case's reference
    SQL, then echoes the result. typed: the generic fake above.
    """
    if mode == "typed":
        return fake_model
    by_question = {c.question.lower(): c for c in cases}

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        case = by_question.get(_last_user_text(messages))
        calls_so_far = sum(1 for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart))
        last_returns = [p for p in getattr(messages[-1], "parts", []) if isinstance(p, ToolReturnPart)]
        plan = ["list_tables", "describe_table", "run_sql"] if mode == "discover" else ["run_sql"]
        if case is None:
            yield "OK"
            return
        if calls_so_far < len(plan):
            name = plan[calls_so_far]
            args = {"run_sql": {"sql": case.reference_sql}, "describe_table": {"table": "events"}}.get(name, {})
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f"c{calls_so_far}")}
            return
        text = f"Answer: {last_returns[0].content}"
        for i in range(0, len(text), 16):
            yield text[i:i + 16]

    return FunctionModel(stream_function=stream, model_name=f"oracle-{mode}")
