"""Offline tests for the store, the typed agent, and model/effort wiring.  Run: pytest -q"""
import asyncio
import json

import httpx2
import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.providers.anthropic import AnthropicProvider

import config
from agent import Deps, build_agent
from runner import run_turn
from store import EventFilter


def test_counts_match_raw_tsv(store):
    with open("mod_channel_15k.tsv") as f:
        next(f)
        bans = sum(1 for line in f if line.split("\t")[7] == "ban")
    assert store.aggregate(EventFilter(actions=["ban"]), "mod").total_matches == bans


def test_partial_author_is_reported(store):
    r = store.find_events(EventFilter(author="grumpy_r1"), limit=1)
    assert r.authors_matched and all("grumpy_r1" in a for a in r.authors_matched)


def test_unknown_author_returns_zero(store):
    assert store.find_events(EventFilter(author="nobody_here_xyz")).total_matches == 0


def test_date_range_is_inclusive(store):
    r = store.find_events(EventFilter(since="2026-10-17", until="2026-10-17"), limit=50)
    assert r.total_matches > 0 and all(row.time.startswith("2026-10-17") for row in r.rows)


def test_typed_agent_tool_roundtrip(store):
    """The typed agent calls aggregate_events with typed filters and gets real numbers back."""
    seen = {}

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        returns = [p for p in getattr(messages[-1], "parts", []) if isinstance(p, ToolReturnPart)]
        if returns:
            seen["result"] = returns[0].model_response_object()
            yield "Mara issued the most bans: 22."
            return
        yield {0: DeltaToolCall(name="aggregate_events",
                                json_args=json.dumps({"group_by": "mod", "filters": {"actions": ["ban"]}}),
                                tool_call_id="t1")}

    agent = build_agent("typed", store)
    with agent.override(model=FunctionModel(stream_function=stream)):
        m, _ = asyncio.run(run_turn(agent, "who banned the most?", deps=Deps(store),
                                    usage_limits=config.MODE_LIMITS["typed"]))
    assert m.error is None and m.tool_calls == ["aggregate_events"] and m.ttft_ms is not None
    assert seen["result"]["groups"][0] == {"key": "Mara", "count": 22}


def test_model_aliases_and_effort_parsing():
    assert config.resolve_model("opus") == "anthropic:claude-opus-5-5"
    assert config.resolve_model("openai:gpt-5.6") == "openai:gpt-5.6"
    with pytest.raises(ValueError):
        config.resolve_model("gpt")
    assert config.parse_model_spec("haiku@off", "medium") == ("haiku@off", "anthropic:claude-haiku-5-5", False)
    assert config.parse_model_spec("sonnet", "medium")[2] == "medium"


def _captured_request(store, model_name: str, effort: str) -> dict:
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}})

    model = AnthropicModel(model_name, provider=AnthropicProvider(
        api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))))
    agent = build_agent("sql", store, effort=effort)
    with models.override_allow_model_requests(True):  # the transport is mocked: nothing leaves the process
        asyncio.run(run_turn(agent, "How many bans?", deps=Deps(store), model=model))
    return captured["body"]


@pytest.mark.parametrize("alias", ["haiku", "sonnet", "opus"])
def test_medium_effort_wire_request(store, alias):
    """Pins what the Thinking/Caching capabilities actually send to each Claude 5.5 model."""
    body = _captured_request(store, config.resolve_model(alias).split(":", 1)[1], "medium")
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    assert body["max_tokens"] == config.MAX_TOKENS
    assert "temperature" not in body
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_haiku_can_disable_thinking(store):
    body = _captured_request(store, "claude-haiku-5-5", "off")
    assert body["thinking"] == {"type": "disabled"} and "effort" not in body.get("output_config", {})
