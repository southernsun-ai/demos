"""Offline tests for the SQL modes, the read-only guard, and the eval grader.  Run: pytest -q"""
import asyncio

import pytest

import config
from agent import Deps, build_agent
from eval_cases import build_cases, build_dataset, grade, lead
from fake_model import offline_model_for
from runner import run_turn


@pytest.mark.parametrize("sql", [
    "DROP TABLE events",
    "DELETE FROM events",
    "UPDATE events SET reason = 'x'",
    "INSERT INTO events SELECT * FROM events LIMIT 1",
    "SELECT 1; DROP TABLE events",
    "COPY events TO 'out.csv'",
    "ATTACH 'other.db'",
    "SET enable_external_access = true",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_text('mod_channel_15k.tsv')",
    "CREATE TABLE x AS SELECT 1",
])
def test_guard_rejects_writes_and_file_access(store, sql):
    with pytest.raises(ValueError):
        store.run_sql(sql)
    assert store.run_sql("SELECT count(*) FROM events").splitlines()[1] == "15000"  # data untouched


def test_sql_errors_are_model_readable(store):
    with pytest.raises(ValueError, match="nope"):
        store.run_sql("SELECT nope FROM events")


def test_sql_output_is_capped(store):
    out = store.run_sql("SELECT * FROM events")
    assert "more rows exist" in out and len(out) < 7000


def test_reference_sql_matches_python_ground_truth(store):
    for case in build_cases():
        assert grade(case, store.run_sql(case.reference_sql), strict=False), case.id


def test_discovery_lists_all_enum_values(store):
    d = store.describe_table("events")
    row = next(line for line in d.splitlines() if line.startswith("action_taken"))
    for v in ["ban", "delete", "kick", "mute", "none", "timeout", "warn"]:
        assert repr(v) in row


def test_grader_numbers_are_whole_tokens():
    case = next(c for c in build_cases() if c.id == "mara_oct_bans")  # expects "6"
    assert grade(case, "Mara issued 6 bans in October 2026.")
    assert not grade(case, "Mara issued 16 bans on 2026-10-06.")   # 6 only inside other tokens
    assert grade(next(c for c in build_cases() if c.id == "edited_count"), "1,089 messages were edited.")


def test_grader_strict_ignores_numbers_in_supporting_detail():
    case = next(c for c in build_cases() if c.id == "warn_then_ban")  # expects 15
    ans = "12 users were warned and later banned.\n\n| mod | bans |\n| Doug | 15 |"
    assert lead(ans).startswith("12") and not grade(case, ans) and grade(case, ans, strict=False)


@pytest.mark.parametrize("mode", ["sql", "discover"])
def test_sql_mode_roundtrip(store, mode):
    cases = build_cases()
    case = next(c for c in cases if c.id == "peak_hour")
    agent = build_agent(mode, store)
    with agent.override(model=offline_model_for(mode, cases)):
        m, _ = asyncio.run(run_turn(agent, case.question, deps=Deps(store), usage_limits=config.MODE_LIMITS[mode]))
    assert m.error is None and grade(case, m.answer)
    assert m.tool_calls == (["list_tables", "describe_table", "run_sql"] if mode == "discover" else ["run_sql"])


def test_sql_error_becomes_retry(store):
    """A bad query is fed back to the model as a retry, and the model can recover."""
    from pydantic_ai.messages import RetryPromptPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    async def stream(messages, info):
        parts = [p for m in messages for p in getattr(m, "parts", [])]
        retries = [p for p in parts if isinstance(p, RetryPromptPart)]
        n_returns = sum(1 for p in parts if type(p).__name__ == "ToolReturnPart")
        if n_returns:
            yield "There are 69 bans."
        elif not retries:
            yield {0: DeltaToolCall(name="run_sql", json_args='{"sql": "SELECT count(*) FROM evnts"}', tool_call_id="a")}
        else:
            yield {0: DeltaToolCall(name="run_sql", json_args='{"sql": "SELECT count(*) FROM events WHERE action_taken=\'ban\'"}', tool_call_id="b")}

    agent = build_agent("sql", store)
    with agent.override(model=FunctionModel(stream_function=stream)):
        m, _ = asyncio.run(run_turn(agent, "How many bans?", deps=Deps(store)))
    assert m.retries == 1 and m.answer == "There are 69 bans."


def test_dataset_and_evaluator(store):
    ds = build_dataset()
    assert ds.name and len(ds.cases) == 16
    agent = build_agent("sql", store)
    cases = build_cases()

    async def task(q: str) -> str:
        case = next(c for c in cases if c.question == q)
        with agent.override(model=offline_model_for("sql", cases)):
            m, _ = await run_turn(agent, case.question, deps=Deps(store))
        return m.answer

    report = asyncio.run(ds.evaluate(task, progress=False))
    assert all(c.assertions["correct"].value for c in report.cases)
