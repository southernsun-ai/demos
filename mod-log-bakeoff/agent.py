"""Agents for the three data-access modes, built the PydanticAI v2 way.

- Agents are model-agnostic: built once per mode (and effort), and the model is chosen
  per run with `run_stream_events(..., model=...)`. Swapping Haiku / Sonnet / Opus is a
  string change (see config.py), with no rebuilding.
- Tools live in `FunctionToolset`s and are composed per mode with `toolsets=[...]`.
- Thinking effort and prompt caching are configured with the `Thinking` and `Caching`
  capabilities, which map to the right provider settings (for Claude 5.5: adaptive
  thinking + `effort`, and automatic prompt caching).

Latency design (unchanged): the static prefix is cacheable, typed tools aggregate on the
server in one call, tool outputs are capped, and everything streams.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import anyio
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Caching, Thinking
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.toolsets import FunctionToolset

from config import MAX_TOKENS, Mode, parse_effort, resolve_model
from store import AggregateResult, EventFilter, EventsResult, GroupBy, ModLogStore


@dataclass
class Deps:
    store: ModLogStore


# ---------------------------------------------------------------------------
# Toolsets
# ---------------------------------------------------------------------------

typed_tools = FunctionToolset[Deps](max_retries=1, id="typed")


@typed_tools.tool
async def find_events(
    ctx: RunContext[Deps],
    filters: EventFilter,
    limit: int = 15,
    newest_first: bool = True,
) -> EventsResult:
    """Return matching mod-log rows (newest first by default) plus the total match count.

    Use for examples, quotes, and a user's or moderator's recent history.
    `limit` is at most 50; the total count is always exact.
    """
    return ctx.deps.store.find_events(filters, limit=min(max(limit, 1), 50), newest_first=newest_first)


@typed_tools.tool
async def aggregate_events(
    ctx: RunContext[Deps],
    group_by: GroupBy,
    filters: EventFilter | None = None,
    top_n: int = 15,
) -> AggregateResult:
    """Count matching events grouped by one dimension.

    Use for "how many", "who did the most", "top offenders", "breakdown by channel",
    and trends (group_by day / week / month). Returns exact counts.
    """
    return ctx.deps.store.aggregate(filters or EventFilter(), group_by, top_n=min(max(top_n, 1), 100))


sql_tools = FunctionToolset[Deps](max_retries=3, id="sql")


@sql_tools.tool
async def run_sql(ctx: RunContext[Deps], sql: str) -> str:
    """Run ONE read-only DuckDB SELECT against the mod-log database and return the result table."""
    try:
        # Model-written SQL has unknown cost, so keep it off the event loop.
        return await anyio.to_thread.run_sync(ctx.deps.store.run_sql, sql)
    except ValueError as e:
        raise ModelRetry(str(e)) from None


discovery_tools = FunctionToolset[Deps](max_retries=2, id="discovery")


@discovery_tools.tool
async def list_tables(ctx: RunContext[Deps]) -> str:
    """List the tables in the database with approximate row counts."""
    return ctx.deps.store.list_tables()


@discovery_tools.tool
async def describe_table(ctx: RunContext[Deps], table: str) -> str:
    """Describe a table's columns: type, distinct count, null count, and example values
    (all values for low-cardinality columns)."""
    try:
        return await anyio.to_thread.run_sync(ctx.deps.store.describe_table, table)
    except ValueError as e:
        raise ModelRetry(str(e)) from None


# ---------------------------------------------------------------------------
# Instructions (static per snapshot, so they stay prompt-cacheable)
# ---------------------------------------------------------------------------

ROLE = """\
You are the admin assistant for a Discord server. Admins ask you about what moderators
have done. You answer ONLY from the mod-log data and from tool results."""


def common_rules(store: ModLogStore) -> str:
    """Rules shared by every mode, so the eval compares data access, not prompting."""
    return f"""\
- Treat "today" as {store.as_of:%Y-%m-%d}. Resolve "last week" and similar phrases against that date,
  and state the date range you used.
- "Mod action" means action_taken is one of warn, mute, timeout, kick, delete, ban.
- Never invent rows, users, counts, or reasons. If a query returns 0 matches, say so.
- Deleted messages show "[content removed]"; the original text is not stored.
- If you need several independent results, issue the tool calls in parallel in a single turn.

## Style
Lead with the answer in one sentence. Then give supporting detail as a short list or a small
table (at most 10 rows). Write numbers as digits. Quote message content exactly. Use UTC times. No preamble."""


SQL_RULES = """\
- The database is DuckDB (use DuckDB SQL: date_trunc, strftime, ILIKE, QUALIFY, etc.).
- Do the computation IN SQL (GROUP BY, COUNT, window functions). Do not pull raw rows to count them yourself.
- Results are capped at 50 rows; always aggregate or use LIMIT.
- Prefer one well-formed query that returns exactly what is needed over several exploratory ones.
- Match names case-insensitively (lower(x) = lower('...') or ILIKE) unless the value is known exactly.
- If a query errors, read the error, fix the SQL, and retry."""

COLUMN_NOTES = """\
-- One row per message or event in the server's channels.
-- ts / edited_at are UTC. content has real newlines. Null = not applicable.
-- action_taken = 'none' for rows with no mod action; then reason, mod_id, mod_username,
--   target_user_id are NULL. author_* is the user who posted (and was actioned)."""


def typed_instructions(store: ModLogStore, include_summary: bool) -> str:
    summary = (
        f"\n## Snapshot (as of {store.as_of:%Y-%m-%d})\n{store.snapshot_summary()}\n"
        "- If the snapshot above already answers the question exactly, answer directly. Do not call a tool.\n"
        if include_summary else ""
    )
    return f"""\
{ROLE}
{summary}
## Rules
- For counts, rankings, "who did the most", or trends over time, call `aggregate_events` (one call).
- For "show me", "what did X say", or examples, call `find_events`.
- If `authors_matched` is present, the author name was a partial match: tell the admin which users were included.
{common_rules(store)}
"""


def sql_instructions(store: ModLogStore) -> str:
    return f"""\
{ROLE}

## Database schema
Table `events` (~15,000 rows):
```sql
{COLUMN_NOTES}
{store.schema_ddl()}
```

## Rules
- Answer by querying with `run_sql`. You may discover values (e.g. SELECT DISTINCT ...) if needed.
{SQL_RULES}
{common_rules(store)}
"""


def discover_instructions(store: ModLogStore) -> str:
    return f"""\
{ROLE}

## Data access
You have a read-only DuckDB database containing the mod log. You have NOT been told its schema.
Discover it with `list_tables` and `describe_table` before writing queries, then answer with `run_sql`.
Within one conversation, reuse the schema you already discovered instead of describing again.

## Rules
{SQL_RULES}
{common_rules(store)}
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _capabilities(effort: ThinkingLevel) -> list:
    caps = [
        Thinking(effort=effort),
        # Instructions + tool schemas are static per snapshot: 1h retention. Conversation
        # history is cached too, so follow-up turns only prefill the new message.
        Caching("1h"),
    ]
    if os.getenv("LOGFIRE_TOKEN") or os.getenv("MOD_CHAT_INSTRUMENT") == "1":
        import logfire  # optional dependency

        logfire.configure(send_to_logfire="if-token-present", console=False)
        logfire.instrument_pydantic_ai()
    return caps


def build_agent(
    mode: Mode,
    store: ModLogStore,
    *,
    effort: ThinkingLevel | str | None = None,
    include_summary: bool = True,
    model: str | None = None,
) -> Agent[Deps, str]:
    """Build the agent for one data-access mode.

    `model` is only the default; pass `model=` to each run to switch models per request.
    """
    effort = parse_effort(effort) if not isinstance(effort, bool) else effort
    instructions, toolsets = {
        "typed": (typed_instructions(store, include_summary), [typed_tools]),
        "sql": (sql_instructions(store), [sql_tools]),
        "discover": (discover_instructions(store), [discovery_tools, sql_tools]),
    }[mode]
    return Agent(
        resolve_model(model) if model is None or isinstance(model, str) else model,
        name=mode,
        deps_type=Deps,
        output_type=str,
        instructions=instructions,
        toolsets=toolsets,
        capabilities=_capabilities(effort),
        model_settings=ModelSettings(max_tokens=MAX_TOKENS, parallel_tool_calls=True),
        defer_model_check=True,   # no API key needed to build; checked on first run
    )
