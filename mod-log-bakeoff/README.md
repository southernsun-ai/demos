# Mod Log Admin Chat (PydanticAI v2)

Admins ask plain-English questions about moderator activity. A PydanticAI agent answers from a
TSV snapshot loaded into in-memory DuckDB and streams the reply over SSE. There are three
data-access modes to compare: typed tools, SQL with the schema given, and SQL with schema discovery.

```
browser ──POST /chat {mode, model, effort}──▶ FastAPI ──▶ Agent (per mode) ──▶ Claude (streamed)
   ▲                                                         │ toolsets
   └──── SSE: status / sql / delta / done (timings) ─────────┘ typed · sql · discovery ──▶ DuckDB (memory, locked)
```

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
uvicorn server:app --port 8000                 # open http://localhost:8000
python eval_modes.py                           # haiku, sonnet, opus × 3 modes @ medium
pytest -q                                      # offline; real model calls are blocked
```

## Changing the model or effort

Everything is set in `config.py`, or overridden by env vars and per-request fields:

| Where | Model | Effort |
|---|---|---|
| Defaults | `MOD_CHAT_MODEL=haiku` | `MOD_CHAT_EFFORT=medium` |
| Per chat request | `"model": "opus"` | `"effort": "high"` |
| Eval | `--models haiku,sonnet,opus,haiku@off` | `--effort medium` (or `@effort` per model) |

Aliases: `haiku`, `sonnet`, `opus` → `anthropic:claude-{haiku,sonnet,opus}-5-5`. Any PydanticAI model
string also works (`openai:gpt-5.6`, `google-gla:gemini-...`), because thinking and caching are configured
through provider-agnostic capabilities.

**What "medium" sends** (pinned by `test_medium_effort_wire_request`): `thinking: {type: adaptive}` and
`output_config: {effort: medium}` on all three models, with 1h cache markers on the system prompt and tools.
On Claude 5.5, adaptive is the only thinking mode. Sonnet and Opus 5.5 always think (`@off` is ignored).
Only Haiku 5.5 can turn thinking off (`haiku@off` sends `thinking: disabled`), which makes it a useful speed baseline.

## v2 practices used

- **Model-agnostic agents.** Agents are built once per mode (`defer_model_check=True`, so no key is needed to build).
  The model is chosen per run with `run_stream_events(..., model=...)`.
- **`FunctionToolset`s** (`typed`, `sql`, `discovery`), composed per mode with `toolsets=[...]`. Each toolset sets its own `max_retries`.
- **Capabilities** instead of provider-specific settings: `Thinking(effort)` and `Caching("1h")`. There are no
  `anthropic_*` keys and no `temperature` (5.5 models manage sampling themselves).
- **`instructions=`** (static per snapshot, so cacheable) instead of `system_prompt`.
- **`run_stream_events`** as an async context manager. `result.usage` is a property, and `AgentRunResultEvent` ends the stream.
- **`ModelRetry`** turns SQL errors into self-correction. `UsageLimits` sets hard ceilings per mode.
- **Testing:** `models.ALLOW_MODEL_REQUESTS = False` in `conftest.py`, `agent.override(model=FunctionModel(...))`,
  and `override_allow_model_requests(True)` only around a mocked HTTP transport.
- **Evals on `pydantic_evals`:** `Dataset(name=...)`, a custom `Evaluator`, and timing recorded with `increment_eval_metric`
  / `set_eval_attribute`. With Logfire configured, every eval run and agent span is traced.
- **Optional observability:** set `LOGFIRE_TOKEN` (or `MOD_CHAT_INSTRUMENT=1`) to call `logfire.instrument_pydantic_ai()`.

## Three data-access modes

| Mode | What the model gets | Toolsets | Typical round trips |
|---|---|---|---|
| `typed` | Snapshot summary + rules | `find_events`, `aggregate_events` | 1–2 |
| `sql` | Table DDL + column notes | `run_sql` | 2 (+1 per SQL error) |
| `discover` | Nothing about the data | `list_tables`, `describe_table`, `run_sql` | 3–4 (+1 per SQL error) |

Model-written SQL is limited to one SELECT statement, checked with DuckDB's own parser. File and network access are off,
and the configuration is locked so they can't be turned back on. Queries time out after 5 s, and output is capped at 50 rows / 6,000 characters.
`describe_table` lists every value for columns with 20 or fewer distinct values.

## Eval

```bash
python eval_modes.py --repeats 3                          # default matrix: 3 models × 3 modes @ medium
python eval_modes.py --models haiku,sonnet,opus,haiku@off
python eval_modes.py --models sonnet --modes sql,discover --effort high
python eval_modes.py --typed-no-summary                    # remove typed mode's head start
python eval_modes.py --offline --repeats 1                 # harness smoke test
```

- **16 questions.** 10 are within what the typed tools can express, and 6 fall outside it (edits, attachments, hour of day,
  COUNT DISTINCT, warn→ban sequences, edit latency). Results are reported for each group.
- **Ground truth** is computed in plain Python from the raw TSV and cross-checked against reference DuckDB queries.
- **Grading is deterministic.** Expected values must appear in the answer's lead paragraph (strict). A loose
  "anywhere" score is shown alongside. A run that errored counts as wrong.
- **Fairness.** Within a model, every mode uses the same effort, settings and usage ceiling. Each question gets a fresh conversation.
  Each (model, mode) prefix is cache-primed before timing starts. Concurrency defaults to 1 so rate limits don't distort latency.
- **Report** (`results/<ts>/report.md`): for each model × mode, accuracy, time to first *thinking* token, TTFT and
  total p50/p95, model and tool calls, SQL retries, and input / cache-read / output tokens (output includes thinking).
  It also gives the best mode per model, a per-question pass matrix, and every failure with its last SQL.

## Latency design

1. Overview questions are answered from the snapshot in the instructions, with zero tool calls (typed mode).
2. `aggregate_events` does group-by counts on the server, in one call.
3. Parallel tool calls are enabled.
4. The static prefix is cached for 1h, and conversation history is cached too. Nothing per-request goes in the instructions.
5. Tool outputs are capped. `max_tokens=4096` leaves room for thinking.
6. Everything streams: thinking → status (sent when a tool call *starts*) → the SQL → answer tokens.
7. Startup loads the data and primes the TLS connection and each mode's cache. Typed tools are async with no thread hop.

Offline overhead (fake model, real HTTP): about 7 ms to first token with no tool, and about 16 ms with one tool. Real TTFT is
dominated by the model. With thinking on, the gap between `first_thinking_ms` and `ttft_ms` is what thinking
costs per turn. Compare `haiku@medium` against `haiku@off` to see it.

## Files

- `config.py`: model aliases, default effort, per-mode limits. **Change models here.**
- `store.py`: DuckDB load and lock, typed queries, guarded `run_sql`, discovery helpers.
- `agent.py`: toolsets, instructions, and `build_agent(mode, store, effort=...)`.
- `runner.py`: one streamed turn with timings, shared by the server and the eval.
- `server.py`, `index.html`: SSE endpoint and chat UI, with mode and model pickers.
- `eval_cases.py`, `eval_modes.py`: the pydantic_evals dataset and evaluator, and the model × mode runner.
- `bench.py`: client-side latency benchmark against a running server.
- `fake_model.py`, `conftest.py`, `test_*.py`: offline models and tests.
