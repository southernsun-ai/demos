# Instructions for coding agents

This project is a PydanticAI v2 admin chat over a Discord mod-log snapshot (`mod_channel_15k.tsv`),
with three data-access modes (`typed`, `sql`, `discover`) and an eval comparing models × modes.
Start with `llms.txt` (project map, data, eval semantics, and the spec for planned work). See `README.md` for the design. This file is the runbook.

## Setup (once)

```bash
make setup          # creates .venv and installs requirements.txt (Python 3.11+)
cp .env.example .env   # then put the user's ANTHROPIC_API_KEY in .env
```

If `make` isn't available, run the commands from the Makefile by hand. Each target is one or two lines.
`make` loads `.env` automatically. Never print the API key, and never commit `.env`.

## Tasks

| User asks for | Run | Needs API key | Approx. cost / time |
|---|---|---|---|
| "check it works" | `make test` then `make smoke` | no | free, <1 min |
| "run the chat" | `make serve`, then open http://localhost:8000 | yes | per question |
| "quick model comparison" | `make eval-quick` (1 repeat) | yes | 144 agent runs |
| "full model comparison" | `make eval` (3 repeats) | yes | 432 agent runs |
| "compare with Haiku thinking off" | `make eval MODELS=haiku,sonnet,opus,haiku@off` | yes | +33% runs |
| a subset | `make eval MODELS=sonnet MODES=sql,discover REPEATS=1` | yes | smaller |
| latency benchmark of a running server | `make bench` | yes (server) | 60 questions |

The variables `MODELS`, `MODES`, `REPEATS`, `EFFORT` and `CASES` can be passed to `make eval`.
Model aliases are `haiku`, `sonnet` and `opus` (Claude 5.5). Any `provider:model` string also works,
and `@effort` can be appended per model (`minimal|low|medium|high|xhigh|on|off`).

## Before running a paid eval

1. Run `make test` and `make smoke` first. Both must pass.
2. Tell the user how many agent runs the command will make (the table above), and confirm before running `make eval`.
3. Run `make eval-quick` before `make eval` unless the user asked for the full run.

## After an eval

- Results go to `results/<timestamp>/report.md` (summary) and `runs.jsonl` (every run).
- Report back to the user:
  - accuracy and median / p95 total time for each model × mode
  - the "Best combination per model" section
  - which questions failed, and why (the Failures section shows the answer and the last SQL)
- Accuracy is strict: the expected values must be in the answer's first paragraph. If strict and "Loose"
  differ a lot, the model is answering correctly but burying the answer. Say so.
- `first_thinking_ms` vs `ttft_ms` shows what thinking costs per turn.

## Rules

- Change models and effort in `config.py` or with env vars / CLI flags. Don't hard-code them elsewhere.
- Don't edit `eval_cases.py` ground truth or the grader to make a model pass. If you think a case is wrong,
  report it. `make test` cross-checks ground truth against independent SQL.
- `store.py`'s SQL guard (SELECT-only, external access off, config locked) must stay intact. `make test` checks it.
- Tests must never call a real model (`conftest.py` blocks it). Use `agent.override(model=FunctionModel(...))`.
- If an eval hits rate limits, keep `--concurrency 1` (the default). Re-run with fewer `REPEATS` rather than raising concurrency.
