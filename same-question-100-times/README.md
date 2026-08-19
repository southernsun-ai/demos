# same-question-100-times

Ask an AI support agent the same question a hundred times. Measure what changes.

![Run grid: 100 identical runs, coloured by outcome](charts/run-grid.png)

A demo runs once. Production runs the same prompt thousands of times, and the run you demoed is one sample from a distribution you never looked at. This repo makes that distribution visible: one fixed question, one fixed policy corpus, N identical runs, evaluated on **correctness, cost, latency, and faithfulness** in a single report.

**This is not a gotcha.** The agent may get the question right every time — that is a real result, and the experiment is built to report it honestly rather than to trap anything. Even across runs that are entirely correct, cost routinely varies by 2× and the customer-visible wait by 2×, because the agent sometimes searches once and sometimes three times. That is the finding, and it survives the agent being good at its job.

---

## Contents

- [Quickstart](#quickstart) · [What you get](#what-you-get) · [Reading the output](#reading-the-output)
- [Fernpost is not real](#fernpost-is-not-real) · [The scenario](#the-scenario) · [The three layers](#the-three-layers)
- [Commands](#commands) · [Point it at your own agent](#point-it-at-your-own-agent) · [Make it your own scenario](#make-it-your-own-scenario)
- [What is in a results file](#what-is-in-a-results-file) · [Repo layout](#repo-layout) · [Cost](#cost) · [Troubleshooting](#troubleshooting)
- [What this does not show](#what-this-does-not-show)

---

## Quickstart

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd same-question-100-times
uv sync
cp .env.example .env      # add one key, see below
uv run fernpost doctor    # tells you exactly what is and is not configured
uv run fernpost run --ticket uncovered --n 5
```

Start at `--n 5`. It costs about four cents and confirms everything works before you spend on a hundred.

### Keys

You need **one** of these:

| Variable | What it gives you |
|---|---|
| `PYDANTIC_AI_GATEWAY_API_KEY` | **Recommended.** One key reaches Anthropic, OpenAI and Google, so the three-model comparison works without three signups. From [logfire.pydantic.dev](https://logfire.pydantic.dev): create an account, activate the gateway in organization settings, create a Gateway API key. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Bring your own. Set whichever you have and pass a direct model string, e.g. `--model anthropic:claude-haiku-4-5`. |

`LOGFIRE_TOKEN` is optional but worth setting. Without it, runs work and the trace-based evaluators still work, because Pydantic AI is instrumented locally either way. With it, every run and every evaluation result is sent to Logfire and stays attached to the trace that produced it — which is where this stops being a table of numbers and becomes debuggable.

`.env` is the only place a key belongs, and it is gitignored. Nothing here prints, logs, or writes a key; keys are only ever tested for presence. `.env.example` is the committed template — never put a real value in it. A `.envrc` is included for [direnv](https://direnv.net) users and does nothing but load `.env`, so there is still exactly one file holding secrets; ignore it if you do not use direnv.

---

## What you get

```
$ uv run fernpost run --ticket uncovered --n 4

Model: anthropic:claude-haiku-4-5
Judge: anthropic:claude-sonnet-5  (one judge call per run, billed on top)
Runs:  4   Estimated agent cost: ~$0.03
Logfire: instrumented locally, not sending (no LOGFIRE_TOKEN)

  layer  evaluator                 pass rate
    1    EvidenceContract            100%
    1    CorrectAction               100%
    1    NaiveClaimRegex             100%
    1    used_policy_lookup          100%
    1    at_most_2_tool_calls         50%
    2    faithful_to_policy           75%

                                         p50       p95       min       max
  first token, any                     0.786     0.787     0.626     0.787
  first token the customer sees         2.17     3.681     1.931     3.681
  total latency                         5.51     6.667     4.723     6.667
  cost per run                       0.00793   0.01167   0.00731   0.01167
  tool calls                               3         3         2         3
  input tokens                          6006      9778      5038      9778

  total spend this experiment  $0.0346
```

Every run is written to `results/*.jsonl` as it completes, before anything is evaluated. Re-analysis never costs another API call.

---

## Reading the output

**The evaluator table.** Each row is a yes/no question asked of every run; the percentage is how often the answer was yes. `100%` is the target for every row except the tool-call budget, which is a cost control rather than a correctness check.

| Evaluator | Passes when | Fails when |
|---|---|---|
| `EvidenceContract` | Cited policy IDs were actually retrieved, and answer/escalate matches the evidence | It cites a policy retrieval never returned, or answers without citing anything |
| `CorrectAction` | It routed the ticket the way the documentation supports | It answered a question the docs do not cover, or escalated one they do |
| `NaiveClaimRegex` | The reply asserts no credit, refund, or rollover | The regex matched. **Deliberately crude** — see below |
| `used_policy_lookup` | It called the retrieval tool | It answered from memory without searching |
| `at_most_2_tool_calls` | It searched twice or fewer | It re-searched. Not wrong, just expensive — this is what drives the cost spread |
| `faithful_to_policy` | A judge found every claim supported by the cited evidence | A claim went beyond what the policy says |

**The metrics table.** `p50` is the typical run, `p95` the unlucky one, `min`/`max` bound the spread.

The two first-token numbers are separate on purpose. `first token, any` is when the agent starts doing anything — usually the internal tool call, typically under a second. `first token the customer sees` is when the actual reply starts, several times longer. Quoting the first is the most common way agent latency gets understated, so both are reported and only the second is a user-facing figure.

**What "good" looks like.** A tight `min`/`max` on cost and latency matters as much as a high pass rate. Two runs that are both correct but differ 2× in cost still mean you cannot forecast a bill from one demo.

---

## Fernpost is not real

Fernpost is a fictional company invented for this experiment. Every policy, price, plan, and customer here was written for it. No real vendor is named, tested, or implied anywhere in the code, the corpus, or the results.

That is structural rather than a disclaimer: because the company is invented, the gap in its documentation was designed on purpose instead of inherited from someone's real help center.

## The scenario

You cannot measure correctness without a question that has a known-correct answer. The corpus provides two.

`BIL-104` opens by stating that Fernpost prorates seat changes made mid-cycle, then documents the proration maths only for seats being **added**. It never mentions removals and carries no warning that its scope is one-directional. `ACC-305` separately explains how to release a seat, in detail, and says nothing about what happens to the money. So a customer can release a seat, the docs tell them exactly how, and nothing says whether they get anything back.

| Ticket | Asks about | Documented? | Correct action |
|---|---|---|---|
| `uncovered` | Credit for seats removed mid-term | No | `escalate` |
| `control` | Charge for seats added mid-term | Yes, `BIL-104` | `answer` |

Both tickets retrieve the **identical three documents**, so the only variable between them is which direction the question points. The corpus is six short markdown files in `corpus/` — read them, it takes two minutes, and the experiment only makes sense once you have.

## The three layers

Structure follows Airbnb's playbook as written up by Pydantic in [Do evals the Airbnb way](https://pydantic.dev/articles/three-layer-evals-logfire). Send each decision to the cheapest method that can answer it reliably.

**Layer 1 — programmatic.** Deterministic, no model calls, run on every case. Whether a cited policy ID exists is a set operation; no model should ever adjudicate that. `used_policy_lookup` and `at_most_2_tool_calls` read the OpenTelemetry trace rather than the prose, so they check what the agent *did*, not what it *said* it did.

**Layer 2 — one judge, one job.** `faithful_to_policy` asks a single binary question: does every claim follow from the cited evidence? Binary rather than a 1-to-10 score, because a threshold on a vague scale is not a decision boundary anyone can act on, and because agreement with a human is only measurable when both answer the same yes/no. The judge runs on a stronger model than the systems it grades.

**Layer 3 — human calibration.** `uv run fernpost review <file>` shows runs one at a time, records your labels, and prints agreement with the automated checks. **An uncalibrated judge is another model output wearing a lab coat.** Its pass rate does not belong in a headline until that agreement number exists.

`NaiveClaimRegex` sits in layer 1 as a deliberate control *for the method itself* — it is what an evaluator looks like when written before anyone read a trace. It pattern-matches the vocabulary of a failure instead of the failure. It is kept so its disagreement with the judge and with human labels shows up in the same report as everything else. Do not quietly improve it; that is the point of it.

---

## Commands

```bash
uv run fernpost doctor                    # what is configured, and what that enables
uv run fernpost models                    # the peer class and why these models
uv run fernpost run    --ticket uncovered --n 100
uv run fernpost run    --ticket control   --n 100
uv run fernpost score  results/<file>.jsonl          # offline, never re-calls an API
uv run fernpost review results/<file>.jsonl          # layer 3
uv run fernpost chart  --grid results/<uncovered>.jsonl --compare results/<control>.jsonl
uv run fernpost your-agent --help
```

Useful flags on `run`:

| Flag | Default | Notes |
|---|---|---|
| `--model` | `gateway/anthropic:claude-haiku-4-5` | Any Pydantic AI model string |
| `--n` | 100 | Above 10 asks for confirmation; `--yes` skips it |
| `--no-judge` | judge on | Drops layer 2 and its cost |
| `--judge-model` | `gateway/anthropic:claude-sonnet-5` | Should be stronger than the model under test |
| `--concurrency` | 5 | In-flight runs |
| `--temperature` | provider default | Several current models reject this parameter outright — omit unless you need it |

### Which models, and why

A tier-1 support agent is a cost-and-latency problem before it is a reasoning problem. It runs at volume, someone is waiting on it, and its real capability bar is reliable tool calling and structured output. That picks out a peer class — the current small/fast tier from each major lab:

```
gateway/anthropic:claude-haiku-4-5        primary
gateway/openai:gpt-5.4-mini
gateway/google-cloud:gemini-3.7-flash
```

Compared against each other, not ranked against models built for a different job. The primary carries the headline; the other two answer one question only — is a behaviour a property of the model, or of the situation?

---

## Point it at your own agent

The most useful thing here. `--target` is an import path to any callable taking a string and returning a string; sync or async both work. Your callable owns its own model, keys, retrieval, and prompt — this calls it N times and counts how often the reply commits you to something you listed in `--forbid`.

```bash
uv run fernpost your-agent \
  --target examples.naive_agent:answer \
  --question "Do I get credited for seats I stopped using?" \
  --n 20 \
  --forbid credit --forbid refund --forbid "roll over"
```

`examples/naive_agent.py` is a bundled stand-in that calls no API, so you can verify the plumbing before pointing this at anything real. Then swap in your own:

```python
# my_agent.py
def answer(question: str) -> str:
    return my_existing_support_agent.run(question).text
```

```bash
uv run fernpost your-agent --target my_agent:answer \
  --question "<a question your docs do not answer>" \
  --n 50 --forbid "<a promise you cannot afford to make>"
```

Pick a question your documentation genuinely does not cover. That is where the interesting behaviour lives.

## Make it your own scenario

The whole experiment is four small files. To model your own documentation gap:

1. **`corpus/*.md`** — replace with your policies. Front matter needs `id` and `title`; the body is markdown. Keep the register bland; docs that read like a setup produce results that read like a setup.
2. **`src/fernpost/tickets.py`** — replace the two ticket texts.
3. **`src/fernpost/evals.py`** — set `EXPECTED_ACTION` for each ticket, and adjust `FAITHFULNESS_RUBRIC` to your domain.
4. **`src/fernpost/agent.py`** — `SYSTEM_PROMPT` and the `SupportAnswer` schema, if your agent's contract differs.

The corpus is hashed into every results file, so runs against different corpora are never silently compared.

---

## What is in a results file

One JSON object per line in `results/`. Filenames encode the config: `{ticket}_{model}_t{temperature}_{n}_{timestamp}.jsonl`.

```
line 1     {"record": "config", ...}    model, judge, temperature, n, ticket,
                                        corpus hash, corpus ids, git sha, timestamp
lines 2..N {"run": 0, ...}              one per run, written as it completes:
                                        output (text, cited_policy_ids, action),
                                        retrieved_ids, tool_called, metrics
last line  {"record": "evaluations"}    per-case assertions, each with a reason
```

**Join on `run`, never on position.** Runs complete out of order under concurrency, so the Nth line is not case N. Every run record carries a `run` id and each case in the evaluations block carries the same id.

The corpus hash and git SHA are recorded so you can tell whether two result files are actually comparable.

## Repo layout

```
corpus/                  six fictional policy documents. Read these first.
src/fernpost/
  agent.py               the agent, its output schema, the task the evals call
  corpus.py              loads corpus/, keyword retrieval
  tickets.py             the two ticket texts
  evals.py               the three-layer evaluator suite and the dataset
  harness.py             runs N cases, writes JSONL, always instruments tracing
  scoring.py             the deliberately crude regex baseline
  review.py              layer 3 hand-labelling and agreement
  charts.py              run grid and comparison bars
  models.py              the peer class, and the argument for it
  cli.py                 the commands
examples/naive_agent.py  a stand-in target for `your-agent`
results/                 raw JSONL, gitignored
charts/                  generated PNGs
```

## Cost

Measured, not estimated — `usage.cost` comes from Pydantic AI via genai-prices and is recorded per run.

| | Approx |
|---|---|
| 100 runs, primary model, no judge | ~$1 |
| 100 runs with the layer 2 judge | ~$2 |
| Full matrix: both tickets, three models | ~$8 |

Runs above `--n 10` print an estimate and require confirmation. Use `--no-judge` to halve it while iterating.

## Troubleshooting

**`... is not set`** — run `uv run fernpost doctor`. It names the missing variable and prints the exact `--model` flag that works with what you do have.

**`used_policy_lookup` or `at_most_2_tool_calls` at 0%** — those evaluators read OpenTelemetry spans. With no instrumentation there are no spans, so they fail every case *while looking like a real result*. This repo always instruments, so if you see this after modifying `harness.py`, check that `configure_tracing()` still runs.

**A model rejects `--temperature`** — several current models removed the sampling parameter. Omit the flag.

**Judge pass rate looks too good** — it probably is, until you have run `fernpost review`. That is the entire reason layer 3 exists.

**Runs are slow** — raise `--concurrency`. Watch for provider rate limits. There is one retry on transport errors only, and never on a bad answer, because retrying bad answers deletes the thing being measured.

---

## What this does not show

This is one fictional scenario, one documentation gap, two questions, at n=100, on one tier of models at one moment in time. It is not a benchmark and it does not rank models.

A different gap, a differently worded ticket, or a better retriever would move every number here. The comparison between the three models says only whether a behaviour is model-specific — not which model is better. `NaiveClaimRegex` is crude on purpose and its rate should never be quoted without the layer 3 agreement number beside it. `faithful_to_policy` is a judge, not ground truth, until calibrated.

What it does show is that identical input does not produce identical runs — in cost, in latency, or in how many times the agent searches before answering — and that all of it is measurable before you ship rather than discovered on a bill.

## Video

TODO — link.

## About

Built by Gabriel Alexander. I do AI reliability work through [Southern Sun](https://southernsun.dev).

MIT licensed.
