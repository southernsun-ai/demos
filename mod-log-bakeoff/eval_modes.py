"""Model x mode eval on pydantic_evals: who answers faster and more accurately?

  export ANTHROPIC_API_KEY=...
  python eval_modes.py                                  # haiku, sonnet, opus x typed, sql, discover @ medium
  python eval_modes.py --models haiku,sonnet,opus,haiku@off --repeats 3
  python eval_modes.py --models sonnet --modes sql,discover --effort high
  python eval_modes.py --models anthropic:claude-opus-5-5@low
  python eval_modes.py --typed-no-summary               # remove typed mode's head start
  python eval_modes.py --offline --repeats 1            # harness smoke test, no API key

`--models` takes aliases from config.py (haiku, sonnet, opus) or full 'provider:model'
strings, each optionally suffixed with @effort. Everything else defaults from config.py.
Writes results/<timestamp>/runs.jsonl and report.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from pydantic_ai.settings import ModelSettings
from pydantic_evals import increment_eval_metric, set_eval_attribute

import config
from agent import Deps, build_agent
from eval_cases import build_dataset
from runner import run_turn
from store import ModLogStore

# Same generous ceiling for every mode in the eval, so no mode is cut off early.
from pydantic_ai.usage import UsageLimits

EVAL_LIMITS = UsageLimits(request_limit=10, tool_calls_limit=14)
METRIC_FIELDS = ("first_thinking_ms", "first_tool_ms", "ttft_ms", "total_ms", "model_requests", "retries",
                 "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")


def make_task(agent, deps: Deps, model):
    async def task(question: str) -> str:
        m, _ = await run_turn(agent, question, deps=deps, model=model, usage_limits=EVAL_LIMITS)
        for k in METRIC_FIELDS:
            v = getattr(m, k)
            if v is not None:
                increment_eval_metric(k, v)
        increment_eval_metric("tool_calls", len(m.tool_calls))
        set_eval_attribute("tools", m.tool_calls)
        set_eval_attribute("sql", m.sql)
        if m.error:
            set_eval_attribute("error", m.error)
        return m.answer

    return task


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))] if xs else None


def fmt(x):
    return "–" if x is None else (f"{x:,.0f}" if abs(x) >= 10 else f"{x:.1f}")


def flatten(report, model_label: str, mode: str) -> list[dict]:
    rows = []
    for c in report.cases:
        rows.append({
            "model": model_label, "mode": mode, "case": c.source_case_name or c.name,
            "typed_capable": (c.metadata or {}).get("typed_capable"),
            "correct": bool(c.assertions.get("correct") and c.assertions["correct"].value),
            "correct_loose": bool(c.assertions.get("correct_loose") and c.assertions["correct_loose"].value),
            "answer": c.output, **{k: c.metrics.get(k) for k in METRIC_FIELDS + ("tool_calls",)},
            "tools": c.attributes.get("tools"), "sql": c.attributes.get("sql"), "error": c.attributes.get("error"),
        })
    for f in report.failures:  # task raised (shouldn't happen: run_turn catches), still counted
        rows.append({"model": model_label, "mode": mode, "case": f.source_case_name or f.name,
                     "typed_capable": (f.metadata or {}).get("typed_capable"), "correct": False,
                     "correct_loose": False, "answer": "", "error": f.error_message})
    return rows


def summary_table(rows: list[dict], combos: list[tuple[str, str]]) -> list[str]:
    out = ["| Model @ effort | Mode | Accuracy | Loose | Thinking p50 | TTFT p50 / p95 | Total p50 / p95 | "
           "Model calls | Tool calls | Retries | Input tok | Cache-read tok | Output tok |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, mode in combos:
        rs = [r for r in rows if r["model"] == model and r["mode"] == mode]
        if not rs:
            continue
        col = lambda k: [r.get(k) for r in rs]  # noqa: E731
        mean = lambda k: statistics.mean(v or 0 for v in col(k))  # noqa: E731
        n_ok = sum(r["correct"] for r in rs)
        out.append(
            f"| {model} | {mode} | {100 * n_ok / len(rs):.0f}% ({n_ok}/{len(rs)}) "
            f"| {100 * sum(r['correct_loose'] for r in rs) / len(rs):.0f}% "
            f"| {fmt(pct(col('first_thinking_ms'), 50))} ms "
            f"| {fmt(pct(col('ttft_ms'), 50))} / {fmt(pct(col('ttft_ms'), 95))} ms "
            f"| {fmt(pct(col('total_ms'), 50))} / {fmt(pct(col('total_ms'), 95))} ms "
            f"| {mean('model_requests'):.1f} | {mean('tool_calls'):.1f} | {mean('retries'):.1f} "
            f"| {mean('input_tokens'):,.0f} | {mean('cache_read_tokens'):,.0f} | {mean('output_tokens'):,.0f} |")
    return out


def render(rows, combos, case_ids, meta) -> str:
    out = ["# Model × mode eval\n",
           f"Repeats: {meta['repeats']} · questions: {len(case_ids)} · typed summary: "
           f"{'on' if meta['typed_summary'] else 'off'} · {meta['started']}\n",
           "Accuracy = expected values in the answer's lead paragraph. Thinking = time to first thinking "
           "token. TTFT = time to first visible answer token. Output tokens include thinking.\n"]
    if meta["offline"]:
        out.append("> **OFFLINE SMOKE TEST.** Scripted fake models, not an LLM. This checks the harness; the "
                   "numbers say nothing about real model behaviour.\n")
    out += ["## All questions\n", *summary_table(rows, combos)]
    out += ["\n## Questions the typed tools were designed for\n", *summary_table([r for r in rows if r["typed_capable"]], combos)]
    out += ["\n## Questions outside the typed tools' filters\n", *summary_table([r for r in rows if not r["typed_capable"]], combos)]

    out.append("\n## Best combination per model (accuracy first, then median total time)\n")
    for model in dict.fromkeys(m for m, _ in combos):
        scored = []
        for _, mode in [c for c in combos if c[0] == model]:
            rs = [r for r in rows if r["model"] == model and r["mode"] == mode]
            if rs:
                scored.append((-sum(r["correct"] for r in rs) / len(rs), pct([r.get("total_ms") for r in rs], 50) or 1e12, mode))
        if scored:
            acc, t, mode = min(scored)
            out.append(f"- **{model}**: `{mode}`: {-acc * 100:.0f}% accurate, {fmt(t)} ms median total")

    out.append("\n## Per question (pass rate)\n")
    out.append("| Question | " + " | ".join(f"{m}<br>{mo}" for m, mo in combos) + " |")
    out.append("|---|" + "---|" * len(combos))
    for cid in case_ids:
        cells = []
        for m, mo in combos:
            rs = [r for r in rows if r["model"] == m and r["mode"] == mo and r["case"] == cid]
            cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}" if rs else "–")
        out.append(f"| `{cid}` | " + " | ".join(cells) + " |")

    fails = [r for r in rows if not r["correct"]][:40]
    if fails:
        out.append("\n## Failures (first 40)\n")
        for r in fails:
            out.append(f"- **{r['model']} / {r['mode']} / {r['case']}**: "
                       + (f"error `{r['error']}`. " if r.get("error") else "")
                       + f"answer {str(r.get('answer'))[:200]!r}"
                       + (f" · last SQL `{r['sql'][-1][:180]}`" if r.get("sql") else ""))
    return "\n".join(out) + "\n"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="haiku,sonnet,opus", help="aliases or provider:model, optional @effort")
    ap.add_argument("--effort", default=config.DEFAULT_EFFORT, help="default effort for models without @effort")
    ap.add_argument("--modes", default=",".join(config.MODES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1, help=">1 is faster but adds rate-limit noise to latency")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--typed-no-summary", action="store_true")
    ap.add_argument("--offline", action="store_true", help="scripted fake models; tests the harness only")
    ap.add_argument("--show-pydantic-report", action="store_true", help="also print pydantic_evals' own tables")
    ap.add_argument("--tsv", default="mod_channel_15k.tsv")
    a = ap.parse_args()

    store = ModLogStore.load(a.tsv)
    deps = Deps(store=store)
    dataset = build_dataset(a.tsv, set(a.cases.split(",")) if a.cases else None)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    specs = [("offline@" + a.effort, None, config.parse_effort(a.effort))] if a.offline else \
        [config.parse_model_spec(s.strip(), a.effort) for s in a.models.split(",") if s.strip()]

    started = datetime.now()
    outdir = Path("results") / started.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    combos: list[tuple[str, str]] = []

    for label, model_str, effort in specs:
        for mode in modes:
            agent = build_agent(mode, store, effort=effort, include_summary=not a.typed_no_summary)
            if a.offline:
                from eval_cases import build_cases
                from fake_model import offline_model_for
                model = offline_model_for(mode, build_cases(a.tsv))
            else:
                model = config.make_model(model_str)
                # Prime connection + this (model, mode) prefix in the prompt cache, outside the timings.
                await agent.run("Reply with OK.", deps=deps, model=model, model_settings=ModelSettings(max_tokens=512))
            report = await dataset.evaluate(
                make_task(agent, deps, model), name=f"{label}/{mode}", task_name=mode,
                repeat=a.repeats, max_concurrency=a.concurrency, progress=False,
                metadata={"model": label, "mode": mode, "effort": str(effort)},
            )
            if a.show_pydantic_report:
                report.print(include_averages=True)
            new = flatten(report, label, mode)
            rows += new
            combos.append((label, mode))
            ok = sum(r["correct"] for r in new)
            print(f"{label:<16} {mode:<9} {ok}/{len(new)} correct · "
                  f"median total {fmt(pct([r.get('total_ms') for r in new], 50))} ms")

    with open(outdir / "runs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    meta = {"repeats": a.repeats, "typed_summary": not a.typed_no_summary, "offline": a.offline,
            "started": started.strftime("%Y-%m-%d %H:%M")}
    text = render(rows, combos, [c.name for c in dataset.cases], meta)
    (outdir / "report.md").write_text(text)
    print("\n" + text + f"\nWrote {outdir}/runs.jsonl and {outdir}/report.md")


if __name__ == "__main__":
    asyncio.run(main())
