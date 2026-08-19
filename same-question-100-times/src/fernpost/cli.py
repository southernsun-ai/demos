"""Command line entry point."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from . import charts, corpus, evals, models, scoring
from .harness import (
    RESULTS_DIR,
    configure_tracing,
    metric_summary,
    read_results,
    run_experiment,
    saved_pass_rates,
)
from .tickets import TICKETS

app = typer.Typer(add_completion=False, help="Run one support question many times and evaluate every run.")

# Pinned to this project's root rather than bare load_dotenv(), which searches
# parent directories and will happily pick up an unrelated .env from somewhere
# above the repo. Explicit beats surprising, and it means the CLI behaves the
# same from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

CONFIRM_ABOVE = 10  # runs above this need confirmation, because they cost money


def _preflight(model: str, n: int, yes: bool, judge_model: str | None) -> None:
    """Check keys exist and confirm the spend before calling anything."""
    for label, target in (("model", model), ("judge", judge_model)):
        if not target:
            continue
        key = models.required_key(target)
        if key and not os.environ.get(key):
            typer.secho(f"{key} is not set ({label}: {target}). See .env.example.", fg="red")
            raise typer.Exit(1)

    estimate = models.estimate_cost(model, n)
    shown = "unknown" if estimate is None else f"~${estimate:.2f}"
    typer.echo(f"Model: {model}")
    if judge_model:
        typer.echo(f"Judge: {judge_model}  (one judge call per run, billed on top)")
    typer.echo(f"Runs:  {n}   Estimated agent cost: {shown}")
    if n > CONFIRM_ABOVE and not yes:
        typer.confirm(f"This will make {n} live API calls. Continue?", abort=True)


def _print_rates(rates: dict[str, float]) -> None:
    layer = {
        "EvidenceContract": "1",
        "CorrectAction": "1",
        "NaiveClaimRegex": "1",
        "used_policy_lookup": "1",
        "at_most_2_tool_calls": "1",
        "faithful_to_policy": "2",
    }
    typer.echo("  layer  evaluator                 pass rate")
    for name, rate in rates.items():
        typer.echo(f"    {layer.get(name, '?')}    {name:<24}  {rate * 100:>5.0f}%")


def _print_metrics(summary: dict) -> None:
    rows = [
        ("ttft_any_s", "first token, any"),
        ("ttft_final_s", "first token the customer sees"),
        ("total_s", "total latency"),
        ("cost_usd", "cost per run"),
        ("tool_calls", "tool calls"),
        ("input_tokens", "input tokens"),
    ]
    typer.echo(f"  {'':<32}{'p50':>10}{'p95':>10}{'min':>10}{'max':>10}")
    def fmt(value, key: str) -> str:
        if value is None:
            return "-"
        return f"{value:.5f}" if key == "cost_usd" else f"{value:g}"

    for key, label in rows:
        d = summary[key]
        cells = "".join(f"{fmt(d[stat], key):>10}" for stat in ("p50", "p95", "min", "max"))
        typer.echo(f"  {label:<32}{cells}")
    if summary.get("total_cost_usd") is not None:
        typer.echo(f"\n  total spend this experiment  ${summary['total_cost_usd']:.4f}")


@app.command()
def run(
    ticket: str = typer.Option("uncovered", help=f"Which ticket: {' | '.join(TICKETS)}"),
    n: int = typer.Option(100, help="Number of runs."),
    model: str = typer.Option(models.PRIMARY, help="Any Pydantic AI model string."),
    temperature: float | None = typer.Option(
        None, help="Sampling temperature. Omit for the provider default. Some newer models reject it."
    ),
    concurrency: int = typer.Option(5, help="Concurrent in-flight runs."),
    judge: bool = typer.Option(True, help="Run the layer 2 faithfulness judge."),
    judge_model: str = typer.Option(evals.DEFAULT_JUDGE_MODEL, help="Model for the judge."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation."),
) -> None:
    """Run one ticket N times, evaluate every run, and write raw results to JSONL."""
    if ticket not in TICKETS:
        typer.secho(f"Unknown ticket {ticket!r}. Choose from: {', '.join(TICKETS)}", fg="red")
        raise typer.Exit(1)
    _preflight(model, n, yes, judge_model if judge else None)

    if configure_tracing():
        typer.echo("Logfire: sending traces and evaluations")
    else:
        typer.echo("Logfire: instrumented locally, not sending (no LOGFIRE_TOKEN)")

    path, report = asyncio.run(
        run_experiment(
            ticket, n, model, temperature, concurrency,
            judge=judge, judge_model=judge_model if judge else None,
        )
    )

    _, runs, _ = read_results(path)
    typer.echo("")
    _print_rates(evals.pass_rates(report))
    typer.echo("")
    _print_metrics(metric_summary(runs))
    typer.echo(f"\nWrote {path}")
    if judge:
        typer.secho(
            "\nThe judge is UNCALIBRATED. Run `fernpost review` before quoting faithful_to_policy.",
            fg="yellow",
        )


@app.command()
def doctor() -> None:
    """Check the setup and report what currently works.

    Prints whether each key is present, never its value.
    """
    import sys

    ok = True
    typer.echo(f"Python           {sys.version.split()[0]}")

    try:
        policies = corpus.load_corpus()
        typer.echo(f"Corpus           {len(policies)} policies, hash {corpus.corpus_hash()}")
    except Exception as exc:  # noqa: BLE001 -- doctor reports problems, it does not raise them
        typer.secho(f"Corpus           FAILED: {exc}", fg="red")
        ok = False

    typer.echo("")
    gateway = bool(os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY"))
    direct = {
        name: bool(os.environ.get(name))
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    }

    typer.echo("Model access")
    mark = "set" if gateway else "not set"
    typer.echo(f"  PYDANTIC_AI_GATEWAY_API_KEY   {mark}")
    for name, present in direct.items():
        typer.echo(f"  {name:<28}  {'set' if present else 'not set'}")

    if gateway:
        typer.secho("  -> gateway/* model strings work, all three peers available.", fg="green")
    elif any(direct.values()):
        usable = [models.DIRECT[p] for p in models.PEERS
                  if direct.get(models.API_KEY_ENV.get(models.provider_of(models.DIRECT[p]), ""))]
        typer.secho("  -> no gateway key. Pass direct model strings instead:", fg="yellow")
        for target in usable:
            typer.echo(f"       --model {target}")
    else:
        typer.secho("  -> no model access. Copy .env.example to .env and add a key.", fg="red")
        ok = False

    typer.echo("")
    if os.environ.get("LOGFIRE_TOKEN"):
        typer.secho("Logfire          set. Traces and evaluations are sent.", fg="green")
    else:
        typer.secho(
            "Logfire          not set. Runs still work and the trace-based\n"
            "                 evaluators still work, but nothing is sent.", fg="yellow",
        )

    typer.echo("")
    if ENV_FILE.exists():
        typer.echo(f".env             found at {ENV_FILE}")
    else:
        typer.secho(f".env             not found. Expected at {ENV_FILE}", fg="yellow")

    typer.echo("")
    if ok:
        typer.secho("Ready. Try:  uv run fernpost run --ticket uncovered --n 5", fg="green")
    else:
        raise typer.Exit(1)


@app.command()
def score(results_file: Path = typer.Argument(..., help="A results JSONL file.")) -> None:
    """Summarise a saved results file. Reads from disk, never calls an API."""
    config, runs, evaluations = read_results(results_file)

    typer.echo(f"Model:       {config['model']}")
    typer.echo(f"Judge:       {config.get('judge_model') or 'not run'}")
    typer.echo(f"Ticket:      {config['ticket_id']}   temperature={config['temperature']}")
    typer.echo(f"Corpus hash: {config['corpus_hash']}   git={config['git_sha']}   n={len(runs)}")

    rates = saved_pass_rates(evaluations)
    if rates:
        typer.echo("")
        _print_rates(rates)
    else:
        typer.secho("\nNo saved evaluations in this file.", fg="yellow")

    typer.echo("")
    _print_metrics(metric_summary(runs))
    typer.secho(
        "\nNaiveClaimRegex is the deliberately crude baseline, kept for contrast.\n"
        "faithful_to_policy is a judge and is not ground truth until calibrated.",
        fg="yellow",
    )


@app.command()
def chart(
    grid: Path = typer.Option(None, help="Results file for the 100-run grid."),
    compare: Path = typer.Option(None, help="Control results file, for the comparison bars."),
    model_files: list[Path] = typer.Option(None, "--model-file", help="One results file per peer model."),
) -> None:
    """Render charts to charts/."""
    if not grid and not model_files:
        typer.secho("Nothing to chart. Pass --grid and/or --model-file.", fg="red")
        raise typer.Exit(1)

    def outcomes(path: Path) -> list[dict]:
        """One row per run, in the shape charts.run_grid expects."""
        _, runs, evaluations = read_results(path)
        # Join on run id. Runs finish out of order, so position is meaningless.
        by_run = {c.get("run"): c["assertions"] for c in evaluations.get("cases", [])}
        rows = []
        for run_record in runs:
            assertions = by_run.get(run_record.get("run"), {})
            correct = assertions.get("CorrectAction", {}).get("value")
            faithful = assertions.get("faithful_to_policy", {}).get("value", True)
            if correct and faithful:
                rows.append({"klass": "escalated"})
            elif correct is None:
                rows.append({"klass": "answered_no_claim"})
            else:
                rows.append({"klass": "invented"})
        return rows

    if grid:
        rows = outcomes(grid)
        typer.echo(f"Wrote {charts.run_grid(rows)}")
        if compare:
            def fail_rate(path: Path) -> float:
                data = outcomes(path)
                return sum(1 for r in data if r["klass"] == "invented") / max(len(data), 1)
            typer.echo(f"Wrote {charts.comparison_bars(fail_rate(grid), fail_rate(compare))}")

    if model_files:
        rates = {}
        for path in model_files:
            config, _runs, evaluations = read_results(path)
            saved = saved_pass_rates(evaluations)
            rates[config["model"]] = 1 - saved.get("CorrectAction", 1.0)
        typer.echo(f"Wrote {charts.model_bars(rates)}")


@app.command()
def review(
    results_file: Path = typer.Argument(...),
    sample: int = typer.Option(30, help="How many runs to hand-label."),
) -> None:
    """Layer 3. Hand-label a sample and print agreement with the automated checks."""
    from .review import run_review

    run_review(results_file, sample)


@app.command("models")
def show_models() -> None:
    """Print the peer class and why these models were chosen."""
    typer.echo(models.__doc__.strip())
    typer.echo("")
    typer.echo(f"PRIMARY  {models.PRIMARY}")
    for peer in models.PEERS:
        marker = "*" if peer == models.PRIMARY else " "
        est = models.estimate_cost(peer, 100)
        typer.echo(f"  {marker} {peer:<42} ~${est:.2f} per 100 runs   direct: {models.DIRECT[peer]}")
    typer.echo(f"ANCHOR   {models.ANCHOR}  (optional, outside the peer class)")
    typer.echo(f"JUDGE    {evals.DEFAULT_JUDGE_MODEL}  (layer 2, stronger than the systems it grades)")


@app.command("your-agent")
def your_agent(
    target: str = typer.Option(..., help="Import path to a callable, e.g. examples.naive_agent:answer"),
    question: str = typer.Option(..., help="The question to send, every time."),
    n: int = typer.Option(20, help="Number of runs."),
    forbid: list[str] = typer.Option(..., "--forbid", help="A claim your system must never make. Repeatable."),
    concurrency: int = typer.Option(5),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Point this harness at your own agent.

    TARGET is a Python import path to a callable that takes a string and returns
    a string. Sync or async both work. Your callable owns its model, keys,
    retrieval, and prompt -- this only calls it N times and counts how often the
    reply commits you to something in --forbid.

    Example:

      uv run fernpost your-agent \\
        --target examples.naive_agent:answer \\
        --question "Do I get credited for seats I stopped using?" \\
        --n 20 --forbid credit --forbid refund --forbid "roll over"
    """
    module_name, _, attr = target.partition(":")
    if not attr:
        typer.secho("TARGET must look like module.path:callable", fg="red")
        raise typer.Exit(1)

    sys.path.insert(0, str(Path.cwd()))
    try:
        func = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        typer.secho(f"Could not import {target}: {exc}", fg="red")
        raise typer.Exit(1) from exc

    typer.echo(f"Target: {target}")
    typer.echo(f"Runs:   {n}   Forbidden claims: {', '.join(forbid)}")
    typer.secho("Cost is whatever your own agent costs per call. This tool cannot estimate it.", fg="yellow")
    if n > CONFIRM_ABOVE and not yes:
        typer.confirm(f"This will call your agent {n} times. Continue?", abort=True)

    async def call_once(index: int, limiter: asyncio.Semaphore) -> dict:
        async with limiter:
            try:
                result = func(question)
                if inspect.isawaitable(result):
                    result = await result
                return {"run": index, "output": {"text": str(result)}, "error": None}
            except Exception as exc:  # noqa: BLE001
                return {"run": index, "output": None, "error": f"{type(exc).__name__}: {exc}"}

    async def go() -> list[dict]:
        limiter = asyncio.Semaphore(concurrency)
        return await asyncio.gather(*(call_once(i, limiter) for i in range(n)))

    records = asyncio.run(go())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"your-agent_{attr}_{n}.jsonl"
    hits = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record": "config", "target": target, "question": question,
                                 "n": n, "forbid": list(forbid)}) + "\n")
        for record in sorted(records, key=lambda r: r["run"]):
            text = (record["output"] or {}).get("text", "")
            asserted, evidence = scoring.assert_forbidden(text, list(forbid))
            record["asserted_forbidden"] = asserted
            record["evidence"] = evidence
            hits += asserted
            handle.write(json.dumps(record) + "\n")

    errors = sum(1 for r in records if r["error"])
    typer.echo("")
    typer.echo(f"  asserted a forbidden claim   {hits}/{n}  ({hits / n * 100:.0f}%)")
    typer.echo(f"  errored                      {errors}")
    typer.echo(f"Wrote {path}")
    typer.secho("Crude regex scoring, same caveat as the rest of the repo. Read the runs.", fg="yellow")
