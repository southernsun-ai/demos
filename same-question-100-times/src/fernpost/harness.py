"""Runs the experiment: n copies of one case through the eval suite.

Two rules matter here:

1. Every completed run is written to disk immediately, before any evaluation
   touches it. A crash at run 87 keeps the first 86, and re-scoring never costs
   another API call.
2. Tracing is always instrumented, even with no Logfire token. The agentic
   evaluators (ToolCorrectness, MaxToolCalls) read OpenTelemetry spans rather
   than the prose -- with no instrumentation there are no spans, and they fail
   every case while looking like a real result. Sending to Logfire is optional;
   instrumenting is not.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from . import corpus
from .agent import SupportResult, make_task
from .evals import build_dataset

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_TRACING_READY = False


def configure_tracing() -> bool:
    """Instrument Pydantic AI. Returns whether results are also sent to Logfire.

    Always instruments. See rule 2 in the module docstring -- the agentic
    evaluators are trace-based, so skipping this silently zeroes them.
    """
    global _TRACING_READY
    import logfire

    send = bool(os.environ.get("LOGFIRE_TOKEN"))
    if not _TRACING_READY:
        logfire.configure(send_to_logfire=send, console=False)
        logfire.instrument_pydantic_ai()
        _TRACING_READY = True
    return send


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- git absent or not a repo is fine, it is metadata
        return "unknown"


def _slug(model: str) -> str:
    """'gateway/anthropic:claude-haiku-4-5' -> 'gateway-anthropic-claude-haiku-4-5'."""
    return model.replace(":", "-").replace("/", "-")


def results_path(ticket_id: str, model: str, temperature: float | None, n: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 -- local time, readable filenames
    temp = "default" if temperature is None else str(temperature)
    return RESULTS_DIR / f"{ticket_id}_{_slug(model)}_t{temp}_{n}_{stamp}.jsonl"


async def run_experiment(
    ticket: str,
    n: int,
    model: str,
    temperature: float | None = None,
    concurrency: int = 5,
    judge: bool = True,
    judge_model: str | None = None,
    progress=None,
):
    """Run `ticket` n times against `model` and evaluate every run.

    Returns (results_path, report).
    """
    configure_tracing()

    dataset = build_dataset(
        ticket, n, judge=judge,
        **({"judge_model": judge_model} if judge_model else {}),
    )
    task = make_task(model, temperature)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = results_path(ticket, model, temperature, n)

    handle = path.open("w", encoding="utf-8")
    handle.write(json.dumps({
        "record": "config",
        "model": model,
        "judge_model": judge_model if judge else None,
        "temperature": temperature,
        "n": n,
        "ticket_id": ticket,
        "corpus_hash": corpus.corpus_hash(),
        "corpus_ids": sorted(corpus.load_corpus()),
        "git_sha": _git_sha(),
        "started_at": datetime.now(UTC).isoformat(),
    }) + "\n")
    handle.flush()

    write_lock = asyncio.Lock()
    counter = {"done": 0}

    async def recording_task(question: str) -> SupportResult:
        """The task under test, wrapped so the raw result lands on disk first."""
        result = await task(question)
        async with write_lock:
            run_id = counter["done"]
            counter["done"] += 1
            result.run_id = run_id
            handle.write(json.dumps({
                "run": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "output": result.answer.model_dump(),
                "retrieved_ids": sorted(result.evidence),
                "tool_called": bool(result.evidence),
                "metrics": result.metrics.model_dump(),
            }) + "\n")
            handle.flush()  # a crash at run 87 keeps the first 86
            if progress:
                progress(counter["done"])
        return result

    try:
        report = await dataset.evaluate(
            recording_task,
            name=f"{ticket}-{_slug(model)}",
            max_concurrency=concurrency,
        )
        # Evaluation results go into the same file, so `fernpost score` never
        # needs to re-run the agent or re-call the judge.
        handle.write(json.dumps({
            "record": "evaluations",
            "cases": [
                {
                    "case": case.name,
                    "run": getattr(case.output, "run_id", None),
                    "assertions": {
                        name: {"value": bool(a.value), "reason": getattr(a, "reason", None)}
                        for name, a in case.assertions.items()
                    },
                }
                for case in report.cases
            ],
        }) + "\n")
        handle.flush()
    finally:
        handle.close()

    return path, report


def read_results(path: Path) -> tuple[dict, list[dict], dict]:
    """Split a results file into (config header, run records, evaluations)."""
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or lines[0].get("record") != "config":
        raise ValueError(f"{path} has no config header")
    evaluations: dict = {}
    runs = []
    for line in lines[1:]:
        if line.get("record") == "evaluations":
            evaluations = line
        else:
            runs.append(line)
    return lines[0], sorted(runs, key=lambda r: r.get("run", 0)), evaluations


def saved_pass_rates(evaluations: dict) -> dict[str, float]:
    """Pass rate per evaluator, read back from a saved results file."""
    totals: dict[str, list[bool]] = {}
    for case in evaluations.get("cases", []):
        for name, assertion in case["assertions"].items():
            totals.setdefault(name, []).append(bool(assertion["value"]))
    return {name: sum(v) / len(v) for name, v in totals.items() if v}


def metric_summary(runs: list[dict]) -> dict:
    """p50 / p95 / total for the cost and latency axes."""
    def percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(round((pct / 100) * (len(ordered) - 1)), len(ordered) - 1)
        return ordered[index]

    def column(key: str) -> list[float]:
        return [r["metrics"][key] for r in runs if r.get("metrics", {}).get(key) is not None]

    summary: dict[str, object] = {}
    for key in ("ttft_any_s", "ttft_final_s", "total_s", "cost_usd", "tool_calls", "input_tokens"):
        values = column(key)
        summary[key] = {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    costs = column("cost_usd")
    summary["total_cost_usd"] = sum(costs) if costs else None
    return summary
