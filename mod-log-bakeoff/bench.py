"""Latency benchmark against a running server, over real HTTP + SSE.

  python bench.py --url http://localhost:8000 --runs 20

Reports p50 / p95 for: first status event, TTFT (first answer token) and total time,
split by turn shape (no tool vs. tool call), measured client-side.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

QUESTIONS = {
    "no_tool": "Give me a quick overview summary of the data.",
    "aggregate": "Who issued the most bans?",
    "find": "Show me the latest kicks.",
}


async def one(client: httpx.AsyncClient, url: str, q: str) -> dict:
    t0 = time.perf_counter()
    first_status = ttft = None
    server = {}
    async with client.stream("POST", f"{url}/chat", json={"message": q}) as r:
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                now = 1000 * (time.perf_counter() - t0)
                if event == "status" and first_status is None:
                    first_status = now
                elif event == "delta" and ttft is None:
                    ttft = now
                elif event == "done":
                    server = json.loads(line[6:])
    return {"first_status": first_status, "ttft": ttft, "total": 1000 * (time.perf_counter() - t0), "server": server}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    async with httpx.AsyncClient(timeout=60) as client:
        await one(client, a.url, QUESTIONS["no_tool"])  # warm the client connection
        print(f"{'shape':<10} {'status p50/p95':>16} {'TTFT p50/p95':>16} {'total p50/p95':>16}  model calls")
        for shape, q in QUESTIONS.items():
            res = [await one(client, a.url, q) for _ in range(a.runs)]
            col = lambda k: f"{pct([r[k] for r in res], 50):6.1f}/{pct([r[k] for r in res], 95):6.1f}"  # noqa: E731
            calls = statistics.mode(r["server"].get("model_requests", 0) for r in res)
            print(f"{shape:<10} {col('first_status'):>16} {col('ttft'):>16} {col('total'):>16}  {calls}")


if __name__ == "__main__":
    asyncio.run(main())
