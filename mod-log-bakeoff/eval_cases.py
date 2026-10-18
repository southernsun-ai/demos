"""Eval questions with ground truth computed independently in plain Python from the raw TSV.

Grading is deterministic: an answer passes if every `must_include` item appears in its
lead paragraph (strict). A loose score (anywhere in the answer) is kept for diagnostics.
Numbers are matched as whole tokens after stripping thousands separators, so "1,089"
matches 1089, but "10" won't match inside "2026-10-01".
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

MOD_ACTIONS = {"warn", "mute", "timeout", "kick", "delete", "ban"}


@dataclass
class EvalCase:
    id: str
    question: str
    must_include: list[str]
    typed_capable: bool            # can the two typed tools answer this exactly?
    note: str = ""
    accept_any: list[list[str]] = field(default_factory=list)  # alternative answer sets (ties)
    reference_sql: str = ""        # a correct DuckDB query (cross-checked in tests)


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        r["_ts"] = datetime.strptime(r["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
    return rows


def _top(counter: Counter) -> tuple[str, int]:
    (k, v), *rest = counter.most_common()
    if rest and rest[0][1] == v:
        raise ValueError(f"tie in ground truth: {counter.most_common(3)}")
    return k, v


def build_cases(path: str = "mod_channel_15k.tsv") -> list[EvalCase]:
    rows = _load(path)
    acted = [r for r in rows if r["action_taken"] in MOD_ACTIONS]
    in_month = lambda r, y, m: r["_ts"].year == y and r["_ts"].month == m  # noqa: E731
    cases: list[EvalCase] = []

    n_bans = sum(r["action_taken"] == "ban" for r in rows)
    cases.append(EvalCase("total_bans", "How many bans are in the log?", [str(n_bans)], True, "Answerable from the snapshot summary"))

    mod, n = _top(Counter(r["mod_username"] for r in rows if r["action_taken"] == "ban"))
    cases.append(EvalCase("top_banner", "Which moderator issued the most bans, and how many?", [mod, str(n)], True))

    n = sum(r["action_taken"] == "ban" and r["mod_username"] == "Mara" and in_month(r, 2026, 10) for r in rows)
    cases.append(EvalCase("mara_oct_bans", "How many bans did Mara issue in October 2026?", [str(n)], True))

    ch, n = _top(Counter(r["channel_name"] for r in acted))
    cases.append(EvalCase("top_channel_actions", "Which channel had the most mod actions, and how many?", [ch, str(n)], True))

    user, n = _top(Counter(r["author_username"] for r in acted))
    cases.append(EvalCase("top_offender", "Which user received the most mod actions, and how many?", [user, str(n)], True))

    lo, hi = datetime(2026, 9, 7), datetime(2026, 9, 14)
    n = sum(r["action_taken"] == "kick" and lo <= r["_ts"] < hi for r in rows)
    cases.append(EvalCase("kicks_week", "How many kicks happened from 2026-09-07 through 2026-09-13 inclusive?", [str(n)], True))

    reason, _ = _top(Counter(r["reason"] for r in rows if r["action_taken"] == "timeout"))
    cases.append(EvalCase("timeout_reason", "What is the most common reason given for timeouts?", [reason], True))

    mod, n = _top(Counter(r["mod_username"] for r in rows if r["action_taken"] == "warn" and in_month(r, 2026, 9)))
    cases.append(EvalCase("sept_warner", "Which moderator issued the most warnings in September 2026, and how many?", [mod, str(n)], True))

    n = sum(r["action_taken"] in MOD_ACTIONS and r["mod_username"] == "Ines" and r["channel_name"] == "#games" for r in rows)
    cases.append(EvalCase("ines_games", "How many mod actions did Ines take in #games?", [str(n)], True))

    last = max((r for r in rows if r["author_username"] == "grumpy_r178" and r["action_taken"] == "timeout"), key=lambda r: r["_ts"])
    cases.append(EvalCase("grumpy_last_timeout", "What was grumpy_r178 most recently timed out for, and by which moderator?",
                      [last["reason"], last["mod_username"]], True))

    # ---- beyond the typed tools' filters / group-bys --------------------------------
    n = sum(bool(r["edited_at_utc"]) for r in rows)
    cases.append(EvalCase("edited_count", "How many messages were edited?", [str(n)], False, "No edited filter in typed tools"))

    n = sum(int(r["attachment_count"]) > 0 for r in rows)
    cases.append(EvalCase("with_attachments", "How many messages have at least one attachment?", [str(n)], False))

    hour, n = _top(Counter(r["_ts"].hour for r in acted))
    cases.append(EvalCase("peak_hour", "Which hour of the day (UTC) has the most mod actions, and how many?",
                      [str(n)], False, "Needs hour-of-day grouping",
                      accept_any=[[f"{hour:02d}:00"], [f"{hour}:00"], [f"hour {hour}"], [f"{hour} UTC"]]))

    n = len({r["author_username"] for r in acted})
    cases.append(EvalCase("distinct_actioned", "How many distinct users received at least one mod action?", [str(n)], False, "Needs COUNT DISTINCT"))

    by_user = {}
    for r in sorted(acted, key=lambda r: r["_ts"]):
        by_user.setdefault(r["author_username"], []).append(r)
    n = sum(1 for evs in by_user.values()
            if any(a["action_taken"] == "warn" and any(b["action_taken"] == "ban" and b["_ts"] > a["_ts"] for b in evs) for a in evs))
    cases.append(EvalCase("warn_then_ban", "How many users were warned and later banned?", [str(n)], False, "Needs per-user sequence logic"))

    edited_fast = sum(1 for r in rows if r["edited_at_utc"] and
                      datetime.strptime(r["edited_at_utc"], "%Y-%m-%dT%H:%M:%SZ") - r["_ts"] <= timedelta(minutes=5))
    cases.append(EvalCase("edited_within_5m", "How many messages were edited within 5 minutes of being posted?", [str(edited_fast)], False))

    for c in cases:
        c.reference_sql = REFERENCE_SQL[c.id]
    return cases


_A = "action_taken IN ('warn','mute','timeout','kick','delete','ban')"
REFERENCE_SQL = {
    "total_bans": "SELECT count(*) FROM events WHERE action_taken='ban'",
    "top_banner": "SELECT mod_username, count(*) c FROM events WHERE action_taken='ban' GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    "mara_oct_bans": "SELECT count(*) FROM events WHERE action_taken='ban' AND mod_username='Mara' AND ts>='2026-10-01' AND ts<'2026-11-01'",
    "top_channel_actions": f"SELECT channel_name, count(*) FROM events WHERE {_A} GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    "top_offender": f"SELECT author_username, count(*) FROM events WHERE {_A} GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    "kicks_week": "SELECT count(*) FROM events WHERE action_taken='kick' AND ts>='2026-09-07' AND ts<'2026-09-14'",
    "timeout_reason": "SELECT reason FROM events WHERE action_taken='timeout' GROUP BY 1 ORDER BY count(*) DESC LIMIT 1",
    "sept_warner": "SELECT mod_username, count(*) FROM events WHERE action_taken='warn' AND ts>='2026-09-01' AND ts<'2026-10-01' GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    "ines_games": f"SELECT count(*) FROM events WHERE {_A} AND mod_username='Ines' AND channel_name='#games'",
    "grumpy_last_timeout": "SELECT reason, mod_username FROM events WHERE author_username='grumpy_r178' AND action_taken='timeout' ORDER BY ts DESC LIMIT 1",
    "edited_count": "SELECT count(edited_at) FROM events",
    "with_attachments": "SELECT count(*) FROM events WHERE attachment_count > 0",
    "peak_hour": f"SELECT 'hour ' || hour(ts) AS hour, count(*) FROM events WHERE {_A} GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    "distinct_actioned": f"SELECT count(DISTINCT author_username) FROM events WHERE {_A}",
    "warn_then_ban": "SELECT count(DISTINCT w.author_username) FROM events w JOIN events b ON w.author_username=b.author_username AND b.action_taken='ban' AND b.ts>w.ts WHERE w.action_taken='warn'",
    "edited_within_5m": "SELECT count(*) FROM events WHERE edited_at <= ts + INTERVAL 5 MINUTE",
}


_NUM = re.compile(r"(?<![\d.\-:])\d+(?![\d\-:])")


def _contains(answer: str, item: str) -> bool:
    if item.isdigit():
        nums = set(_NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", answer)))
        return item in nums
    return item.lower() in answer.lower()


def lead(answer: str) -> str:
    """The answer's first paragraph. The style rules require the answer to come first, so
    grading only this avoids false passes from numbers that appear in supporting tables."""
    return answer.strip().split("\n\n")[0]


def grade_values(must_include: list[str], accept_any: list[list[str]], answer: str, strict: bool = True) -> bool:
    if strict:
        answer = lead(answer)
    if not all(_contains(answer, x) for x in must_include):
        return False
    if accept_any:
        return any(all(_contains(answer, x) for x in alt) for alt in accept_any)
    return True


def grade(case: EvalCase, answer: str, strict: bool = True) -> bool:
    return grade_values(case.must_include, case.accept_any, answer, strict)


# ---- pydantic_evals dataset -------------------------------------------------------

@dataclass
class AnswerCorrect(Evaluator[str, str, dict]):
    """Deterministic grader: expected values must appear in the answer's lead paragraph
    (`correct`); `correct_loose` accepts them anywhere. A run that errored is incorrect."""

    def evaluate(self, ctx: EvaluatorContext[str, str, dict]) -> dict[str, bool]:
        if ctx.attributes.get("error"):
            return {"correct": False, "correct_loose": False}
        md = ctx.metadata or {}
        args = (md["must_include"], md.get("accept_any") or [], ctx.output or "")
        return {"correct": grade_values(*args, strict=True), "correct_loose": grade_values(*args, strict=False)}


def build_dataset(path: str = "mod_channel_15k.tsv", only: set[str] | None = None) -> Dataset[str, str, dict]:
    cases = [c for c in build_cases(path) if not only or c.id in only]
    return Dataset(
        name="mod-log-admin-chat",
        cases=[
            Case(
                name=c.id,
                inputs=c.question,
                metadata={"typed_capable": c.typed_capable, "must_include": c.must_include,
                          "accept_any": c.accept_any, "note": c.note},
            )
            for c in cases
        ],
        evaluators=[AnswerCorrect()],
    )


if __name__ == "__main__":
    for c in build_cases():
        print(f"{c.id:<22} typed_capable={c.typed_capable!s:<5}  {c.question}  ->  {c.must_include} {c.accept_any[:1] or ''}")
