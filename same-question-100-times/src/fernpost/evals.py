"""The three-layer evaluation suite.

Structure follows Airbnb's playbook as written up by Pydantic:
https://pydantic.dev/articles/three-layer-evals-logfire

  Layer 1  Programmatic checks for failures code can identify exactly.
           Deterministic, no model calls, run on everything.
  Layer 2  One LLM judge with one narrow rubric, binary, with a recorded reason.
  Layer 3  Human review as the calibration set. See review.py.

The rule that makes it work: read roughly 100 traces before writing a single
evaluator. Everything in this file was written after reading real runs, and the
one evaluator written before reading anything -- the regex in scoring.py -- is
included on purpose, clearly labelled, so the difference is visible in the same
report.

Send each decision to the cheapest method that can answer it reliably. Whether
a cited policy ID exists is a set operation. Whether the prose follows from the
policy needs a judge. Whether the rubric matches expert judgment needs a human.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    LLMJudge,
    MaxToolCalls,
    ToolCorrectness,
)

from .agent import SupportResult
from .scoring import asserts_money_back
from .tickets import TICKETS

# ---------------------------------------------------------------------------
# Layer 1 -- programmatic. No model calls. Run these on everything.
# ---------------------------------------------------------------------------


@dataclass
class EvidenceContract(Evaluator[str, SupportResult, dict]):
    """Objective contract between the answer and what retrieval returned.

    Nothing here is a matter of opinion, so nothing here needs a model: every
    question is a set membership test or a branch on a literal field.
    """

    def evaluate(self, ctx: EvaluatorContext[str, SupportResult, dict]) -> EvaluationReason:
        answer = ctx.output.answer
        cited = {c.strip().upper() for c in answer.cited_policy_ids}
        available = {k.strip().upper() for k in ctx.output.evidence}

        unknown = sorted(cited - available)
        if unknown:
            return EvaluationReason(
                value=False,
                reason=f"Cited policy IDs that retrieval never returned: {', '.join(unknown)}",
            )
        if answer.action == "answer" and not cited:
            return EvaluationReason(value=False, reason="Answered without citing any policy.")
        if answer.action == "escalate" and not available:
            # Escalating with nothing retrieved is defensible but means the tool
            # never produced evidence -- worth surfacing rather than passing.
            return EvaluationReason(value=False, reason="Escalated with no evidence retrieved.")
        return EvaluationReason(value=True)


@dataclass
class CorrectAction(Evaluator[str, SupportResult, dict]):
    """Did it route the ticket the way the documentation supports?

    This is the correctness axis. The expected action comes from case metadata,
    not from the agent, and it is decided by whether the corpus answers the
    question at all:

      uncovered  nothing documents removing seats mid-term  -> escalate
      control    BIL-104 documents adding seats             -> answer
    """

    def evaluate(self, ctx: EvaluatorContext[str, SupportResult, dict]) -> EvaluationReason:
        expected = (ctx.metadata or {}).get("expected_action")
        if expected is None:
            return EvaluationReason(value=False, reason="Case metadata has no expected_action.")
        actual = ctx.output.answer.action
        if actual == expected:
            return EvaluationReason(value=True)
        return EvaluationReason(value=False, reason=f"Expected {expected!r}, got {actual!r}.")


@dataclass
class NaiveClaimRegex(Evaluator[str, SupportResult, dict]):
    """The crude regex scorer, kept deliberately, as the control for the method.

    This is what an evaluator looks like when it is written before anyone has
    read a trace: it pattern-matches the vocabulary of the failure instead of
    the failure. It is in the suite so its disagreement with the judge and with
    human labels shows up in the same report as everything else.

    Passing means the reply did NOT assert a credit, refund, or rollback.
    """

    def evaluate(self, ctx: EvaluatorContext[str, SupportResult, dict]) -> EvaluationReason:
        asserted, sentence = asserts_money_back(ctx.output.answer.text)
        if asserted:
            return EvaluationReason(value=False, reason=f"Regex matched: {sentence}")
        return EvaluationReason(value=True)


used_policy_lookup = ToolCorrectness(
    expected_tools=["lookup_policy"],
    allow_extra=True,
    evaluation_name="used_policy_lookup",
)

# Retrieval is called more than once on some runs, and each call returns full
# policy bodies. That is the single biggest driver of cost variance between two
# runs of the identical question, so it is an assertion, not just a statistic.
bounded_tool_calls = MaxToolCalls(2, evaluation_name="at_most_2_tool_calls")


# ---------------------------------------------------------------------------
# Layer 2 -- one judge, one job.
# ---------------------------------------------------------------------------

FAITHFULNESS_RUBRIC = """
The output contains `answer` (the reply sent to the customer) and `evidence`
(the policy documents retrieval actually returned).

Pass only when every factual claim about Fernpost policy in `answer.text` is
supported by the text of an `evidence` entry named in `answer.cited_policy_ids`.

Accurate paraphrase passes. Saying that a question is not covered passes.
Routing to the billing team passes.

Fail when the reply states or implies any of the following without direct
support in the cited evidence: a credit, a refund, a rollover of unused seats,
proration applied to seats being removed, a specific monetary amount that the
evidence does not contain, or any deadline, eligibility rule, or guarantee not
present in the cited policy.
"""

# Judged by a model materially stronger than the ones under test. A judge from
# the same tier as the system it grades cannot be assumed to see what that tier
# misses.
DEFAULT_JUDGE_MODEL = "gateway/anthropic:claude-sonnet-5"


def faithfulness_judge(model: str = DEFAULT_JUDGE_MODEL) -> LLMJudge:
    """Binary, with a recorded reason.

    Binary rather than 1-to-10 because a threshold on a vague scale is not a
    decision boundary anyone can act on, and because agreement with a human is
    only measurable when both are answering the same yes/no question.

    UNCALIBRATED until it has been checked against human labels -- see
    `fernpost review`. An uncalibrated judge is another model output wearing a
    lab coat, and its pass rate should not be quoted before that check.
    """
    return LLMJudge(
        rubric=FAITHFULNESS_RUBRIC,
        model=model,
        include_input=True,
        assertion={"evaluation_name": "faithful_to_policy", "include_reason": True},
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

EXPECTED_ACTION = {
    "uncovered": "escalate",  # nothing in the corpus covers removing seats
    "control": "answer",      # BIL-104 covers adding seats outright
}


def build_dataset(
    ticket: str,
    n: int,
    judge: bool = True,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dataset[str, SupportResult, dict]:
    """n copies of one case.

    Repeating a single case is the whole experiment. Every case has identical
    input, so any spread in the report -- correctness, cost, latency -- is
    variance in the system rather than variance in the question.
    """
    if ticket not in TICKETS:
        raise ValueError(f"Unknown ticket {ticket!r}. Choose from {sorted(TICKETS)}.")

    metadata = {"ticket_id": ticket, "expected_action": EXPECTED_ACTION[ticket]}
    evaluators = [
        EvidenceContract(),
        CorrectAction(),
        NaiveClaimRegex(),
        used_policy_lookup,
        bounded_tool_calls,
    ]
    if judge:
        evaluators.append(faithfulness_judge(judge_model))

    return Dataset[str, SupportResult, dict](
        name=f"fernpost-{ticket}",
        cases=[
            Case(name=f"{ticket}_{i:03d}", inputs=TICKETS[ticket], metadata=dict(metadata))
            for i in range(n)
        ],
        evaluators=evaluators,
    )


def pass_rates(report) -> dict[str, float]:
    """Pass rate per evaluator, over the cases that produced a result."""
    totals: dict[str, list[bool]] = {}
    for case in report.cases:
        for name, assertion in case.assertions.items():
            totals.setdefault(name, []).append(bool(assertion.value))
    return {name: sum(values) / len(values) for name, values in totals.items() if values}
