"""The support agent under test, and the task function the evals call.

Structured output, one retrieval tool, four sentences of system prompt.

The prompt is deliberately NOT hardened against hallucination beyond telling the
agent to answer from retrieved policy and escalate when it cannot. This is not
an attempt to make the agent fail -- if it gets the question right every time,
that is a real and useful result. It is an attempt to measure an ordinary agent
rather than one tuned until it passes its own test.

The task returns the retrieved evidence and the run metrics alongside the
answer. Evaluators need the evidence to check claims against what was actually
retrieved, and the metrics carry cost and latency into the same report as
correctness, so all four axes are compared in one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResultEvent, RunContext
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.settings import ModelSettings

from . import corpus


class SupportAnswer(BaseModel):
    """What the agent must return. Shape is enforced; truth is not."""

    text: str = Field(description="The reply sent to the customer.")
    cited_policy_ids: list[str] = Field(
        default_factory=list,
        description="IDs of every policy used, e.g. ['BIL-104'].",
    )
    action: Literal["answer", "escalate"] = Field(
        description="'answer' if the policies resolve the question, 'escalate' if they do not.",
    )


class RunMetrics(BaseModel):
    """Cost and latency for one run.

    Two time-to-first-token numbers, on purpose:

      ttft_any    first activity of any kind, including the internal tool call
      ttft_final  first token of the reply the customer actually sees

    For an agent that retrieves before answering these differ by several
    seconds, and only the second is latency anyone experiences. Reporting the
    first is the most common way agent latency gets understated.
    """

    ttft_any_s: float | None = None
    ttft_final_s: float | None = None
    total_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = None
    tool_calls: int | None = None
    cost_usd: float | None = None


class SupportResult(BaseModel):
    """Everything one run produced. This is what evaluators receive.

    `run_id` exists because runs complete out of order under concurrency. The
    raw record is written the moment a run finishes, while evaluator results
    arrive later in case order, so the two cannot be joined by position. Every
    consumer joins on this instead.
    """

    run_id: int = -1
    answer: SupportAnswer
    evidence: dict[str, str] = Field(default_factory=dict)
    metrics: RunMetrics = Field(default_factory=RunMetrics)


@dataclass
class SupportDeps:
    """Records what the tool actually returned, so scoring never has to trust
    the agent's own account of what it looked up."""

    evidence: dict[str, str] = field(default_factory=dict)
    retrieved_ids: list[str] = field(default_factory=list)
    tool_called: bool = False


SYSTEM_PROMPT = """You are a customer support agent for Fernpost, a subscription invoicing tool for small businesses.
Always call lookup_policy before answering, and answer only from the policies it returns.
Cite the ID of every policy you use in cited_policy_ids.
If the returned policies do not answer the customer's question, set action to "escalate" and tell the customer you are passing it to the billing team."""


def build_agent(model: str) -> Agent[SupportDeps, SupportAnswer]:
    agent = Agent(
        model,
        deps_type=SupportDeps,
        output_type=SupportAnswer,
        system_prompt=SYSTEM_PROMPT,
        retries=0,
    )

    @agent.tool
    def lookup_policy(ctx: RunContext[SupportDeps], question: str) -> dict[str, str]:
        """Search the Fernpost help center. Returns matching policies keyed by policy ID."""
        hits = corpus.search(question)
        ctx.deps.tool_called = True
        for policy in hits:
            if policy.id not in ctx.deps.retrieved_ids:
                ctx.deps.retrieved_ids.append(policy.id)
            ctx.deps.evidence[policy.id] = f"{policy.title}\n\n{policy.body}"
        return dict(ctx.deps.evidence)

    return agent


def make_task(model: str, temperature: float | None = None):
    """Build the callable that Dataset.evaluate runs, bound to one model.

    Streaming is used only to measure time to first token -- nothing is
    displayed. It is the only way to separate "the agent started working" from
    "the customer started seeing an answer".
    """
    agent = build_agent(model)
    settings = None if temperature is None else ModelSettings(temperature=temperature)

    async def answer_support_question(question: str) -> SupportResult:
        deps = SupportDeps()
        started = time.perf_counter()
        ttft_any = ttft_final = None
        result = None

        async with agent.run_stream_events(question, deps=deps, model_settings=settings) as stream:
            async for event in stream:
                now = time.perf_counter() - started
                if ttft_any is None and isinstance(event, (PartStartEvent, PartDeltaEvent)):
                    ttft_any = round(now, 3)
                if ttft_final is None and isinstance(event, FinalResultEvent):
                    ttft_final = round(now, 3)
                if isinstance(event, AgentRunResultEvent):
                    result = event.result

        usage = result.usage
        return SupportResult(
            answer=result.output,
            evidence=dict(deps.evidence),
            metrics=RunMetrics(
                ttft_any_s=ttft_any,
                ttft_final_s=ttft_final,
                total_s=round(time.perf_counter() - started, 3),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                requests=usage.requests,
                tool_calls=usage.tool_calls,
                # Real cost from pydantic-ai via genai-prices, not our estimate table.
                cost_usd=float(usage.cost) if usage.cost is not None else None,
            ),
        )

    return answer_support_question
