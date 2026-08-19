"""Which models this experiment runs against, and why those three.

The model choice is not arbitrary and it is not a benchmark selection. A tier-1
support agent is a cost-and-latency problem before it is a reasoning problem:

  - Volume. A mid-size SaaS company fields hundreds of billing tickets a week.
    Per-ticket cost is the binding constraint, so nobody puts a frontier model
    on first-line deflection.
  - Latency. Either a customer is waiting or a human agent is waiting. The
    budget is low single-digit seconds end to end.
  - Tool calling and structured output. The agent has to call retrieval and
    return a schema. This is the real capability bar, and all three clear it.
  - Context. Enough room to hold a policy corpus. Not a differentiator here.

That specification defines a peer class: the current small/fast tier from each
major lab. These three are what a team shipping a support agent would actually
choose between. They are compared against each other, not ranked against models
built for a different job.

PRIMARY is the spine of the experiment. The run grid, the trace autopsy, and the
headline number all come from it. The other two answer exactly one question --
is this a property of the model, or of the situation? -- and nothing else.

Model strings default to the Pydantic AI Gateway, so one key reaches all three
providers. Every constant here is an ordinary model string, so direct provider
strings work identically: pass --model anthropic:claude-haiku-4-5 and bring your
own ANTHROPIC_API_KEY. See DIRECT below.
"""

PRIMARY = "gateway/anthropic:claude-haiku-4-5"

PEERS = [
    "gateway/anthropic:claude-haiku-4-5",
    "gateway/openai:gpt-5.4-mini",
    "gateway/google-cloud:gemini-3.7-flash",
]

# An optional fourth data point, deliberately outside the peer class. If a
# frontier model invents the same policy at a non-zero rate, the finding is
# structural rather than a small-model artifact. If it does not, that is a real
# result too, and it becomes the recommendation. Not a member of PEERS --
# comparing a $5/MTok model against a $1/MTok model as though they were
# alternatives for this job would be dishonest.
ANCHOR = "gateway/anthropic:claude-opus-5"

# Same models without the gateway, for anyone who already has provider keys.
DIRECT = {
    "gateway/anthropic:claude-haiku-4-5": "anthropic:claude-haiku-4-5",
    "gateway/openai:gpt-5.4-mini": "openai:gpt-5.4-mini",
    "gateway/google-cloud:gemini-3.7-flash": "google:gemini-3.7-flash",
    "gateway/anthropic:claude-opus-5": "anthropic:claude-opus-5",
}

# Approximate USD per million tokens (input, output), used only for the pre-run
# cost estimate. Deliberately rough. Published list prices drift, gateway
# billing may differ, and these are not billing figures. They exist so the CLI
# can refuse to spend real money silently.
PRICING = {
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "anthropic:claude-opus-5": (5.00, 25.00),
    "openai:gpt-5.4-mini": (0.25, 2.00),
    "google:gemini-3.7-flash": (0.30, 2.50),
}

# Rough per-run token cost, measured from real runs on 2026-08-19. The agent
# often calls lookup_policy more than once and each call returns full policy
# bodies, so input dominates. Rounded up -- an estimate that under-reports what
# a run costs is worse than useless.
EST_INPUT_TOKENS = 6_000
EST_OUTPUT_TOKENS = 450

# Which environment variable each provider needs, so we can fail before
# spending anything rather than deep inside an async gather.
API_KEY_ENV = {
    "gateway": "PYDANTIC_AI_GATEWAY_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google-cloud": "GOOGLE_API_KEY",
}


def provider_of(model: str) -> str:
    """Provider segment of a model string.

    'gateway/anthropic:claude-haiku-4-5' -> 'gateway'
    'anthropic:claude-haiku-4-5'         -> 'anthropic'
    """
    head = model.split(":", 1)[0]
    return head.split("/", 1)[0]


def bare_model(model: str) -> str:
    """Strip the gateway prefix so pricing lookups work for both forms."""
    return model.split("/", 1)[1] if model.startswith("gateway/") else model


def required_key(model: str) -> str | None:
    return API_KEY_ENV.get(provider_of(model))


def estimate_cost(model: str, n: int) -> float | None:
    """Rough USD estimate for n runs. None if we have no price on file."""
    key = bare_model(model)
    if key.startswith("google-cloud:"):
        key = key.replace("google-cloud:", "google:", 1)
    price = PRICING.get(key)
    if price is None:
        return None
    inp, out = price
    per_run = (EST_INPUT_TOKENS / 1e6) * inp + (EST_OUTPUT_TOKENS / 1e6) * out
    return per_run * n
