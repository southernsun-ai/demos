"""The one place to change models and effort.

Model: set MOD_CHAT_MODEL to an alias below or any PydanticAI model string
       ("anthropic:claude-sonnet-5-5", "openai:gpt-5.6", ...).
Effort: set MOD_CHAT_EFFORT to minimal | low | medium | high | xhigh | on | off.

On the Claude 5.5 models, thinking is always adaptive. The effort level is sent as
`effort`, and Sonnet/Opus 5.5 can't turn thinking off ("off" is ignored there).

Gateway: set PYDANTIC_AI_GATEWAY_API_KEY to route Anthropic calls through the
         Pydantic AI Gateway instead of directly via ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.usage import UsageLimits

MODEL_ALIASES: dict[str, str] = {
    "haiku": "anthropic:claude-haiku-5-5",
    "sonnet": "anthropic:claude-sonnet-5-5",
    "opus": "anthropic:claude-opus-5-5",
}

DEFAULT_MODEL = os.getenv("MOD_CHAT_MODEL", "haiku")
DEFAULT_EFFORT = os.getenv("MOD_CHAT_EFFORT", "medium")

# Answers are short. Thinking tokens count toward max_tokens, so leave headroom for them.
MAX_TOKENS = int(os.getenv("MOD_CHAT_MAX_TOKENS", "4096"))

Mode = Literal["typed", "sql", "discover"]
MODES: tuple[Mode, ...] = ("typed", "sql", "discover")

# Hard ceilings per mode. Discovery needs more round trips by design.
MODE_LIMITS: dict[str, UsageLimits] = {
    "typed": UsageLimits(request_limit=4, tool_calls_limit=6),
    "sql": UsageLimits(request_limit=6, tool_calls_limit=8),
    "discover": UsageLimits(request_limit=8, tool_calls_limit=10),
}


def resolve_model(name: str | None = None) -> str:
    """Alias or full 'provider:model' string -> full model string."""
    name = (name or DEFAULT_MODEL).strip()
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if ":" not in name:
        raise ValueError(f"Unknown model {name!r}. Use one of {sorted(MODEL_ALIASES)} or 'provider:model'.")
    return name


def parse_effort(value: str | bool | None = None) -> ThinkingLevel:
    v = DEFAULT_EFFORT if value is None else value
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("off", "false", "none", "0"):
        return False
    if v in ("on", "true", "default"):
        return True
    if v in ("minimal", "low", "medium", "high", "xhigh"):
        return v  # type: ignore[return-value]
    raise ValueError(f"Unknown effort {value!r}")


def parse_model_spec(spec: str, default_effort: str) -> tuple[str, str, ThinkingLevel]:
    """'sonnet' or 'haiku@off' -> (label, model string, effort)."""
    name, _, effort = spec.partition("@")
    effort = effort or default_effort
    return f"{name}@{effort}", resolve_model(name), parse_effort(effort)


_GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://gateway-us.pydantic.dev/proxy")


def make_model(name: str | None = None):
    """Return the model to pass to run_stream_events / agent.run.

    When PYDANTIC_AI_GATEWAY_API_KEY is set, Anthropic models are routed through
    the Pydantic AI Gateway (same key your other services use). Otherwise falls
    back to the resolved model string, which PydanticAI uses with ANTHROPIC_API_KEY.
    Non-Anthropic model strings are always returned as-is.
    """
    model_str = resolve_model(name)
    key = os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")
    if not key or not model_str.startswith("anthropic:"):
        return model_str
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.gateway import gateway_provider

    model_id = model_str.split(":", 1)[1]  # "anthropic:claude-haiku-5-5" -> "claude-haiku-5-5"
    provider = gateway_provider(
        "anthropic",
        api_key=key,
        base_url=_GATEWAY_BASE_URL,
    )
    return AnthropicModel(model_id, provider=provider)
