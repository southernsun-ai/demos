"""FastAPI server: streams the agent's answer over Server-Sent Events.

Run:  ANTHROPIC_API_KEY=... uvicorn server:app --port 8000
Open: http://localhost:8000

Each request can pick a mode (typed | sql | discover), a model (haiku | sonnet | opus or
any 'provider:model'), and an effort. Defaults come from config.py / env vars.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings

import config
from agent import Deps, build_agent
from runner import stream_turn
from store import ModLogStore

TSV_PATH = os.getenv("MOD_LOG_TSV", str(Path(__file__).with_name("mod_channel_15k.tsv")))
MAX_HISTORY_MESSAGES = 40      # bounded so long chats don't grow prefill cost forever
MAX_CONVERSATIONS = 500
OFFLINE = os.getenv("MOD_CHAT_OFFLINE") == "1"

log = logging.getLogger("mod_chat")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class State:
    store: ModLogStore
    deps: Deps
    agents: dict[tuple[str, str], Agent] = {}
    offline_models: dict = {}
    conversations: "OrderedDict[str, list[ModelMessage]]" = OrderedDict()


state = State()


def get_agent(mode: str, effort: str) -> Agent:
    """Agents are model-agnostic and cheap to build; one per (mode, effort), built on demand."""
    key = (mode, effort)
    if key not in state.agents:
        state.agents[key] = build_agent(mode, state.store, effort=effort)
    return state.agents[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    state.store = ModLogStore.load(TSV_PATH)
    state.deps = Deps(store=state.store)
    for mode in config.MODES:
        get_agent(mode, config.DEFAULT_EFFORT)
    if OFFLINE:
        from eval_cases import build_cases
        from fake_model import offline_model_for

        cases = build_cases(TSV_PATH)
        state.offline_models = {m: offline_model_for(m, cases) for m in config.MODES}
    log.info(json.dumps({"event": "startup", "load_ms": round(1000 * (time.perf_counter() - t)),
                         "default_model": config.resolve_model(), "effort": config.DEFAULT_EFFORT}))

    # Warmup: opens the pooled HTTPS connection (TLS handshake off the hot path) and writes
    # each mode's static prefix into the prompt cache for the default model.
    if not OFFLINE and os.getenv("MOD_CHAT_WARMUP", "1") == "1" and os.getenv("ANTHROPIC_API_KEY"):
        t = time.perf_counter()
        try:
            for mode in config.MODES:
                # Same settings as real turns (thinking params are part of the cache key).
                await get_agent(mode, config.DEFAULT_EFFORT).run(
                    "Reply with OK.", deps=state.deps, model=config.make_model(),
                    model_settings=ModelSettings(max_tokens=512))
            log.info(json.dumps({"event": "warmup", "ms": round(1000 * (time.perf_counter() - t))}))
        except Exception as e:  # warmup must never block startup
            log.warning(json.dumps({"event": "warmup_failed", "error": str(e)}))
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    mode: config.Mode = "typed"
    model: str | None = None      # alias or 'provider:model'; default from config
    effort: str | None = None     # minimal | low | medium | high | xhigh | on | off


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        model = state.offline_models[req.mode] if OFFLINE else config.make_model(req.model)
        effort = req.effort or config.DEFAULT_EFFORT
        config.parse_effort(effort)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    agent = get_agent(req.mode, effort)
    conv_id = req.conversation_id or uuid.uuid4().hex
    model_label = getattr(model, "model_name", model)
    hist_key = f"{req.mode}:{model_label}:{conv_id}"   # tool vocabularies differ per mode
    history = state.conversations.get(hist_key, [])

    async def stream():
        yield sse("start", {"conversation_id": conv_id})
        async for kind, payload in stream_turn(
            agent, req.message, deps=state.deps, model=model,
            message_history=history, usage_limits=config.MODE_LIMITS[req.mode],
        ):
            if kind != "done":
                yield sse("status" if kind == "thinking" else kind, payload)
                continue
            m = payload["metrics"]
            state.conversations[hist_key] = payload["messages"][-MAX_HISTORY_MESSAGES:]
            state.conversations.move_to_end(hist_key)
            while len(state.conversations) > MAX_CONVERSATIONS:
                state.conversations.popitem(last=False)
            metrics = {"mode": req.mode, "model": model_label, "effort": effort,
                       **{k: v for k, v in asdict(m).items() if k not in ("answer", "sql")}}
            log.info(json.dumps({"event": "turn", "conversation_id": conv_id, **metrics}))
            if m.error:
                yield sse("error", {"message": m.error})
            yield sse("done", metrics)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no proxy buffering
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path(__file__).with_name("index.html").read_text()
