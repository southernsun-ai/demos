# One command per task. See AGENTS.md for when to use each.
-include .env
export

PY      := .venv/bin/python
MODELS  ?= haiku,sonnet,opus
MODES   ?= typed,sql,discover
REPEATS ?= 3
EFFORT  ?= medium
CASES   ?=
PORT    ?= 8000

.PHONY: setup test smoke serve serve-offline eval eval-quick bench clean check-key

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "Setup done. Next: cp .env.example .env and add ANTHROPIC_API_KEY"

test:
	PYDANTIC_AI_NO_BANNER=1 $(PY) -m pytest -q

smoke:
	PYDANTIC_AI_NO_BANNER=1 $(PY) eval_modes.py --offline --repeats 1 | head -20

check-key:
	@test -n "$$PYDANTIC_AI_GATEWAY_API_KEY" -o -n "$$ANTHROPIC_API_KEY" || \
		(echo "Set PYDANTIC_AI_GATEWAY_API_KEY (gateway) or ANTHROPIC_API_KEY (direct) in .env" && exit 1)

serve: check-key
	PYDANTIC_AI_NO_BANNER=1 $(PY) -m uvicorn server:app --port $(PORT)

serve-offline:
	MOD_CHAT_OFFLINE=1 PYDANTIC_AI_NO_BANNER=1 $(PY) -m uvicorn server:app --port $(PORT)

eval: check-key
	PYDANTIC_AI_NO_BANNER=1 $(PY) eval_modes.py --models $(MODELS) --modes $(MODES) \
		--repeats $(REPEATS) --effort $(EFFORT) $(if $(CASES),--cases $(CASES),)

eval-quick:
	$(MAKE) eval REPEATS=1

bench:
	$(PY) bench.py --url http://localhost:$(PORT) --runs 20

clean:
	rm -rf .venv results __pycache__ .pytest_cache server.log
