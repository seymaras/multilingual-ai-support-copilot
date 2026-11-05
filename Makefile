# Short commands for everyday use. Run `make` to list them.
#
# Every target uses the project's virtualenv directly, so there is no need to activate it.
# PYTHONPATH=src makes the package importable even when the editable install's .pth file is
# skipped (macOS can mark it hidden, and Python then ignores it).

PYTHON ?= .venv/bin/python
RUN := PYTHONPATH="$(CURDIR)/src" TOKENIZERS_PARALLELISM=false $(PYTHON)

.DEFAULT_GOAL := help
.PHONY: help start ask demo test test-slow lint format typecheck check

help:
	@echo "make start               ask questions interactively (Ollama must be running)"
	@echo "make ask Q=\"question\"    answer one question"
	@echo "make demo                run the sample questions in data/demo_questions.json"
	@echo "make test                fast test suite (no models, no Ollama)"
	@echo "make test-slow           tests with real models and a live Ollama"
	@echo "make check               lint, format check, type check and fast tests"
	@echo "make format              apply ruff formatting and safe lint fixes"

start: $(PYTHON)
	@$(RUN) -m multilingual_support_copilot

ask: $(PYTHON)
	@test -n "$(Q)" || { echo 'Usage: make ask Q="İade için kaç günüm var?"'; exit 2; }
	@$(RUN) -m multilingual_support_copilot "$(Q)"

demo: $(PYTHON)
	@$(RUN) -m multilingual_support_copilot --demo

test: $(PYTHON)
	@$(RUN) -m pytest -q

test-slow: $(PYTHON)
	@$(RUN) -m pytest -q -m slow

lint: $(PYTHON)
	@$(PYTHON) -m ruff check src tests
	@$(PYTHON) -m ruff format --check src tests

format: $(PYTHON)
	@$(PYTHON) -m ruff check --fix src tests
	@$(PYTHON) -m ruff format src tests

typecheck: $(PYTHON)
	@$(RUN) -m mypy

check: lint typecheck test

$(PYTHON):
	@echo "No virtualenv at .venv. Create it with:"
	@echo "  python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
	@exit 1
