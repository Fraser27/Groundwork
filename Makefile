.PHONY: help setup test lint fmt run up down logs synth diff deploy install destroy clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CDK_DIR := cdk

# `docker compose` on modern Docker Desktop, `docker-compose` elsewhere (and on
# podman-backed setups, where the compose subcommand does not exist).
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install deps, install CDK node modules
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'
	cd $(CDK_DIR) && npm install
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

test: ## Run the test suite
	$(PY) -m pytest tests/ -q

lint: ## ruff + tsc. Both, because a synth failure is as blocking as a test failure.
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/ruff format --check src/ tests/
	cd $(CDK_DIR) && npx tsc --noEmit

fmt: ## Apply ruff formatting and autofixes
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

run: ## Run the API on the host against the compose Neo4j
	$(VENV)/bin/uvicorn src.api.app:app --reload --port 8000

up: ## Start Neo4j + the API in Docker
	$(COMPOSE) up -d --build
	@echo "API      http://localhost:8000"
	@echo "Neo4j    http://localhost:7474"

down: ## Stop the stack, keeping the Neo4j volume
	$(COMPOSE) down

logs: ## Tail the API logs
	$(COMPOSE) logs -f api

synth: ## Synthesise all six stacks. No AWS calls, no deploy.
	cd $(CDK_DIR) && npx cdk synth --quiet

diff: ## Show what a deploy would change
	cd $(CDK_DIR) && npx cdk diff

deploy: ## Deploy everything. Read cdk/README.md first — this bills continuously.
	cd $(CDK_DIR) && npx cdk deploy --all

install: ## First deploy into a fresh account, non-interactive. REGION=eu-west-1 to override.
	./scripts/deploy.sh

destroy: ## Tear down. The document bucket, its KMS key, and two tables SURVIVE by design.
	cd $(CDK_DIR) && npx cdk destroy --all
	@echo
	@echo "RETAINed and still billing: document bucket, its KMS key,"
	@echo "TenantTable, GrantTable, and the final Neptune snapshot."
	@echo "See 'Teardown gotchas' in cdk/README.md."

clean: ## Remove build artefacts and the Neo4j volume
	$(COMPOSE) down -v
	rm -rf $(CDK_DIR)/cdk.out .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
