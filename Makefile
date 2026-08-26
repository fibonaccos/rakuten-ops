.DEFAULT_GOAL := help
.PHONY: help install lint test test-api test-inference test-ui test-ml test-stack up down logs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install every dependency group into .venv
	uv sync --all-groups

lint:  ## Run ruff over the repository
	uv run --group dev ruff check .

# Each service ships its own image with its own requirements, and their modules
# share names (main, routes, services), so each suite gets its own process.
test: test-stack test-api test-inference test-ui test-ml  ## Run every suite

test-stack:  ## Check docker-compose.yaml against .env.example
	uv run --group dev pytest tests/stack -q

test-api:  ## Test the gateway API
	uv run --group api --group dev pytest tests/api -q

test-inference:  ## Test the inference service
	uv run --group inference --group dev pytest tests/inference -q

test-ui:  ## Test the Streamlit front-end
	uv run --group streamlit --group dev pytest tests/streamlit -q

test-ml:  ## Test the model pipeline and its parity with training
	uv run --group ml --group dev pytest tests/model tests/ml -q

up:  ## Start the stack
	docker compose up -d

down:  ## Stop the stack and drop its volumes
	docker compose down -v

logs:  ## Follow the logs of every service
	docker compose logs -f
