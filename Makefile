install:
	uv pip install -e ".[agentic,bedrock,chroma,cli,documents,gemini,huggingface,langfuse,legacy,mistral,ollama,openai,qdrant,server,ui]"

install-dev:
	uv sync --all-extras --group dev

test:
	$(MAKE) test-unit
	$(MAKE) test-integration
	$(MAKE) test-e2e

test-unit:
	PYTHONPATH=src uv run --all-extras python scripts/run_test_tier.py unit

test-integration:
	PYTHONPATH=src uv run --all-extras python scripts/run_test_tier.py integration

test-e2e:
	PYTHONPATH=src uv run --all-extras python scripts/run_test_tier.py e2e

architecture-check:
	PYTHONPATH=src uv run python scripts/check_architecture.py

format:
	uv run black .

format-check:
	uv run black --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy scripts src/raglight/pipeline

package-check:
	uv run python scripts/check_wheel.py

check: format-check lint typecheck architecture-check test package-check
