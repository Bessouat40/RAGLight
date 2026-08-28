install:
	uv pip install -e ".[chroma,qdrant]"
test:
	uv sync --all-extras
	PYTHONPATH=src uv run python -m unittest -v

format:
	uv run black .
