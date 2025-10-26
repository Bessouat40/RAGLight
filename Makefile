install:
	uv pip install -r pyproject.toml
test:
	PYTHONPATH=src python3 -m unittest -v