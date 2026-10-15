"""Install the freshly built wheel in isolation and verify the public package."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to validate the package.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="raglight-package-check-") as directory:
        temporary_root = Path(directory)
        artifacts = temporary_root / "dist"
        subprocess.run(
            [uv, "build", "--out-dir", str(artifacts)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        wheels = sorted(artifacts.glob("raglight-*.whl"))
        if not wheels:
            print("The package build produced no wheel.", file=sys.stderr)
            return 1

        environment = temporary_root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheels[-1])],
            check=True,
        )
        core_check = """
from importlib.metadata import requires
from pathlib import Path
import sys

import raglight
from raglight import PipelineConfig, RAGPipeline
from raglight.contracts import EmbeddingModel, LLM, VectorStore
from raglight.integrations import default_registries
from raglight.models import ComponentConfig, Document, ExecutionEvent, RAGResponse
from raglight.pipeline import ConversationSession

assert raglight.__all__ == ["PipelineConfig", "RAGPipeline"]
assert Path(raglight.__file__).with_name("py.typed").is_file()
assert PipelineConfig(top_k=1).top_k == 1
assert ComponentConfig(name="CUSTOM").name == "custom"
assert Document(page_content="test").page_content == "test"
event = ExecutionEvent(stage="retrieval", status="completed", duration_ms=0)
response = RAGResponse(answer="test", trace_id="trace", events=[event])
assert response.to_dict()["events"][0]["stage"] == "retrieval"
assert ConversationSession is not None
assert "ollama" in default_registries.llms.names
assert all(contract is not None for contract in (EmbeddingModel, LLM, VectorStore))
unconditional = [item for item in requires("raglight") or [] if "extra ==" not in item]
assert not unconditional, unconditional
forbidden = ("fastapi", "langchain", "streamlit", "chromadb", "qdrant_client")
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, loaded
"""
        subprocess.run(
            [str(python), "-c", core_check],
            check=True,
        )

        provider = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from raglight.integrations import default_registries; "
                    "default_registries.llms.create('ollama', model_name='test')"
                ),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if provider.returncode == 0 or "raglight[ollama]" not in provider.stderr:
            print(provider.stderr, file=sys.stderr)
            print("Missing Ollama dependencies were not actionable.", file=sys.stderr)
            return 1

    print(f"Validated wheel: {wheels[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
