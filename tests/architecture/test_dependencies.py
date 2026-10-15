import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py"
SPEC = importlib.util.spec_from_file_location("check_architecture", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load architecture checker from {SCRIPT_PATH}")
CHECK_ARCHITECTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_ARCHITECTURE)


class ArchitectureDependencyTests(unittest.TestCase):
    def test_current_package_respects_delivery_boundary(self):
        self.assertEqual([], CHECK_ARCHITECTURE.violations())

    def test_core_importing_fastapi_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "core.py").write_text(
                "from fastapi import FastAPI\n", encoding="utf-8"
            )

            self.assertEqual(
                ["core.py:1: forbidden dependency on fastapi"],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_delivery_code_may_import_delivery_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            api = package / "api"
            api.mkdir()
            (api / "app.py").write_text(
                "from fastapi import FastAPI\n", encoding="utf-8"
            )

            self.assertEqual([], CHECK_ARCHITECTURE.violations(package))

    def test_v4_pipeline_cannot_import_legacy_provider_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            pipeline = package / "pipeline"
            pipeline.mkdir()
            (pipeline / "pipeline.py").write_text(
                "from raglight.llm.ollama_model import OllamaModel\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "pipeline/pipeline.py:1: v4 application dependency on "
                    "raglight.llm.ollama_model"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_standard_pipeline_cannot_import_agentic_package(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            pipeline = package / "pipeline"
            pipeline.mkdir()
            (pipeline / "pipeline.py").write_text(
                "from raglight.agentic import AgenticPipeline\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "pipeline/pipeline.py:1: v4 application dependency on "
                    "raglight.agentic"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_public_contracts_cannot_import_concrete_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            contracts = package / "contracts"
            contracts.mkdir()
            (contracts / "__init__.py").write_text(
                "from raglight.llm.openai_model import OpenAIModel\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "contracts/__init__.py:1: public contract dependency on "
                    "raglight.llm.openai_model"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_public_contracts_cannot_hide_provider_behind_relative_import(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            contracts = package / "contracts"
            contracts.mkdir()
            (contracts / "__init__.py").write_text(
                "from ..llm.openai_model import OpenAIModel\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "contracts/__init__.py:1: public contract dependency on "
                    "raglight.llm.openai_model"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_public_contracts_cannot_import_optional_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            contracts = package / "contracts"
            contracts.mkdir()
            (contracts / "__init__.py").write_text(
                "from langchain_core.documents import Document\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "contracts/__init__.py:1: public contract dependency on "
                    "langchain_core.documents"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )


if __name__ == "__main__":
    unittest.main()
