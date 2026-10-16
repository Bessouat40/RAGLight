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
    def test_current_package_respects_v4_boundaries(self):
        self.assertEqual([], CHECK_ARCHITECTURE.violations())

    def test_pipeline_cannot_import_a_concrete_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            pipeline = package / "pipeline"
            pipeline.mkdir()
            (pipeline / "pipeline.py").write_text(
                "from raglight.integrations.ollama import OllamaLLM\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "pipeline/pipeline.py:1: v4 application dependency on "
                    "raglight.integrations.ollama"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_classic_pipeline_cannot_depend_on_agentic_runtime(self):
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

    def test_agentic_runtime_cannot_import_a_concrete_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            agentic = package / "agentic"
            agentic.mkdir()
            (agentic / "pipeline.py").write_text(
                "from raglight.integrations.ollama import OllamaLLM\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "agentic/pipeline.py:1: v4 application dependency on "
                    "raglight.integrations.ollama"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_public_contracts_cannot_import_concrete_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            contracts = package / "contracts"
            contracts.mkdir()
            (contracts / "__init__.py").write_text(
                "from raglight.integrations.openai import OpenAILLM\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "contracts/__init__.py:1: core dependency on "
                    "raglight.integrations.openai"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_public_contracts_cannot_hide_provider_behind_relative_import(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            contracts = package / "contracts"
            contracts.mkdir()
            (contracts / "__init__.py").write_text(
                "from ..integrations.openai import OpenAILLM\n",
                encoding="utf-8",
            )

            self.assertEqual(
                [
                    "contracts/__init__.py:1: core dependency on "
                    "raglight.integrations.openai"
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
                    "contracts/__init__.py:1: core dependency on "
                    "langchain_core.documents"
                ],
                CHECK_ARCHITECTURE.violations(package),
            )

    def test_registry_primitives_cannot_import_concrete_integrations(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "registry.py").write_text(
                "from raglight.integrations.openai import OpenAILLM\n",
                encoding="utf-8",
            )

            self.assertEqual(
                ["registry.py:1: core dependency on " "raglight.integrations.openai"],
                CHECK_ARCHITECTURE.violations(package),
            )


if __name__ == "__main__":
    unittest.main()
