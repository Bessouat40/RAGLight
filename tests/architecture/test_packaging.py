"""Packaging guardrails for the lightweight v4 core."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            cls.project = tomllib.load(project_file)["project"]

    def test_core_has_no_mandatory_third_party_dependencies(self):
        self.assertEqual([], self.project["dependencies"])

    def test_integrations_have_named_extras(self):
        extras = self.project["optional-dependencies"]

        self.assertEqual(
            {
                "agentic-mcp",
                "bedrock",
                "chroma",
                "documents",
                "gemini",
                "huggingface",
                "mistral",
                "ollama",
                "openai",
                "qdrant",
            },
            set(extras),
        )

    def test_mcp_sdk_is_owned_only_by_the_agentic_mcp_extra(self):
        extras = self.project["optional-dependencies"]

        self.assertTrue(
            any(
                requirement.startswith("mcp>=") for requirement in extras["agentic-mcp"]
            )
        )
        for name, requirements in extras.items():
            if name == "agentic-mcp":
                continue
            with self.subTest(extra=name):
                self.assertFalse(
                    any(
                        requirement.split(">", 1)[0] == "mcp"
                        for requirement in requirements
                    )
                )

    def test_removed_v3_applications_have_no_console_entry_point(self):
        self.assertNotIn("scripts", self.project)

    def test_vector_store_extras_do_not_bundle_document_processing(self):
        extras = self.project["optional-dependencies"]
        document_dependencies = {
            "langchain-core",
            "langchain-text-splitters",
            "pymupdf",
            "sentence-transformers",
        }

        for extra in ("chroma", "qdrant"):
            with self.subTest(extra=extra):
                dependency_names = {
                    requirement.split(">", 1)[0].split("=", 1)[0]
                    for requirement in extras[extra]
                }
                self.assertTrue(document_dependencies.isdisjoint(dependency_names))


if __name__ == "__main__":
    unittest.main()
