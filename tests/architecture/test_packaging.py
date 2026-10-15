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

    def test_optional_features_have_named_extras(self):
        extras = self.project["optional-dependencies"]

        self.assertEqual(
            {
                "agentic",
                "bedrock",
                "chroma",
                "cli",
                "documents",
                "gemini",
                "huggingface",
                "langfuse",
                "legacy",
                "mistral",
                "ollama",
                "openai",
                "qdrant",
                "server",
                "ui",
            },
            set(extras),
        )

    def test_cli_entry_point_uses_lightweight_dependency_guard(self):
        self.assertEqual(
            {"raglight": "raglight._entrypoints:cli"}, self.project["scripts"]
        )


if __name__ == "__main__":
    unittest.main()
