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

    def test_removed_v3_applications_have_no_console_entry_point(self):
        self.assertNotIn("scripts", self.project)


if __name__ == "__main__":
    unittest.main()
