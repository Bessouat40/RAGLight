"""Guardrails for the public, copy-paste example suite."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
EXAMPLE_PATHS = (
    "classic/directory_ingestion.py",
    "classic/local_ollama.py",
    "classic/openai.py",
    "classic/openai_compatible_endpoint.py",
    "agentic/custom_agent.py",
    "agentic/mcp.py",
    "agentic/local_ollama.py",
    "extending/custom_components.py",
    "extending/registry_configuration.py",
)


class ExampleSuiteTests(unittest.TestCase):
    def test_documented_examples_exist_and_compile(self) -> None:
        for relative_path in EXAMPLE_PATHS:
            with self.subTest(example=relative_path):
                example = EXAMPLES_ROOT / relative_path
                source = example.read_text(encoding="utf-8")
                compile(source, str(example), "exec")

    def test_executable_examples_have_main_entry_points(self) -> None:
        for relative_path in EXAMPLE_PATHS:
            with self.subTest(example=relative_path):
                example = EXAMPLES_ROOT / relative_path
                tree = ast.parse(example.read_text(encoding="utf-8"))
                function_names = {
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                self.assertIn("main", function_names)
                self.assertIn('if __name__ == "__main__":', example.read_text())

    def test_examples_do_not_contain_local_paths_or_credentials(self) -> None:
        forbidden_fragments = (
            "/Users/",
            "/home/",
            "/absolute/path/",
            "sk-proj-",
            "sk_live_",
            "gpt-4.1-mini",
        )
        for relative_path in EXAMPLE_PATHS:
            with self.subTest(example=relative_path):
                source = (EXAMPLES_ROOT / relative_path).read_text(encoding="utf-8")
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, source)

    def test_examples_do_not_shadow_runtime_packages(self) -> None:
        shadowed_package_names = {"ollama.py"}

        for relative_path in EXAMPLE_PATHS:
            with self.subTest(example=relative_path):
                self.assertNotIn(Path(relative_path).name, shadowed_package_names)

    def test_examples_index_covers_every_canonical_example(self) -> None:
        index = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")
        for relative_path in EXAMPLE_PATHS:
            with self.subTest(example=relative_path):
                self.assertIn(relative_path, index)

        for extra in ("ollama", "openai", "chroma", "documents", "agentic-mcp"):
            with self.subTest(extra=extra):
                self.assertIn(extra, index)

    def test_root_readme_links_the_examples_index(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("examples/README.md", readme)


if __name__ == "__main__":
    unittest.main()
