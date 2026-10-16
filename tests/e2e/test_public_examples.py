"""Execute representative offline programs exactly through their public entry points."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PublicExampleWorkflowTests(unittest.TestCase):
    def run_example(self, relative_path: str) -> str:
        output = StringIO()
        example = PROJECT_ROOT / "examples" / relative_path
        with patch("sys.argv", [str(example)]), redirect_stdout(output):
            runpy.run_path(str(example), run_name="__main__")
        return output.getvalue()

    def test_custom_components_example_runs_offline(self) -> None:
        output = self.run_example("extending/custom_components.py")
        self.assertIn("Custom answer", output)
        self.assertIn("custom-store", output)

    def test_registry_configuration_example_runs_offline(self) -> None:
        output = self.run_example("extending/registry_configuration.py")
        self.assertIn("Generated:", output)
        self.assertIn("Registry-backed context", output)

    def test_custom_agent_example_runs_offline(self) -> None:
        output = self.run_example("agentic/custom_agent.py")
        self.assertIn("The custom agent used this context", output)
        self.assertIn("custom-agent-example", output)


if __name__ == "__main__":
    unittest.main()
