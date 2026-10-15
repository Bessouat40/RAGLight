"""Unit tests for actionable optional-feature failures."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class OptionalDependencyTests(unittest.TestCase):
    def test_cli_names_the_extra_when_cli_dependencies_are_missing(self):
        from raglight import _entrypoints

        with patch(
            "builtins.__import__",
            side_effect=ModuleNotFoundError("No module named 'typer'", name="typer"),
        ):
            with self.assertRaisesRegex(
                ImportError, r"pip install ['\"]raglight\[cli\]['\"]"
            ):
                _entrypoints.cli()

    def test_error_preserves_unrelated_import_failures(self):
        from raglight._optional import optional_import_error

        cause = ModuleNotFoundError("No module named 'unrelated'", name="unrelated")

        with self.assertRaises(ModuleNotFoundError) as raised:
            optional_import_error(
                cause,
                dependency="typer",
                extra="cli",
                feature="RAGLight CLI",
            )

        self.assertIs(cause, raised.exception)


if __name__ == "__main__":
    unittest.main()
