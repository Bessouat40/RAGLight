"""Validate dependency rules that are enforceable in the current codebase."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "raglight"
DELIVERY_DIRECTORIES = {"api", "cli", "ui"}
FORBIDDEN_INTERNAL_PREFIXES = ("raglight.api", "raglight.cli", "raglight.ui")
FORBIDDEN_EXTERNAL_MODULES = {"fastapi", "streamlit"}
V4_APPLICATION_DIRECTORY = "pipeline"
FORBIDDEN_V4_APPLICATION_PREFIXES = (
    "raglight.agentic",
    "raglight.config",
    "raglight.cross_encoder",
    "raglight.document_processing",
    "raglight.embeddings",
    "raglight.llm",
    "raglight.rag",
    "raglight.scrapper",
    "raglight.vectorstore",
)


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Return line numbers and absolute module names imported by a syntax tree."""
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


def violations(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Find imports that make non-delivery code depend on delivery frameworks."""
    failures: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if relative.parts and relative.parts[0] in DELIVERY_DIRECTORIES:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, module in imported_modules(tree):
            internal_violation = module.startswith(FORBIDDEN_INTERNAL_PREFIXES)
            external_violation = module.split(".", 1)[0] in FORBIDDEN_EXTERNAL_MODULES
            if internal_violation or external_violation:
                failures.append(f"{relative}:{line}: forbidden dependency on {module}")
                continue

            is_v4_application = (
                relative.parts and relative.parts[0] == V4_APPLICATION_DIRECTORY
            )
            forbidden_application_dependency = module.startswith(
                FORBIDDEN_V4_APPLICATION_PREFIXES
            )
            if is_v4_application and forbidden_application_dependency:
                failures.append(
                    f"{relative}:{line}: v4 application dependency on {module}"
                )
    return failures


def main() -> int:
    failures = violations()
    if failures:
        print("Architecture violations:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Architecture checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
