"""Validate the dependency direction of the v4 package."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "raglight"
V4_APPLICATION_DIRECTORIES = {"agentic", "pipeline"}
CORE_DIRECTORIES = {"contracts", "models"}
CORE_FILES = {"registry.py"}
FORBIDDEN_APPLICATION_PREFIXES = {
    "agentic": ("raglight.integrations", "raglight.pipeline"),
    "pipeline": ("raglight.agentic", "raglight.integrations"),
}
FORBIDDEN_CORE_PREFIXES = (
    "raglight.agentic",
    "raglight.integrations",
    "raglight.pipeline",
)
OPTIONAL_SDK_MODULES = {
    "boto3",
    "chromadb",
    "langchain",
    "langchain_aws",
    "langchain_core",
    "langchain_google_genai",
    "langchain_mistralai",
    "langchain_ollama",
    "langchain_openai",
    "qdrant_client",
    "sentence_transformers",
}


def imported_modules(tree: ast.AST, relative: Path) -> list[tuple[int, str]]:
    """Return line numbers and absolute module names imported by a syntax tree."""
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:
                module = node.module
            else:
                package = ("raglight", *relative.parts[:-1])
                ancestor_count = max(0, len(package) - node.level + 1)
                module = ".".join((*package[:ancestor_count], *node.module.split(".")))
            imports.append((node.lineno, module))
    return imports


def violations(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Find imports that make the core depend on optional integrations."""
    failures: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, module in imported_modules(tree, relative):
            application = relative.parts[0] if relative.parts else ""
            is_v4_application = application in V4_APPLICATION_DIRECTORIES
            forbidden_application_dependency = module.startswith(
                FORBIDDEN_APPLICATION_PREFIXES.get(application, ())
            )
            if is_v4_application and forbidden_application_dependency:
                failures.append(
                    f"{relative}:{line}: v4 application dependency on {module}"
                )
                continue

            is_core = bool(
                relative.parts
                and (
                    relative.parts[0] in CORE_DIRECTORIES
                    or relative.as_posix() in CORE_FILES
                )
            )
            forbidden_core_dependency = module.startswith(FORBIDDEN_CORE_PREFIXES)
            forbidden_core_sdk = module.split(".", 1)[0] in OPTIONAL_SDK_MODULES
            if is_core and (forbidden_core_dependency or forbidden_core_sdk):
                failures.append(f"{relative}:{line}: core dependency on {module}")
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
