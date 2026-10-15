"""Helpers for consistent, actionable optional-dependency failures."""

from __future__ import annotations

from typing import NoReturn


def optional_import_error(
    cause: ModuleNotFoundError,
    *,
    dependency: str | tuple[str, ...],
    extra: str,
    feature: str,
) -> NoReturn:
    """Raise an actionable error when an optional dependency is unavailable."""
    dependencies = (dependency,) if isinstance(dependency, str) else dependency
    missing = cause.name or ""
    if not any(
        missing == candidate or missing.startswith(f"{candidate}.")
        for candidate in dependencies
    ):
        raise cause

    raise ImportError(
        f"{feature} requires optional dependencies. "
        f"Install them with: pip install 'raglight[{extra}]'"
    ) from cause
