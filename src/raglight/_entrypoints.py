"""Console entry points that keep optional delivery dependencies isolated."""

from __future__ import annotations

from ._optional import optional_import_error


def cli() -> None:
    """Run the optional CLI or explain how to install it."""
    try:
        from .cli.main import app
    except ModuleNotFoundError as error:
        optional_import_error(
            error,
            dependency=error.name or "",
            extra="cli",
            feature="RAGLight CLI",
        )

    app()
