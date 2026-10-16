"""Provider-independent models for filesystem ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_IGNORED_DIRECTORIES = (
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
)


@dataclass(frozen=True, slots=True)
class FolderSource:
    """Describe one directory tree to discover and process."""

    path: Path | str
    recursive: bool = True
    ignored_directories: tuple[str, ...] = DEFAULT_IGNORED_DIRECTORIES
    chunk_size: int = 2500
    chunk_overlap: int = 250

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "ignored_directories", tuple(self.ignored_directories))
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if any(
            not name or Path(name).name != name for name in self.ignored_directories
        ):
            raise ValueError("ignored directories must be non-empty directory names")


@dataclass(frozen=True, slots=True)
class IngestionSkip:
    """Explain why one discovered filesystem entry was not processed."""

    path: str
    reason: Literal["unsupported_file_type", "symlink"]

    def to_dict(self) -> dict[str, str]:
        """Return a transport-friendly representation."""

        return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    """Describe a per-file failure without discarding successful files."""

    path: str
    stage: Literal["discovery", "processing", "indexing"]
    error: str

    def to_dict(self) -> dict[str, str]:
        """Return a transport-friendly representation."""

        return {"path": self.path, "stage": self.stage, "error": self.error}


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summarize a completed ingestion operation."""

    discovered_count: int
    processed_count: int
    indexed_count: int
    skipped: tuple[IngestionSkip, ...] = field(default_factory=tuple)
    failures: tuple[IngestionFailure, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        counts = (self.discovered_count, self.processed_count, self.indexed_count)
        if any(count < 0 for count in counts):
            raise ValueError("ingestion counts must be non-negative")
        classified_count = self.processed_count + self.skipped_count + self.failed_count
        if self.discovered_count != classified_count:
            raise ValueError(
                "discovered_count must equal processed, skipped, and failed entries"
            )

    @property
    def skipped_count(self) -> int:
        """Return the number of explicitly skipped entries."""

        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        """Return the number of failed entries."""

        return len(self.failures)

    @property
    def skipped_files(self) -> tuple[str, ...]:
        """Return skipped paths in deterministic discovery order."""

        return tuple(skipped.path for skipped in self.skipped)

    def to_dict(self) -> dict[str, object]:
        """Return a transport-friendly representation for delivery layers."""

        return {
            "discovered_count": self.discovered_count,
            "processed_count": self.processed_count,
            "indexed_count": self.indexed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "skipped": [entry.to_dict() for entry in self.skipped],
            "failures": [failure.to_dict() for failure in self.failures],
        }
