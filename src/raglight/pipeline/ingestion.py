"""Provider-independent filesystem ingestion orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ..contracts import DocumentProcessor, IngestionVectorStore
from ..models import (
    Document,
    FolderSource,
    IngestionFailure,
    IngestionResult,
    IngestionSkip,
)


class DocumentProcessorRegistry(Protocol):
    """Resolve document processor factories by normalized file extension."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def create(self, name: str, /, **options: Any) -> DocumentProcessor: ...


def ingest_folders(
    sources: Sequence[FolderSource],
    *,
    vector_store: object,
    processors: DocumentProcessorRegistry | None,
) -> IngestionResult:
    """Discover, process, and persist files in deterministic order."""

    if not sources:
        raise ValueError("at least one folder source is required")
    if processors is None:
        raise RuntimeError("directory ingestion requires a document processor registry")
    if not isinstance(vector_store, IngestionVectorStore):
        raise TypeError(
            "directory ingestion requires a vector store with add_documents"
        )

    validated_sources = [(source, _validated_root(source.path)) for source in sources]
    available_extensions = set(processors.names)
    discovered_count = 0
    processed_count = 0
    indexed_count = 0
    skipped: list[IngestionSkip] = []
    failures: list[IngestionFailure] = []

    for source, root in validated_sources:
        discovered, discovery_failures = _discover_files(root, source)
        discovered_count += len(discovered) + len(discovery_failures)
        failures.extend(discovery_failures)

        for path in discovered:
            if path.is_symlink():
                skipped.append(IngestionSkip(path=str(path), reason="symlink"))
                continue

            extension = path.suffix.lower().removeprefix(".")
            if not extension or extension not in available_extensions:
                skipped.append(
                    IngestionSkip(path=str(path), reason="unsupported_file_type")
                )
                continue

            try:
                processor = processors.create(extension)
                processed = processor.process(
                    str(path),
                    chunk_size=source.chunk_size,
                    chunk_overlap=source.chunk_overlap,
                )
                chunks = _normalized_chunks(processed, path)
            except Exception as error:
                failures.append(
                    IngestionFailure(
                        path=str(path),
                        stage="processing",
                        error=_error_message(error),
                    )
                )
                continue

            try:
                vector_store.add_documents(chunks)
            except Exception as error:
                failures.append(
                    IngestionFailure(
                        path=str(path),
                        stage="indexing",
                        error=_error_message(error),
                    )
                )
                continue

            processed_count += 1
            indexed_count += len(chunks)

    return IngestionResult(
        discovered_count=discovered_count,
        processed_count=processed_count,
        indexed_count=indexed_count,
        skipped=tuple(skipped),
        failures=tuple(failures),
    )


def _validated_root(path: Path | str) -> Path:
    root = Path(path)
    if not root.exists():
        raise ValueError(f"folder source does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"folder source is not a directory: {root}")
    return root


def _discover_files(
    root: Path,
    source: FolderSource,
) -> tuple[list[Path], list[IngestionFailure]]:
    discovered: list[Path] = []
    failures: list[IngestionFailure] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda entry: (entry.name.casefold(), entry.name),
            )
        except OSError as error:
            failures.append(
                IngestionFailure(
                    path=str(directory),
                    stage="discovery",
                    error=_error_message(error),
                )
            )
            return

        for entry in entries:
            if entry.is_symlink():
                discovered.append(entry)
            elif entry.is_file():
                discovered.append(entry)
            elif (
                source.recursive
                and entry.is_dir()
                and entry.name not in source.ignored_directories
            ):
                visit(entry)

    visit(root)
    return discovered, failures


def _normalized_chunks(
    processed: dict[str, list[Document]],
    path: Path,
) -> list[Document]:
    if not isinstance(processed, dict):
        raise TypeError("document processor must return a dictionary")
    chunks = processed.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("document processor result must contain a 'chunks' list")

    normalized: list[Document] = []
    for chunk in chunks:
        if not isinstance(chunk, Document):
            raise TypeError("document processor chunks must be Document values")
        metadata = {
            key: value if isinstance(value, (str, int, float, bool)) else str(value)
            for key, value in chunk.metadata.items()
        }
        metadata["source"] = str(path)
        normalized.append(Document(page_content=chunk.page_content, metadata=metadata))
    return normalized


def _error_message(error: Exception) -> str:
    message = str(error).strip()
    return message or type(error).__name__
