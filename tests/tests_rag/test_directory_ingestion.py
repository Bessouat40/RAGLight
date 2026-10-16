"""Unit tests for provider-independent directory ingestion."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from raglight import RAGPipeline
from raglight.integrations import IntegrationRegistry
from raglight.models import Document, FolderSource, IngestionFailure


class StubLLM:
    def generate(self, input):
        return str(input["question"])

    def generate_streaming(self, input):
        yield self.generate(input)


class StubEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


class RecordingStore:
    def __init__(self, *, rejected_content: str | None = None) -> None:
        self.documents: list[Document] = []
        self.calls: list[list[Document]] = []
        self.rejected_content = rejected_content

    def add_documents(self, documents):
        copied = list(documents)
        self.calls.append(copied)
        if self.rejected_content and any(
            document.page_content == self.rejected_content for document in copied
        ):
            raise RuntimeError("store rejected document")
        self.documents.extend(copied)

    def similarity_search(self, question, k=5):
        return self.documents[:k]


class AlternateRecordingStore(RecordingStore):
    """A second structurally compatible store with no shared base requirement."""


class RecordingProcessor:
    def __init__(self, *, failing_names: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.failing_names = failing_names

    def process(self, file_path, chunk_size, chunk_overlap):
        self.calls.append((file_path, chunk_size, chunk_overlap))
        path = Path(file_path)
        if path.name in self.failing_names:
            raise RuntimeError("processor could not parse file")
        return {
            "chunks": [
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={
                        "source": "processor-supplied",
                        "tags": ["custom", path.suffix],
                    },
                )
            ],
            "classes": [],
        }


def processor_registry(**processors):
    registry = IntegrationRegistry("document processor")
    for extension, processor in processors.items():
        registry.register(extension, lambda processor=processor: processor)
    return registry


def build_pipeline(store, processors=None):
    return RAGPipeline(
        llm=StubLLM(),
        embeddings=StubEmbeddings(),
        vector_store=store,
        document_processors=processors,
    )


class FolderSourceTests(unittest.TestCase):
    def test_defaults_and_paths_are_normalized(self):
        source = FolderSource(path="knowledge")

        self.assertEqual(Path("knowledge"), source.path)
        self.assertTrue(source.recursive)
        self.assertIn(".git", source.ignored_directories)
        self.assertEqual(2500, source.chunk_size)
        self.assertEqual(250, source.chunk_overlap)

    def test_chunk_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "chunk_size must be greater than zero"):
            FolderSource(path="knowledge", chunk_size=0)

        with self.assertRaisesRegex(ValueError, "chunk_overlap must be non-negative"):
            FolderSource(path="knowledge", chunk_overlap=-1)

        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            FolderSource(path="knowledge", chunk_size=10, chunk_overlap=10)


class DirectoryIngestionTests(unittest.TestCase):
    def test_recursive_ingestion_is_deterministic_and_reports_skipped_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledge"
            root.mkdir()
            (root / "b.txt").write_text("second", encoding="utf-8")
            (root / "a.txt").write_text("first", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.md").write_text("third", encoding="utf-8")
            (nested / "unsupported.bin").write_bytes(b"ignored")
            ignored = root / "private"
            ignored.mkdir()
            (ignored / "secret.txt").write_text("secret", encoding="utf-8")

            processor = RecordingProcessor()
            store = RecordingStore()
            pipeline = build_pipeline(
                store,
                processor_registry(txt=processor, md=processor),
            )
            result = pipeline.ingest(
                [FolderSource(path=root, ignored_directories=("private",))]
            )

        self.assertEqual(4, result.discovered_count)
        self.assertEqual(3, result.processed_count)
        self.assertEqual(3, result.indexed_count)
        self.assertEqual(1, result.skipped_count)
        self.assertEqual(0, result.failed_count)
        self.assertEqual(
            ["a.txt", "b.txt", "c.md"],
            [Path(call[0]).name for call in processor.calls],
        )
        self.assertEqual(
            (str(nested / "unsupported.bin"),),
            result.skipped_files,
        )
        self.assertEqual(3, len(store.calls))

    def test_non_recursive_source_only_processes_root_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "root.txt").write_text("root", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "nested.txt").write_text("nested", encoding="utf-8")
            processor = RecordingProcessor()
            pipeline = build_pipeline(
                RecordingStore(), processor_registry(txt=processor)
            )

            result = pipeline.ingest([FolderSource(path=root, recursive=False)])

        self.assertEqual(1, result.discovered_count)
        self.assertEqual(["root.txt"], [Path(call[0]).name for call in processor.calls])

    def test_symlinks_are_not_followed_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "knowledge"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            link = root / "outside-link.txt"
            try:
                link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")
            processor = RecordingProcessor()
            pipeline = build_pipeline(
                RecordingStore(), processor_registry(txt=processor)
            )

            result = pipeline.ingest([FolderSource(path=root)])

        self.assertEqual(1, result.discovered_count)
        self.assertEqual(0, result.processed_count)
        self.assertEqual((str(link),), result.skipped_files)
        self.assertEqual([], processor.calls)

    def test_validation_rejects_missing_and_non_directory_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "document.txt"
            file_path.write_text("content", encoding="utf-8")
            pipeline = build_pipeline(
                RecordingStore(), processor_registry(txt=RecordingProcessor())
            )

            with self.assertRaisesRegex(ValueError, "does not exist"):
                pipeline.ingest([FolderSource(path=root / "missing")])

            with self.assertRaisesRegex(ValueError, "not a directory"):
                pipeline.ingest([FolderSource(path=file_path)])

    def test_all_sources_are_validated_before_any_file_is_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid"
            valid.mkdir()
            (valid / "document.txt").write_text("content", encoding="utf-8")
            store = RecordingStore()
            pipeline = build_pipeline(
                store,
                processor_registry(txt=RecordingProcessor()),
            )

            with self.assertRaisesRegex(ValueError, "does not exist"):
                pipeline.ingest(
                    [
                        FolderSource(path=valid),
                        FolderSource(path=root / "missing"),
                    ]
                )

        self.assertEqual([], store.calls)

    def test_processing_and_indexing_failures_are_isolated_per_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.txt").write_text("broken", encoding="utf-8")
            (root / "good.txt").write_text("good", encoding="utf-8")
            (root / "rejected.txt").write_text("rejected", encoding="utf-8")
            processor = RecordingProcessor(failing_names=("broken.txt",))
            store = RecordingStore(rejected_content="rejected")
            pipeline = build_pipeline(store, processor_registry(txt=processor))

            result = pipeline.ingest([FolderSource(path=root)])

        self.assertEqual(3, result.discovered_count)
        self.assertEqual(1, result.processed_count)
        self.assertEqual(1, result.indexed_count)
        self.assertEqual(0, result.skipped_count)
        self.assertEqual(2, result.failed_count)
        self.assertEqual(
            (
                IngestionFailure(
                    path=str(root / "broken.txt"),
                    stage="processing",
                    error="processor could not parse file",
                ),
                IngestionFailure(
                    path=str(root / "rejected.txt"),
                    stage="indexing",
                    error="store rejected document",
                ),
            ),
            result.failures,
        )
        self.assertEqual(
            ["good"], [document.page_content for document in store.documents]
        )
        self.assertEqual(
            {
                "discovered_count": 3,
                "processed_count": 1,
                "indexed_count": 1,
                "skipped_count": 0,
                "failed_count": 2,
                "skipped": [],
                "failures": [failure.to_dict() for failure in result.failures],
            },
            result.to_dict(),
        )

    def test_source_metadata_is_added_and_non_scalar_values_are_flattened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "document.txt"
            path.write_text("content", encoding="utf-8")
            store = RecordingStore()
            pipeline = build_pipeline(
                store, processor_registry(txt=RecordingProcessor())
            )

            pipeline.ingest([FolderSource(path=root, chunk_size=100, chunk_overlap=10)])

        document = store.documents[0]
        self.assertEqual(str(path), document.metadata["source"])
        self.assertEqual("['custom', '.txt']", document.metadata["tags"])

    def test_same_orchestration_supports_distinct_store_implementations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.txt").write_text("content", encoding="utf-8")

            for store in (RecordingStore(), AlternateRecordingStore()):
                with self.subTest(store=type(store).__name__):
                    pipeline = build_pipeline(
                        store,
                        processor_registry(txt=RecordingProcessor()),
                    )
                    result = pipeline.ingest([FolderSource(path=root)])

                    self.assertEqual(1, result.indexed_count)

    def test_ingestion_requires_processors_and_a_write_capable_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.txt").write_text("content", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "document processor registry"):
                build_pipeline(RecordingStore()).ingest([FolderSource(path=root)])

            class ReadOnlyStore:
                def similarity_search(self, question, k=5):
                    return []

            with self.assertRaisesRegex(TypeError, "add_documents"):
                build_pipeline(
                    ReadOnlyStore(), processor_registry(txt=RecordingProcessor())
                ).ingest([FolderSource(path=root)])


if __name__ == "__main__":
    unittest.main()
