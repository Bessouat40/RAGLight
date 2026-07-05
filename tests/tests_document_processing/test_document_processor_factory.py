import unittest
from unittest.mock import MagicMock, patch

from raglight.document_processing.document_processor_factory import (
    DocumentProcessorFactory,
)
from raglight.document_processing.pdf_processor import PDFProcessor


class TestDocumentProcessorFactory(unittest.TestCase):
    def test_uses_docling_processor_when_available(self):
        docling_instance = MagicMock(name="docling_processor")

        with (
            patch(
                "raglight.document_processing.document_processor_factory.HAS_DOCLING",
                True,
            ),
            patch(
                "raglight.document_processing.document_processor_factory.DoclingPDFProcessor",
                return_value=docling_instance,
                create=True,
            ) as mock_docling,
        ):
            factory = DocumentProcessorFactory()

        self.assertIs(factory.get_processor("sample.pdf"), docling_instance)
        mock_docling.assert_called_once()

    def test_falls_back_to_pdf_processor_when_docling_init_fails(self):
        with (
            patch(
                "raglight.document_processing.document_processor_factory.HAS_DOCLING",
                True,
            ),
            patch(
                "raglight.document_processing.document_processor_factory.DoclingPDFProcessor",
                side_effect=ValueError("unsupported docling field"),
                create=True,
            ),
        ):
            factory = DocumentProcessorFactory()

        self.assertIsInstance(factory.get_processor("sample.pdf"), PDFProcessor)

    def test_falls_back_to_pdf_processor_when_docling_is_unavailable(self):
        with patch(
            "raglight.document_processing.document_processor_factory.HAS_DOCLING", False
        ):
            factory = DocumentProcessorFactory()

        self.assertIsInstance(factory.get_processor("sample.pdf"), PDFProcessor)


if __name__ == "__main__":
    unittest.main()
