from typing import Dict, Optional

from ...contracts import DocumentProcessor
from .pdf_processor import PDFProcessor
from .code_processor import CodeProcessor
from .text_processor import TextProcessor


class DocumentProcessorFactory:
    """
    A stateless factory that returns the appropriate DocumentProcessor
    strategy for a given file path.
    """

    def __init__(
        self, custom_processors: Optional[Dict[str, DocumentProcessor]] = None
    ):
        # Default processors
        self._processors: Dict[str, DocumentProcessor] = {
            # PDF files
            "pdf": PDFProcessor(),
            # Code files
            "py": CodeProcessor(),
            "js": CodeProcessor(),
            "ts": CodeProcessor(),
            "java": CodeProcessor(),
            "cpp": CodeProcessor(),
            "cs": CodeProcessor(),
            # Text files
            "txt": TextProcessor(),
            "md": TextProcessor(),
            "html": TextProcessor(),
        }

        if custom_processors:
            self._processors.update(custom_processors)

    def get_processor(self, file_path: str) -> DocumentProcessor | None:
        """
        Returns the appropriate processor for the given file path based on its extension.
        """
        extension = file_path.split(".")[-1].lower()
        return self._processors.get(extension)
