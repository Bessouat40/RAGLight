from abc import ABC, abstractmethod
from typing import Dict, List

from ..models import Document


class DocumentProcessor(ABC):
    """
    Abstract base class for processing a file and turning it into document chunks.
    This defines the 'Strategy' interface for handling different file types.
    """

    @abstractmethod
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        """
        Loads a document from a file path, splits it into chunks,
        and returns a dict with two keys:
        - "chunks": the full document split into chunks.
        - "classes": class-level documents (for code files; empty list otherwise).

        Args:
            file_path (str): The path to the document file.
            chunk_size (int): The target size for each document chunk.
            chunk_overlap (int): The amount of overlap between consecutive chunks.

        Returns:
            Dict[str, List[Document]]: {"chunks": [...], "classes": [...]}
        """
        pass
