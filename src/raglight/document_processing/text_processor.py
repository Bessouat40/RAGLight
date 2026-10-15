import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from ..models import Document
from .document_processor import DocumentProcessor


class TextProcessor(DocumentProcessor):
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decode failed for {file_path}, trying latin-1.")
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

            if not text:
                logging.warning(f"File {file_path} is empty.")
                return {"chunks": [], "classes": []}

            doc = LangChainDocument(page_content=text, metadata={"source": file_path})

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            split_documents = text_splitter.split_documents([doc])
            chunks = [
                Document(
                    page_content=chunk.page_content,
                    metadata=dict(chunk.metadata),
                )
                for chunk in split_documents
            ]

            return {"chunks": chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process text file {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
