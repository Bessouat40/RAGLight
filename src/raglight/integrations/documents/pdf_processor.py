import fitz
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from ...models import Document


class PDFProcessor:
    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        with fitz.open(file_path) as pdf:
            raw_documents = []

            for page_index, page in enumerate(pdf):
                blocks = page.get_text("dict")["blocks"]
                page_text_parts = []

                for block in blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text_parts.append(span["text"])
                            page_text_parts.append("\n")
                        page_text_parts.append("\n")

                full_page_text = "".join(page_text_parts).strip()

                if full_page_text:
                    document = LangChainDocument(
                        page_content=full_page_text,
                        metadata={"source": file_path, "page": page_index},
                    )
                    raw_documents.append(document)

        if not raw_documents:
            return {"chunks": [], "classes": []}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_documents = text_splitter.split_documents(raw_documents)
        chunks = []
        for chunk in split_documents:
            metadata = dict(chunk.metadata)
            metadata["source"] = file_path
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

        return {"chunks": chunks, "classes": []}
