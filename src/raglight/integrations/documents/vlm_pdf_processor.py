import base64
import logging
import os
import shutil
import tempfile
import fitz
from typing import List

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...contracts import LLM
from ...models import Document

import uuid


def to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


class VlmPDFProcessor:
    """
    PDF processor that extracts both text and images from a PDF file,
    generates detailed captions for each image using a Vision-Language Model (VLM),
    and returns the resulting documents split into semantic chunks.
    """

    def __init__(self, vlm: LLM):
        """
        Initialize the processor with a Vision-Language Model interface.

        Args:
            vlm (LLM): Any implementation of the LLM interface capable of
                       handling image inputs and generating captions.
        """
        self.vlm = vlm

    def process(self, file_path: str, chunk_size: int, chunk_overlap: int):
        """
        Process a PDF and extract a sequence of text and image-based documents.

        Workflow:
            1. Iterate through the PDF using PyMuPDF
            2. Extract text blocks and image blocks in reading order
            3. Convert images to detailed captions via the VLM
            4. Split all resulting documents with RecursiveCharacterTextSplitter

        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Maximum chunk size for text splitting.
            chunk_overlap (int): Overlap between chunks.

        Returns:
            dict: A dictionary with the fields:
                  - "chunks": List[Document]
                  - "classes": Empty list (for compatibility)
        """

        tmp_dir = tempfile.mkdtemp(prefix="pdf_images_")
        try:
            doc = fitz.open(file_path)
            documents: List[LangChainDocument] = []

            for page_index, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" in block:
                        text = ""

                        for line in block["lines"]:
                            for span in line["spans"]:
                                text += span["text"] + " "

                        text = text.strip()
                        if text:
                            documents.append(
                                LangChainDocument(
                                    page_content=text,
                                    metadata={
                                        "source": file_path,
                                        "page": page_index,
                                        "type": "text",
                                    },
                                )
                            )

                    if "image" in block:
                        img_bytes = block["image"]
                        img_ext = block["ext"]
                        img_id = uuid.uuid4()

                        img_path = os.path.join(
                            tmp_dir,
                            f"{os.path.basename(file_path)}_p{page_index}_{img_id}.{img_ext}",
                        )

                        with open(img_path, "wb") as f:
                            f.write(img_bytes)

                        img_b64 = to_base64(img_bytes)

                        try:
                            caption = self.vlm.generate(
                                {
                                    "question": "Describe this image in detail.",
                                    "images": [
                                        {
                                            "bytes": img_bytes,
                                            "base64": img_b64,
                                        }
                                    ],
                                }
                            )
                        except Exception as e:
                            logging.error(f"VLM caption failed for {img_path}: {e}")
                            caption = "Image (caption unavailable due to VLM error)."

                        documents.append(
                            LangChainDocument(
                                page_content=f"[IMAGE CAPTION]\n{caption}",
                                metadata={
                                    "source": file_path,
                                    "page": page_index,
                                    "type": "image",
                                    "path": img_path,
                                },
                            )
                        )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            split_docs = [
                Document(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                )
                for document in splitter.split_documents(documents)
            ]

            return {"chunks": split_docs, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
