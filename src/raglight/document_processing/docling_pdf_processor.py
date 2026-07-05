import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from .document_processor import DocumentProcessor


def _set_pipeline_option_if_supported(
    pipeline_options: PdfPipelineOptions, option_name: str, value: bool
) -> None:
    """
    Safely set a Docling PdfPipelineOptions field when available.

    Older Docling versions may not expose newer enrichment flags.
    """
    model_fields = getattr(type(pipeline_options), "model_fields", None)
    if isinstance(model_fields, dict) and option_name not in model_fields:
        logging.info(
            "Docling option '%s' is not available in this version, skipping.",
            option_name,
        )
        return

    try:
        setattr(pipeline_options, option_name, value)
    except (AttributeError, ValueError) as e:
        logging.warning(
            "Unable to set Docling option '%s' (continuing without it): %s",
            option_name,
            e,
        )


class DoclingPDFProcessor(DocumentProcessor):
    """
    Advanced PDF processor using IBM's Docling for high-fidelity document parsing.
    Supports table structure recognition, formula extraction (LaTeX), code snippet
    recognition, and chart recognition.
    """

    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        _set_pipeline_option_if_supported(
            pipeline_options, option_name="do_table_structure", value=True
        )
        _set_pipeline_option_if_supported(
            pipeline_options, option_name="do_formula_enrichment", value=True
        )
        _set_pipeline_option_if_supported(
            pipeline_options, option_name="do_code_enrichment", value=True
        )
        _set_pipeline_option_if_supported(
            pipeline_options, option_name="do_chart_extraction", value=True
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        try:
            result = self.converter.convert(file_path)

            # Export to markdown to preserve structure (tables, formulas as LaTeX, etc.)
            markdown_content = result.document.export_to_markdown()

            if not markdown_content:
                logging.warning(f"Docling returned no text for {file_path}.")
                return {"chunks": [], "classes": []}

            # Create a single document for the entire content
            # Note: Docling's markdown export currently doesn't easily split by page
            # while maintaining the structural integrity of elements that span pages.
            raw_document = Document(
                page_content=markdown_content,
                metadata={"source": file_path, "processor": "docling"},
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents([raw_document])

            return {"chunks": chunks, "classes": []}

        except Exception as e:
            logging.error(f"Failed to process PDF with Docling {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}
