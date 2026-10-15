"""Application facade for the standard RAG pipeline."""

from .config import PipelineConfig
from .pipeline import RAGPipeline

__all__ = ["PipelineConfig", "RAGPipeline"]
