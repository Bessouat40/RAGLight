from .vectorstore.vector_store import VectorStore


def __getattr__(name: str):
    if name == "ChromaVS":
        try:
            from .vectorstore.chroma import ChromaVS
        except ImportError:
            raise ImportError(
                "chromadb is required to use ChromaVS. "
                "Install it with: pip install raglight[chroma]"
            )
        return ChromaVS
    if name == "QdrantVS":
        from .vectorstore.qdrant import QdrantVS

        return QdrantVS
    raise AttributeError(f"module 'raglight' has no attribute {name!r}")


from .embeddings.embeddings_model import EmbeddingsModel
from .embeddings.ollama_embeddings import OllamaEmbeddingsModel
from .embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel

from .cross_encoder.cross_encoder_model import CrossEncoderModel
from .cross_encoder.huggingface_cross_encoder import HuggingfaceCrossEncoderModel

from .llm.llm import LLM
from .llm.ollama_model import OllamaModel
from .llm.lmstudio_model import LMStudioModel
from .llm.mistral_model import MistralModel
from .llm.minimax_model import MiniMaxModel

from .rag.rag import RAG
from .rag.simple_rag_api import RAGPipeline
from .rag.simple_agentic_rag_api import AgenticRAGPipeline
from .rag.agentic_rag import AgenticRAG
from .rag.builder import Builder

from .config.settings import Settings
from .config.rag_config import RAGConfig
from .config.agentic_rag_config import AgenticRAGConfig
from .config.langfuse_config import LangfuseConfig
