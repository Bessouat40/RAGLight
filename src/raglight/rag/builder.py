from __future__ import annotations
import logging
from typing import Optional

from ..config.langfuse_config import LangfuseConfig
from ..embeddings.ollama_embeddings import OllamaEmbeddingsModel
from ..cross_encoder.cross_encoder_model import CrossEncoderModel
from ..cross_encoder.huggingface_cross_encoder import HuggingfaceCrossEncoderModel
from ..llm.llm import LLM
from ..llm.ollama_model import OllamaModel
from ..embeddings.openai_embeddings import OpenAIEmbeddingsModel
from ..llm.lmstudio_model import LMStudioModel
from ..llm.mistral_model import MistralModel
from ..llm.openai_model import OpenAIModel
from ..llm.minimax_model import MiniMaxModel
from ..vectorstore.vector_store import VectorStore
from ..config.settings import Settings
from .rag import RAG
from ..embeddings.embeddings_model import EmbeddingsModel
from ..embeddings.huggingface_embeddings import HuggingfaceEmbeddingsModel
from ..embeddings.gemini_embeddings import GeminiEmbeddingsModel
from ..llm.gemini_model import GeminiModel


class Builder:
    """
    Builder class for creating and configuring components of a Retrieval-Augmented Generation (RAG)
    or Retrieval-Augmented Thinking (RAT) pipeline.

    Attributes:
        vector_store (Optional[VectorStore]): The configured vector store instance.
        embeddings (Optional[EmbeddingsModel]): The configured embeddings model instance.
        llm (Optional[LLM]): The configured large language model (LLM) instance.
        rag (Optional[RAG]): The configured RAG pipeline instance.
    """

    def __init__(self) -> None:
        """
        Initializes a Builder instance with no configured components.
        """
        self.vector_store: Optional[VectorStore] = None
        self.embeddings: Optional[EmbeddingsModel] = None
        self.cross_encoder: Optional[CrossEncoderModel] = None
        self.llm: Optional[LLM] = None
        self.reasoning_llm: Optional[LLM] = None
        self.rag: Optional[RAG] = None

    def with_embeddings(self, type: str, **kwargs) -> Builder:
        """
        Configures the embeddings model.

        Args:
            type (str): The type of embeddings model to create (e.g., HUGGINGFACE).
            **kwargs: Additional parameters required to initialize the embeddings model.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown embeddings model type is specified.
        """
        logging.info("⏳ Creating an Embeddings Model...")
        if type == Settings.HUGGINGFACE:
            kwargs.pop("api_base", None)
            self.embeddings = HuggingfaceEmbeddingsModel(**kwargs)
        elif type == Settings.OLLAMA:
            self.embeddings = OllamaEmbeddingsModel(**kwargs)
        elif type in (Settings.VLLM, Settings.OPENAI):
            self.embeddings = OpenAIEmbeddingsModel(**kwargs)
        elif type == Settings.GOOGLE_GEMINI:
            self.embeddings = GeminiEmbeddingsModel(**kwargs)
        elif type == Settings.AWS_BEDROCK:
            from ..embeddings.bedrock_embeddings import BedrockEmbeddingsModel

            kwargs.pop("api_base", None)
            self.embeddings = BedrockEmbeddingsModel(**kwargs)
        else:
            raise ValueError(f"Unknown Embeddings Model type: {type}")
        logging.info("✅ Embeddings Model created")
        return self

    def with_cross_encoder(self, type: str, **kwargs) -> Builder:
        """
        Configures the cross-encoder

        Args:
            type (str): The type of cross encoder model to create (e.g., HUGGINGFACE).
            **kwargs: Additional parameters required to initialize the cross encoder model.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown cross encoder model type is specified.
        """
        logging.info("⏳ Creating a Cross Encoder Model...")
        if type == Settings.HUGGINGFACE:
            self.cross_encoder = HuggingfaceCrossEncoderModel(**kwargs)
        else:
            raise ValueError(f"Unknown Cross Encoder Model type: {type}")
        logging.info("✅ Cross Encoder Model created")
        return self

    def with_vector_store(self, type: str, **kwargs) -> Builder:
        """
        Configures the vector store.

        Args:
            type (str): The type of vector store to create (e.g., CHROMA).
            **kwargs: Additional parameters required to initialize the vector store.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If the embeddings model is not set or an unknown vector store type is specified.
        """
        logging.info("⏳ Creating a VectorStore...")
        if self.embeddings is None:
            raise ValueError(
                "You need to set an embedding model before setting a vector store"
            )
        elif type == Settings.CHROMA:
            try:
                from ..vectorstore.chroma import ChromaVS
            except ImportError:
                raise ImportError(
                    "chromadb is required to use ChromaVS. "
                    "Install it with: pip install raglight[chroma]"
                )
            search_type = kwargs.pop("search_type", Settings.SEARCH_HYBRID)
            alpha = kwargs.pop("alpha", 0.5)
            self.vector_store = ChromaVS(
                embeddings_model=self.embeddings,
                search_type=search_type,
                alpha=alpha,
                **kwargs,
            )
        elif type == Settings.QDRANT:
            from ..vectorstore.qdrant import QdrantVS

            search_type = kwargs.pop("search_type", Settings.SEARCH_HYBRID)
            alpha = kwargs.pop("alpha", 0.5)
            self.vector_store = QdrantVS(
                embeddings_model=self.embeddings,
                search_type=search_type,
                alpha=alpha,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown VectorStore type: {type}")
        logging.info("✅ VectorStore created")
        return self

    def with_llm(self, type: str, **kwargs) -> Builder:
        """
        Configures the large language model (LLM).

        Args:
            type (str): The type of LLM to create (e.g., OLLAMA).
            **kwargs: Additional parameters required to initialize the LLM.

        Returns:
            Builder: The current instance of the Builder for method chaining.

        Raises:
            ValueError: If an unknown LLM type is specified.
        """
        logging.info("⏳ Creating an LLM...")
        if type == Settings.OLLAMA:
            self.llm = OllamaModel(**kwargs)
        elif type == Settings.LMSTUDIO:
            self.llm = LMStudioModel(**kwargs)
        elif type == Settings.MISTRAL:
            self.llm = MistralModel(**kwargs)
        elif type in (Settings.VLLM, Settings.OPENAI):
            self.llm = OpenAIModel(**kwargs)
        elif type == Settings.GOOGLE_GEMINI:
            self.llm = GeminiModel(**kwargs)
        elif type == Settings.MINIMAX:
            self.llm = MiniMaxModel(**kwargs)
        elif type == Settings.AWS_BEDROCK:
            from ..llm.bedrock_model import BedrockModel

            kwargs.pop("api_base", None)
            self.llm = BedrockModel(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {type}")
        logging.info("✅ LLM created")
        return self

    def build_rag(
        self,
        k: int = 10,
        langfuse_config: Optional[LangfuseConfig] = None,
        reformulation: bool = True,
        max_history: Optional[int] = 20,
    ) -> RAG:
        """
        Builds the RAG pipeline with the configured components.

        Args:
            k (int, optional): The number of top documents to retrieve. Defaults to 10.
            langfuse_config (Optional[LangfuseConfig]): Langfuse observability
                configuration (v3+). When provided, every ``RAG.generate()`` call
                is traced in Langfuse. Defaults to ``None``.

        Returns:
            RAG: The fully configured RAG pipeline instance.

        Raises:
            ValueError: If any of the required components (vector store, LLM, embeddings) are not set.
        """
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.llm is None:
            raise ValueError("LLM is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        logging.info("⏳ Building the RAG pipeline...")
        self.rag = RAG(
            self.embeddings,
            self.vector_store,
            self.llm,
            k,
            self.cross_encoder,
            langfuse_config=langfuse_config,
            reformulation=reformulation,
            max_history=max_history,
        )
        logging.info("✅ RAG pipeline created")
        return self.rag

    def build_vector_store(self) -> VectorStore:
        """
        Returns the configured vector store instance.

        Returns:
            VectorStore: The configured vector store instance.

        Raises:
            ValueError: If the vector store or embeddings model is not set.
        """
        if self.vector_store is None:
            raise ValueError("VectorStore is required")
        if self.embeddings is None:
            raise ValueError("Embeddings Model is required")
        logging.info("✅ VectorStore instance returned")
        return self.vector_store

    def build_llm(self) -> LLM:
        """
        Returns the configured LLM instance.

        Returns:
            LLM: The configured large language model instance.

        Raises:
            ValueError: If the LLM is not set.
        """
        if self.llm is None:
            raise ValueError("LLM is required")
        logging.info("✅ LLM instance returned")
        return self.llm
