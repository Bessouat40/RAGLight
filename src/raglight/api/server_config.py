import os
from dataclasses import dataclass, field
from typing import Optional

from ..config.langfuse_config import LangfuseConfig
from ..config.settings import Settings
from ..config.rag_config import RAGConfig
from ..config.vector_store_config import VectorStoreConfig


@dataclass
class ServerConfig:
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_LLM_MODEL", Settings.DEFAULT_LLM
        )
    )
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_LLM_PROVIDER", Settings.OLLAMA)
    )
    llm_api_base: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_LLM_API_BASE", Settings.DEFAULT_OLLAMA_CLIENT
        )
    )
    embeddings_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_EMBEDDINGS_MODEL", Settings.DEFAULT_EMBEDDINGS_MODEL
        )
    )
    embeddings_provider: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_EMBEDDINGS_PROVIDER", Settings.HUGGINGFACE
        )
    )
    embeddings_api_base: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_EMBEDDINGS_API_BASE", Settings.DEFAULT_OLLAMA_CLIENT
        )
    )
    persist_dir: str = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_PERSIST_DIR", "./raglight_db")
    )
    collection: str = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_COLLECTION", "default")
    )
    k: int = field(
        default_factory=lambda: int(
            os.environ.get("RAGLIGHT_K", str(Settings.DEFAULT_K))
        )
    )
    system_prompt: str = field(
        default_factory=lambda: os.environ.get(
            "RAGLIGHT_SYSTEM_PROMPT", Settings.DEFAULT_SYSTEM_PROMPT
        )
    )
    db: str = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_DB", Settings.CHROMA)
    )
    db_host: Optional[str] = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_DB_HOST") or None
    )
    db_port: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ.get("RAGLIGHT_DB_PORT"))
            if os.environ.get("RAGLIGHT_DB_PORT")
            else None
        )
    )
    langfuse_host: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL")
        or None
    )
    langfuse_public_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY") or None
    )
    langfuse_secret_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY") or None
    )
    mcp_urls: Optional[str] = field(
        default_factory=lambda: os.environ.get("RAGLIGHT_MCP_URLS") or None
    )

    def _build_langfuse_config(self) -> Optional[LangfuseConfig]:
        if self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key:
            return LangfuseConfig(
                public_key=self.langfuse_public_key,
                secret_key=self.langfuse_secret_key,
                host=self.langfuse_host,
            )
        return None

    def to_rag_config(self) -> RAGConfig:
        return RAGConfig(
            llm=self.llm_model,
            provider=self.llm_provider,
            api_base=self.llm_api_base,
            system_prompt=self.system_prompt,
            k=self.k,
            langfuse_config=self._build_langfuse_config(),
        )

    def to_vector_store_config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            embedding_model=self.embeddings_model,
            provider=self.embeddings_provider,
            api_base=self.embeddings_api_base,
            persist_directory=self.persist_dir if not self.db_host else None,
            host=self.db_host,
            port=self.db_port,
            database=self.db,
            collection_name=self.collection,
        )
