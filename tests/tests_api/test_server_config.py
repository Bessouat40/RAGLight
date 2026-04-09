import os
import unittest
from unittest.mock import patch

from raglight.api.server_config import ServerConfig
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings


def _clean_env():
    """Return os.environ without any RAGLIGHT_* keys."""
    return {k: v for k, v in os.environ.items() if not k.startswith("RAGLIGHT_")}


class TestServerConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = ServerConfig()
        self.assertEqual(cfg.llm_model, Settings.DEFAULT_LLM)
        self.assertEqual(cfg.llm_provider, Settings.OLLAMA)
        self.assertEqual(cfg.embeddings_model, Settings.DEFAULT_EMBEDDINGS_MODEL)
        self.assertEqual(cfg.embeddings_provider, Settings.HUGGINGFACE)
        self.assertEqual(cfg.collection, "default")
        self.assertEqual(cfg.k, Settings.DEFAULT_K)
        self.assertIsNone(cfg.db_host)
        self.assertIsNone(cfg.db_port)

    def test_env_override(self):
        env = {
            "RAGLIGHT_LLM_MODEL": "mistral",
            "RAGLIGHT_LLM_PROVIDER": "Mistral",
            "RAGLIGHT_LLM_API_BASE": "http://mistral:11434",
            "RAGLIGHT_EMBEDDINGS_MODEL": "nomic-embed-text",
            "RAGLIGHT_EMBEDDINGS_PROVIDER": "Ollama",
            "RAGLIGHT_EMBEDDINGS_API_BASE": "http://ollama:11434",
            "RAGLIGHT_PERSIST_DIR": "/tmp/mydb",
            "RAGLIGHT_COLLECTION": "myproject",
            "RAGLIGHT_K": "10",
            "RAGLIGHT_DB": "chromadb",
            "RAGLIGHT_DB_PORT": "8001",
            "RAGLIGHT_DB_HOST": "localhost",
            "RAGLIGHT_MCP_URLS": "http://mcp:8080,http://mcp:8081",
        }
        with patch.dict(os.environ, {**_clean_env(), **env}, clear=True):
            cfg = ServerConfig()
        self.assertEqual(cfg.llm_model, "mistral")
        self.assertEqual(cfg.llm_provider, "Mistral")
        self.assertEqual(cfg.llm_api_base, "http://mistral:11434")
        self.assertEqual(cfg.embeddings_model, "nomic-embed-text")
        self.assertEqual(cfg.embeddings_provider, "Ollama")
        self.assertEqual(cfg.mcp_urls, "http://mcp:8080,http://mcp:8081")
        self.assertEqual(cfg.embeddings_api_base, "http://ollama:11434")
        self.assertEqual(cfg.persist_dir, "/tmp/mydb")
        self.assertEqual(cfg.collection, "myproject")
        self.assertEqual(cfg.k, 10)
        self.assertEqual(cfg.db_host, "localhost")
        self.assertEqual(cfg.db_port, 8001)
        self.assertEqual(cfg.db, "chromadb")

    def test_to_rag_config_returns_correct_type(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = ServerConfig()
        rag_config = cfg.to_rag_config()
        self.assertIsInstance(rag_config, RAGConfig)
        self.assertEqual(rag_config.llm, cfg.llm_model)
        self.assertEqual(rag_config.provider, cfg.llm_provider)
        self.assertEqual(rag_config.k, cfg.k)
        self.assertEqual(rag_config.system_prompt, cfg.system_prompt)

    def test_to_vector_store_config_returns_correct_type(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            cfg = ServerConfig()
        vs_config = cfg.to_vector_store_config()
        self.assertIsInstance(vs_config, VectorStoreConfig)
        self.assertEqual(vs_config.embedding_model, cfg.embeddings_model)
        self.assertEqual(vs_config.provider, cfg.embeddings_provider)
        self.assertEqual(vs_config.collection_name, cfg.collection)

    def test_to_vector_store_config_chroma_host(self):
        env = {
            **_clean_env(),
            "RAGLIGHT_DB_HOST": "chromadb",
            "RAGLIGHT_DB_PORT": "8000",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ServerConfig()
        vs_config = cfg.to_vector_store_config()
        self.assertEqual(vs_config.host, "chromadb")
        self.assertEqual(vs_config.port, 8000)
        self.assertIsNone(vs_config.persist_directory)

    def test_to_vector_store_config_persistent(self):
        env = {**_clean_env(), "RAGLIGHT_PERSIST_DIR": "/tmp/testdb"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ServerConfig()
        vs_config = cfg.to_vector_store_config()
        self.assertEqual(vs_config.persist_directory, "/tmp/testdb")
        self.assertIsNone(vs_config.host)


if __name__ == "__main__":
    unittest.main()
