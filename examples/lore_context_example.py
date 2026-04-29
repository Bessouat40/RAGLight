"""
Example: RAGLight with Lore Context for cross-session memory.

This example shows how to configure RAGLight's Agentic RAG pipeline
with Lore Context integration, enabling the pipeline to recall relevant
past conversations across sessions.

Prerequisites:
    1. Start Lore Context API:
       cd /path/to/lore-context
       pnpm build && PORT=3000 pnpm start:api

    2. Set your Lore API key:
       export LORE_API_KEY=your-api-key-here

    3. Install RAGLight with dependencies:
       pip install raglight
"""

from raglight.config.settings import Settings
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.models.data_source_model import FolderSource

Settings.setup_logging()

# Define your knowledge base
knowledge_base = [
    FolderSource(path="./knowledge_base"),
]

# Configure vector store
vector_store_config = VectorStoreConfig(
    embedding_model=Settings.DEFAULT_EMBEDDINGS_MODEL,
    database=Settings.CHROMA,
    persist_directory="./defaultDb",
    provider=Settings.HUGGINGFACE,
    collection_name=Settings.DEFAULT_COLLECTION_NAME,
)

# Configure agentic RAG with Lore Context memory
config = AgenticRAGConfig(
    provider=Settings.OPENAI,
    model="gpt-4o",
    k=10,
    system_prompt=Settings.DEFAULT_AGENT_PROMPT,
    knowledge_base=knowledge_base,
    max_steps=2,
    api_key=Settings.OPENAI_API_KEY,
    # Enable Lore Context for cross-session memory
    lore_config={
        "api_url": "http://127.0.0.1:3000",  # Lore API URL
        "api_key": Settings.LORE_API_KEY if hasattr(Settings, "LORE_API_KEY") else "",
        "project_id": "my-rag-project",  # Namespace in Lore
    },
)

# Build and use the pipeline
pipeline = AgenticRAGPipeline(config, vector_store_config)
pipeline.build()

# First session — the conversation will be saved to Lore
response1 = pipeline.generate("What are the main components of RAGLight?")
print("Response 1:", response1)

response2 = pipeline.generate("How do I add a new LLM provider?")
print("Response 2:", response2)

# In a future session, Lore will recall relevant past context
# even if the pipeline is restarted, helping maintain continuity

pipeline.close()
