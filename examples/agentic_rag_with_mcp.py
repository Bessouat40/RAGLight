from raglight.config.settings import Settings
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings
from raglight.models.data_source_model import FolderSource
from raglight.models.data_source_model import GitHubSource
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

knowledge_base = [
    FolderSource(path="<path to your folder with pdf>/knowledge_base"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight"),
]

persist_directory = "./defaultDb"
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
collection_name = Settings.DEFAULT_COLLECTION_NAME

vector_store_config = VectorStoreConfig(
    embedding_model=model_embeddings,
    # api_base = ... # If you have a custom client URL for your embeddings provider
    database=Settings.CHROMA,
    persist_directory=persist_directory,
    provider=Settings.HUGGINGFACE,
    collection_name=collection_name,
)

# Custom ignore folders - you can override the default list
# custom_ignore_folders = [
#     ".venv",
#     "venv",
#     "node_modules",
#     "__pycache__",
#     ".git",
#     "build",
#     "dist",
#     "my_custom_folder_to_ignore"  # Add your custom folders here
# ]

config = AgenticRAGConfig(
    provider=Settings.OPENAI,
    model="gpt-4o",
    k=10,
    system_prompt=Settings.DEFAULT_AGENT_PROMPT,
    knowledge_base=knowledge_base,
    mcp_config={
        "mcp_server_0": {
            "url": "http://127.0.0.1:8001/sse",
            "transport": "sse"
        }
    },
    max_steps=2,
    api_key=Settings.OPENAI_API_KEY,  # os.environ.get('OPENAI_API_KEY')
    ignore_folders=Settings.DEFAULT_IGNORE_FOLDERS,  # Use custom ignore folders
    # api_base = ... # If you have a custom client URL
)

agenticRag = AgenticRAGPipeline(config, vector_store_config)

agenticRag.build()

response = agenticRag.generate(
    "Please implement AgenticRAGPipeline for me using RAGLight framework."
)

print("response : ", response)
