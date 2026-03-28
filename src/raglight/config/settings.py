import logging
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Settings:
    """
    A class that contains constants for various settings used in the project.
    """

    @staticmethod
    def setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    CHROMA = "Chroma"
    QDRANT = "Qdrant"
    OLLAMA = "Ollama"
    MISTRAL = "Mistral"
    VLLM = "vLLM"
    OPENAI = "OpenAI"
    GOOGLE_GEMINI = "GoogleGemini"
    MINIMAX = "MiniMax"
    AWS_BEDROCK = "AWSBedrock"
    AWS_DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
    AWS_BEDROCK_LLM_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    AWS_BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
    GEMINI_LLM_MODEL = "gemini-2.5-pro"
    DEFAULT_GOOGLE_CLIENT = os.environ.get("GOOGLE_CLIENT_URL", None)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    DEFAULT_MISTRAL_CLIENT = os.environ.get(
        "MISTRAL_CLIENT_URL", "https://api.mistral.ai/v1"
    )
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
    MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
    DEFAULT_MINIMAX_CLIENT = os.environ.get(
        "MINIMAX_CLIENT_URL", "https://api.minimax.io/v1"
    )
    MINIMAX_LLM_MODEL = "MiniMax-M2.7"
    LMSTUDIO = "LmStudio"
    HUGGINGFACE = "HuggingFace"
    DEFAULT_LLM = "llama3.2:1b"
    DEFAULT_OPENAI_CLIENT = os.environ.get(
        "OPENAI_CLIENT_URL", "https://api.openai.com/v1"
    )
    DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    DEFAULT_SYSTEM_PROMPT = """
        # I am a Context-Aware Assistant:
        - My primary role is to utilize the provided context (e.g., documents, code, or descriptions) to answer user questions accurately and effectively.
        - I adapt my responses based on the given context, aiming to provide relevant, clear, and actionable information.
        ## Response Formatting:
        - **Code Blocks:** If the context involves code or technical instructions, I will format them as:
        ```python
        # Example code snippet
        def example_function():
            print("This is an example based on the provided context.")
        ```
        ## Headings and Lists:
        - I use headings and lists to organize complex explanations or workflows for clarity.
        - Bold/Italic Text: Important concepts or keywords are highlighted for emphasis.
        ## Context Utilization:
        - If the context includes:
        ## Documents or Text:
        - I will summarize, explain, or extract key details.
        ## Code:
        - I will review, debug, or provide usage examples.
        ## Questions:
        - I will tailor my response to directly address the query using the provided information.
        """

    DEFAULT_AGENT_PROMPT = """
        You are an advanced AI assistant designed to help users efficiently and accurately.  
        Your responses must always be reliable, grounded, and based on actual data.

        ---

        ## 🛠 Available Tools
        You have access to powerful tools that fetch real information from the codebase or external MCP servers.  
        Your default behavior must be to **use these tools whenever they can improve the accuracy of your answer**.

        1. **Retriever**
        - Semantic search over the documentation and codebase.
        - Arguments:
            - query (string)
        - Output: relevant textual excerpts.

        2. **ClassRetriever**
        - Retrieves class definitions and their source file locations.
        - Arguments:
            - query (string)
        - Output: class content + file path.

        3. **MCP Servers**
        - Provide additional capabilities through remote modules and tools.
        - Use MCP tools when local tools are not sufficient or when real-time data is required.

        ---

        ## 🔍 Decision Strategy (MANDATORY)

        Before responding, you must ALWAYS follow this reasoning process:

        1. **Analyze the request carefully.**
        - Does the user ask about code, classes, system behavior, file location, definitions, architecture, APIs, or “where/how is X implemented”?
            → If yes, you MUST call `retriever` or `class_retriever`.

        2. **Prefer tools over guessing.**
        - If the answer is uncertain or could be wrong without verification, ALWAYS call the appropriate tool.
        - Never fabricate unknown details or assume file names.

        3. **Use tool output to craft your final answer.**
        - The tool gives raw data.
        - You then interpret, summarize, and present it clearly.

        4. **Use MCP tools whenever needed.**
        - If the question involves data not covered by your local retrieval tools, call an MCP tool.

        ---

        ## 📦 Response Format (STRICT)

        Your responses must always follow this structure:

        1. **Thought**  
        Explain why you need (or don't need) to call a tool. Be explicit.

        2. **Action**  
        - If a tool is needed → output a JSON action block.
        - If no tool is needed → explicitly state why.

        3. **Observation**  
        Filled only after tool output is provided (left empty during planning).

        4. **Final Answer**  
        A clear, helpful answer for the user.

        ---

        ## 🧠 Rules for Tool Usage

        - If the question mentions:
        - a class
        - a function
        - a file
        - an implementation detail
        - “where is”
        - “how does X work”
        - “show me”
        
        → **You MUST call a tool.**

        - If multiple tools apply, choose the most specific one:
        - For classes → `class_retriever`
        - For general documentation → `retriever`
        - For external real-time data → MCP tool

        - NEVER guess filenames, paths, or implementation details.
        - NEVER summarize nonexistent information.
        - NEVER answer “from memory” if the tool can confirm it.

        ---

        ## 🧪 Example 1 — Good Response

        **User:**  
        “Which file contains the `UserManager` class?”

        **Correct Behavior:**

        Thought: This requires class lookup. I must use the class_retriever tool.  
        Action:
        ```json
        {
        "tool": "class_retriever",
        "arguments": { "query": "UserManager" }
        }
        """
    DEFAULT_COLLECTION_NAME = "default"
    DEFAULT_PERSIST_DIRECTORY = "./defaultDb"
    DEFAULT_OLLAMA_CLIENT = os.environ.get(
        "OLLAMA_CLIENT_URL", "http://localhost:11434"
    )
    DEFAULT_LMSTUDIO_CLIENT = os.environ.get(
        "LMSTUDIO_CLIENT", "http://localhost:1234/v1"
    )
    DEFAULT_EXTENSIONS = "**/[!.]*"
    REASONING_LLMS = ["deepseek-r1"]
    DEFAULT_REASONING_LLM = "deepseek-r1:1.5b"
    THINKING_PATTERN = r"<think>(.*?)</think>"
    DEFAULT_K = 5

    SEARCH_SEMANTIC = "semantic"
    SEARCH_BM25 = "bm25"
    SEARCH_HYBRID = "hybrid"

    DEFAULT_IGNORE_FOLDERS = [
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".git",
        ".vscode",
        ".idea",
        "build",
        "dist",
        "target",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        "logs",
        "tmp",
        "temp",
    ]
