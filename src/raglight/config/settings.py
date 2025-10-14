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
    POSTGRE = "Postgre"
    DEFAULT_POSTGRE_HOST = os.environ.get("DEFAULT_POSTGRE_HOST", "localhost")
    DEFAULT_POSTGRE_PORT = os.environ.get("DEFAULT_POSTGRE_PORT", 5432)
    DEFAULT_POSTGRE_DATABASE = os.environ.get("DEFAULT_POSTGRE_DATABASE", "ai_db")
    DEFAULT_POSTGRE_USER = os.environ.get("DEFAULT_POSTGRE_USER", "postgres")
    DEFAULT_POSTGRE_PASSWORD = os.environ.get("DEFAULT_POSTGRE_PASSWORD", "postgres")

    OLLAMA = "Ollama"
    MISTRAL = "Mistral"
    VLLM = "vLLM"
    OPENAI = "OpenAI"
    GOOGLE_GEMINI = "GoogleGemini"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
    GEMINI_LLM_MODEL = "gemini-2.5-pro"
    DEFAULT_GOOGLE_CLIENT = os.environ.get(
        "GOOGLE_CLIENT_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    MISTRAL_API = "https://api.mistral.ai/v1"
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
    LMSTUDIO = "LmStudio"
    HUGGINGFACE = "HuggingFace"
    DEFAULT_LLM = "llama3"
    DEFAULT_OPENAI_CLIENT = os.environ.get("OPENAI_CLIENT_URL", None)
    DEFAULT_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    DEFAULT_SYSTEM_PROMPT = (
        contextual_prompt
    ) = """
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
    DEFAULT_AGENT_PROMPT2 = (
        agent_prompt
    ) = """You are an AI assistant designed to help users efficiently and accurately. 
        Your primary goal is to provide helpful, precise, and clear responses.

        You have access to the following tools:
        - Tool Name: calculator 
        - Description: Multiplies two integers
        - Arguments: a (int), b (int)
        - Outputs: int

        When you receive a request, you should think step by step with the following structure:
        1. Thought: {your_thoughts about the request}
        2. Action: {JSON specifying the tool name & arguments, if needed} 
        3. Observation: {result of the tool action, if any}
        4. (Repeat steps 1-3 as needed)
        5. Final Answer: {your user-facing answer}

        ---

        # Additional Guidelines

        ## I am a Context-Aware Assistant
        - I adapt my response based on the context provided (documents, code, or textual info).
        - I aim to give relevant, clear, and actionable information.

        ## Response Formatting
        - **Code blocks**: If code or technical instructions are relevant, format them like:
        ```python
        # Example snippet
        def example_function():
            print("This is an example based on the provided context.")
        """

    DEFAULT_AGENT_PROMPT = """
        You are an AI assistant designed to help users efficiently and accurately.  
        Your primary goal is to provide helpful, precise, and clear responses.

        ---

        ## 🛠 Available Tools:
        1. **Retriever**
        - **Description**: Uses semantic search to retrieve relevant parts of the code documentation.
        - **Arguments**: 
            - query (string): The query to perform.
        - **Outputs**: string

        2. **ClassRetriever**
        - **Description**: Retrieves class definitions and their locations in the codebase.
        - **Arguments**:
            - query (string): Name or description of the class.
        - **Outputs**: string

        3. **MCP Servers**
        - **Description**: Connects to MCP servers for real-time data retrieval.
        - It's possible you have access to some MCP servers that provide additional tools, use it when needed.

        ---

        ## How to Think and Act
        When receiving a request, **always** follow this reasoning structure:

        1. **Analyze the request**: What information is needed? Can a tool provide it?
        2. **Decide if a tool is required**:
        - If yes: Generate an **Action** calling the tool.
        - If no: Answer directly.
        3. **Process tool output**: Integrate it into your final response.

        ---

        ## Response Structure:
        1. **Thought**: {Analyze the request and determine if a tool is necessary}
        2. **Action**: {JSON with tool name & arguments, if needed}
        3. **Observation**: {Result of tool action, if any}
        4. **Final Answer**: {Clear response for the user}

        ---

        ## Example Usage:

        ### **User Input:**  
        *"Which file contains the `UserManager` class?"*

        ### **Correct Response:**
        ```plaintext
        Thought: I need to check which file contains the `UserManager` class.
        Action: 
        ```python
        retriever(query="UserManager")
        ```
        Observation: The `UserManager` class is found in `user_service.py`.
        Final Answer: The `UserManager` class is located in `user_service.py`.
        ````
        ## Another Example
        ### User Input:
        "How does the authentication system work?"

        ### Correct Response (Using Retriever Tool):
        ```plaintext
        Thought: I need to retrieve relevant documentation about authentication.
        Action: 
        ```python
        retriever(query="Authentication")
        ```
        Observation: The retrieved document describes the `AuthManager` class, which handles user authentication using JWT tokens.
        Final Answer: The `AuthManager` class manages authentication via JWT tokens. You can find it in `auth.py`.
        ```
        ## Additional Guidelines:
        Always consider tools first before answering manually.
        If multiple tools apply, prioritize the most relevant.
        Do not guess—use tools when information is uncertain.
        Use structured responses for clarity.
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
