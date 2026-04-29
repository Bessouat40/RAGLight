import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from .agentic_rag_utils.rag_tools import RetrieverTool, ClassRetrieverTool
from .builder import Builder
from ..vectorstore.vector_store import VectorStore
from ..config.agentic_rag_config import AgenticRAGConfig
from ..config.settings import Settings
from ..config.vector_store_config import VectorStoreConfig


class AgenticRAG:
    """
    Main engine for the Agentic RAG pipeline.

    This class orchestrates the interaction between the Large Language Model (LLM),
    local Vector Store tools, and optional Model Context Protocol (MCP) servers
    to generate grounded responses.
    """

    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ):
        """
        Initializes the Agentic RAG engine.

        Args:
            config (AgenticRAGConfig): Configuration for the RAG agent (LLM, MCP, etc.).
            vector_store_config (VectorStoreConfig): Configuration for the underlying vector database.
        """
        self.config = config
        self.vector_store = self.create_vector_store(vector_store_config)
        self.k: int = config.k
        self.conversation_history: list = []

        self.local_tools = [
            RetrieverTool(k=config.k, vector_store=self.vector_store),
            ClassRetrieverTool(k=config.k, vector_store=self.vector_store),
        ]

        try:
            if hasattr(self.vector_store, "get_available_collections"):
                collections = self.vector_store.get_available_collections()
                if collections:
                    cols_list = ", ".join([str(c) for c in collections])
                    for tool in self.local_tools:
                        tool.description += (
                            f" Available collections in the database: {cols_list}."
                        )
        except Exception:
            pass

        self.model = self._create_llm_model(config)

        # Lore Context integration for cross-session memory
        self.lore_memory = None
        if config.lore_config:
            from ..memory.lore_memory import LoreMemory
            self.lore_memory = LoreMemory(**config.lore_config)

        # Pre-compile the agent for local-only mode (reused across calls)
        self._local_agent = create_agent(
            self.model,
            tools=self.local_tools,
            system_prompt=config.system_prompt,
        )

    def _create_llm_model(self, config: AgenticRAGConfig) -> BaseChatModel:
        """
        Factory method that instantiates the appropriate LangChain ChatModel.

        Args:
            config (AgenticRAGConfig): The configuration object containing provider settings.

        Returns:
            BaseChatModel: A LangChain-compatible chat model instance.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        provider = config.provider

        if provider == Settings.GOOGLE_GEMINI:
            return ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.api_key or Settings.GEMINI_API_KEY,
                temperature=0.5,
            )

        elif provider == Settings.MISTRAL:
            return ChatMistralAI(
                model=config.model,
                mistral_api_key=config.api_key or Settings.MISTRAL_API_KEY,
                temperature=0.5,
            )

        elif provider == Settings.OPENAI:
            return ChatOpenAI(
                model=config.model,
                base_url=config.api_base or Settings.DEFAULT_OPENAI_CLIENT,
                api_key=config.api_key,
                temperature=0.5,
            )

        elif provider == Settings.OLLAMA:
            return ChatOllama(
                model=config.model,
                base_url=config.api_base or Settings.DEFAULT_OLLAMA_CLIENT,
                temperature=0.5,
            )

        elif provider == Settings.LMSTUDIO:
            return ChatOpenAI(
                model=config.model,
                openai_api_key="not-needed",
                base_url=config.api_base or Settings.DEFAULT_LMSTUDIO_CLIENT,
                temperature=0.5,
            )

        else:
            raise ValueError(
                f"Provider '{provider}' not supported by LangChain factory."
            )

    def create_vector_store(self, config: VectorStoreConfig) -> VectorStore:
        """
        Creates and configures the vector store instance.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            VectorStore: An initialized instance of the vector store.
        """
        return (
            Builder()
            .with_embeddings(
                config.provider,
                model_name=config.embedding_model,
                api_base=config.api_base,
            )
            .with_vector_store(
                type=config.database,
                persist_directory=config.persist_directory,
                collection_name=config.collection_name,
                host=config.host,
                port=config.port,
            )
            .build_vector_store()
        )

    def clear_history(self) -> None:
        """Resets the conversation history."""
        self.conversation_history = []

    async def generate(self, query: str, stream: bool = False) -> str:
        """
        Asynchronously generates a response to the user's query.

        Maintains conversation history across calls for multi-turn dialogue.
        When MCP servers are configured, fetches their tools and builds a
        per-call agent that includes both local and remote tools.

        Args:
            query (str): The user's question or command.
            stream (bool): Reserved for future streaming support.

        Returns:
            str: The final text response generated by the agent.
        """
        self.conversation_history.append(HumanMessage(content=query))

        # Inject relevant past context from Lore if available
        if self.lore_memory:
            from langchain_core.messages import SystemMessage
            past_context = self.lore_memory.recall_context(query)
            if past_context:
                self.conversation_history.append(
                    SystemMessage(
                        content=f"[Relevant past context from previous sessions]:\n{past_context}"
                    )
                )

        if self.config.mcp_config:
            mcp_client = MultiServerMCPClient(self.config.mcp_config)
            mcp_tools = await mcp_client.get_tools()
            all_tools = self.local_tools + mcp_tools
            agent = create_agent(
                self.model,
                tools=all_tools,
                system_prompt=self.config.system_prompt,
            )
        else:
            agent = self._local_agent

        result = await self._invoke_agent(agent, query)
        self.conversation_history.append(AIMessage(content=result))

        # Persist conversation to Lore for cross-session recall
        if self.lore_memory:
            self.lore_memory.save_conversation(self.conversation_history)

        return result

    async def _invoke_agent(self, agent, query: str) -> str:
        """
        Invokes the compiled agent with the full conversation history.

        Args:
            agent: The compiled LangGraph agent.
            query (str): The current user query (already appended to history).

        Returns:
            str: The agent's output text.
        """
        try:
            response = await agent.ainvoke(
                {"messages": self.conversation_history},
                config={"recursion_limit": self.config.max_steps},
            )

            if isinstance(response, dict) and "messages" in response:
                last_msg = response["messages"][-1]
                return (
                    last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                )
            else:
                return response.get("output", str(response))
        except Exception as e:
            logging.error(f"Agent execution failed: {e}")
            raise SystemError(f"Error during generation: {str(e)}")
