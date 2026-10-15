from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Iterable, Optional

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import Dict, List, TypedDict

from ..config.langfuse_config import LangfuseConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel
from ..embeddings.embeddings_model import EmbeddingsModel
from ..llm.llm import LLM
from ..vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class State(TypedDict):
    """
    Represents the state of the RAG process.

    Attributes:
        question (str): The input question for the RAG process.
        context (List[Document]): A list of documents retrieved from the vector store as context.
        answer (str): The generated answer based on the input question and context.
        history (List[Dict[str, str]]): The history of the conversation.
    """

    question: str
    answer: str
    context: List[Document] = []
    history: List[Dict[str, str]] = []


class RAG:
    """
    Implementation of a Retrieval-Augmented Generation (RAG) pipeline.

    This class integrates embeddings, a vector store, and a large language model (LLM) to
    retrieve relevant documents and generate answers based on a user's query.

    Attributes:
        embeddings: The embedding model used for vectorization.
        vector_store (VectorStore): The vector store instance for document retrieval.
        llm (LLM): The large language model instance for answer generation.
        k (int, optional): The number of top documents to retrieve. Defaults to 5.
        graph (StateGraph): The state graph that manages the RAG process flow.
    """

    def __init__(
        self,
        embedding_model: EmbeddingsModel,
        vector_store: VectorStore,
        llm: LLM,
        k: int,
        cross_encoder_model: CrossEncoderModel = None,
        langfuse_config: Optional[LangfuseConfig] = None,
        reformulation: bool = True,
        max_history: Optional[int] = 20,
    ) -> None:
        """
        Initializes the RAG pipeline.

        Args:
            embedding_model (EmbeddingsModel): The embedding model used for vectorization.
            vector_store (VectorStore): The vector store for retrieving relevant documents.
            llm (LLM): The language model for generating answers.
            reformulation (bool): Whether to rewrite the question before retrieval. Defaults to True.
            max_history (Optional[int]): Maximum number of messages to keep in history.
                                         None means unlimited. Defaults to 20.
        """
        self.embeddings: EmbeddingsModel = embedding_model.get_model()
        self.cross_encoder: CrossEncoderModel = (
            cross_encoder_model if cross_encoder_model else None
        )
        self.vector_store: VectorStore = vector_store
        self.llm: LLM = llm
        self.k: int = k
        self.reformulation: bool = reformulation
        self.max_history: Optional[int] = max_history
        self.langfuse_config: Optional[LangfuseConfig] = langfuse_config
        self.langfuse_session_id: str = (
            langfuse_config.session_id
            if langfuse_config and langfuse_config.session_id
            else uuid.uuid4().hex  # 32 lowercase hex chars, required by Langfuse v4
        )
        self.state: State = State(question="", answer="", context=[], history=[])
        self.graph: Any = (
            self._createGraph()
        )  # Here type is CompiledGraph but it's not exposed by https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/graph.py

    def _reformulate(self, state: State) -> Dict[str, str]:
        """
        Rewrites the question as a standalone question using the conversation history.

        If there is no history, the original question is returned unchanged.

        Args:
            state (State): Current pipeline state with 'question' and 'history'.

        Returns:
            Dict[str, str]: Updated state with the reformulated question.
        """
        if not state["history"]:
            return {"question": state["question"]}

        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in state["history"]
        )
        prompt = (
            f"Given the following conversation history and a follow-up question, "
            f"rewrite the follow-up question as a standalone question that captures all necessary context.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {state['question']}\n\n"
            f"Standalone question (output ONLY the reformulated question, nothing else):"
        )
        reformulated = self.llm.generate({"question": prompt, "history": []})
        logger.info(f"Reformulated question: {reformulated.strip()}")
        return {"question": reformulated.strip()}

    def _retrieve(self, state: State) -> Dict[str, List[Document]]:
        """
        Retrieves relevant documents based on the input question.

        Args:
            state (Dict[str, str]): A dictionary containing the input question under the key 'question'.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the retrieved documents under the key 'context'.
        """
        retrieved_docs = self.vector_store.similarity_search(
            state["question"], k=self.k
        )
        return {"context": retrieved_docs, "question": state["question"]}

    def _build_prompt(self, state: Dict) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        return f"""
            Here is the retrieved context (excerpts from the document):
            {docs_content}

            Here is the question:
            {state["question"]}


            FINAL ANSWER (based only on the context):
            """

    def _generate_graph(self, state: Dict[str, List[Document]]) -> Dict[str, str]:
        """
        Generates an answer based on the input question and retrieved context.

        Args:
            state (Dict[str, List[Document]]): A dictionary containing:
                - 'question': The input question.
                - 'context': The list of retrieved documents.

        Returns:
            Dict[str, str]: A dictionary containing the generated answer under the key 'answer'.
        """
        prompt = self._build_prompt(state)
        response = self.llm.generate({"question": prompt, "history": state["history"]})
        return {"answer": response}

    def _rerank(self, state: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        Reranks the retrieved documents based on the cross-encoder model.

        Args:
            state (Dict[str, List[Document]]): A dictionary containing the list of retrieved documents under the key 'context'.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the reranked documents under the key 'context'.
        """
        try:
            question = state["question"]
            docs = state["context"]
            doc_texts = [doc.page_content for doc in docs]

            ranked_texts = self.cross_encoder.predict(
                question, doc_texts, int(self.k / 4)
            )

            ranked_docs = [Document(page_content=text) for text in ranked_texts]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            ranked_docs = state["context"]

        return {"context": ranked_docs, "question": state["question"]}

    def _createGraph(self) -> Any:
        """
        Creates and compiles the state graph for the RAG pipeline.

        Returns:
            StateGraph: The compiled state graph for managing the RAG process flow.
        """
        if self.cross_encoder:
            steps = [self._retrieve, self._rerank, self._generate_graph]
            self.k = 4 * self.k  # Increase retrieval window for reranking
        else:
            steps = [self._retrieve, self._generate_graph]

        if self.reformulation:
            steps = [self._reformulate] + steps

        graph_builder = StateGraph(State).add_sequence(steps)
        first_step = "_reformulate" if self.reformulation else "_retrieve"
        graph_builder.add_edge(START, first_step)
        return graph_builder.compile()

    def _build_langfuse_callback(self) -> Any:
        """
        Builds a Langfuse ``CallbackHandler`` from the stored configuration.

        Sets the required environment variables and returns a handler whose
        ``trace_id`` is fixed to ``self.langfuse_session_id`` so that all turns
        of the same conversation are grouped under the same Langfuse trace.

        Returns:
            CallbackHandler: A ready-to-use Langfuse LangChain callback.

        Raises:
            ImportError: If ``langfuse==4.0.0`` is not installed.
        """
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:
            raise ImportError(
                "Langfuse support requires optional dependencies. "
                "Install them with: pip install 'raglight[langfuse]'"
            ) from exc

        os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
        os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        return CallbackHandler(trace_context={"trace_id": self.langfuse_session_id})

    def generate(self, question: str) -> str:
        """
        Executes the RAG pipeline for a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The generated answer from the pipeline.
        """
        self.state["question"] = question

        if self.max_history is not None:
            self.state["history"] = self.state["history"][-self.max_history :]

        if self.langfuse_config:
            callback = self._build_langfuse_callback()
            response = self.graph.invoke(self.state, config={"callbacks": [callback]})
        else:
            response = self.graph.invoke(self.state)

        answer = response["answer"]
        self.state["history"].extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    def generate_streaming(self, question: str) -> Iterable[str]:
        """
        Executes the RAG pipeline and streams the answer token by token.

        Runs reformulation, retrieval, and reranking via the existing methods,
        then delegates to the LLM's streaming interface.

        Args:
            question (str): The input question.

        Yields:
            str: Successive chunks of the generated answer.
        """
        if self.max_history is not None:
            self.state["history"] = self.state["history"][-self.max_history :]

        state: Dict = {
            "question": question,
            "context": [],
            "history": list(self.state["history"]),
        }

        if self.reformulation:
            state.update(self._reformulate(state))

        state.update(self._retrieve(state))

        if self.cross_encoder:
            state.update(self._rerank(state))

        prompt = self._build_prompt(state)

        callbacks = [self._build_langfuse_callback()] if self.langfuse_config else None

        full_answer = ""
        for chunk in self.llm.generate_streaming(
            {"question": prompt, "history": state["history"]}, callbacks=callbacks
        ):
            full_answer += chunk
            yield chunk

        self.state["history"].extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ]
        )
