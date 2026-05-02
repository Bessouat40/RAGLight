from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
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


@dataclass
class RetrievedDocument:
    """
    Represents a retrieved document with all relevant metadata and scores.
    
    Attributes:
        content (str): The document text content.
        source (str): The source of the document (file path, URL, etc.).
        metadata (Dict[str, Any]): Full metadata dictionary.
        bm25_score (Optional[float]): BM25 relevance score if applicable.
        rerank_score (Optional[float]): Cross-encoder rerank score if applicable.
        rrf_score (Optional[float]): RRF fusion score if applicable.
    """
    content: str
    source: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    
    @classmethod
    def from_langchain_doc(cls, doc: Document) -> "RetrievedDocument":
        """Create a RetrievedDocument from a LangChain Document."""
        metadata = doc.metadata or {}
        return cls(
            content=doc.page_content,
            source=metadata.get("source", "Unknown"),
            metadata=dict(metadata),
            bm25_score=metadata.get("bm25_score"),
            rerank_score=metadata.get("rerank_score"),
            rrf_score=metadata.get("rrf_score") or metadata.get("rrf_combined_score"),
        )


@dataclass
class RAGResult:
    """
    Complete result from a RAG pipeline execution.
    
    Attributes:
        answer (str): The generated answer from the LLM.
        retrieved_docs (List[RetrievedDocument]): All documents retrieved during the process.
        reformulated_question (Optional[str]): The reformulated question if query rewriting was used.
        original_question (str): The original user question.
        has_evidence (bool): Whether relevant evidence was found in the knowledge base.
        error_message (Optional[str]): Error message if any stage failed and fallback was used.
    """
    answer: str
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    reformulated_question: Optional[str] = None
    original_question: str = ""
    has_evidence: bool = True
    error_message: Optional[str] = None


class State(TypedDict):
    """
    Represents the state of the RAG process.

    Attributes:
        question (str): The input question for the RAG process.
        context (List[Document]): A list of documents retrieved from the vector store as context.
        answer (str): The generated answer based on the input question and context.
        history (List[Dict[str, str]]): The history of the conversation.
        reformulated_question (Optional[str]): The reformulated question if query rewriting was used.
        error_message (Optional[str]): Error message if any stage failed.
    """

    question: str
    answer: str
    context: List[Document] = []
    history: List[Dict[str, str]] = []
    reformulated_question: Optional[str] = None
    error_message: Optional[str] = None


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

    def _reformulate(self, state: State) -> Dict[str, Any]:
        """
        Rewrites the question as a standalone question using the conversation history.

        If there is no history or if reformulation fails, the original question is returned unchanged.

        Args:
            state (State): Current pipeline state with 'question' and 'history'.

        Returns:
            Dict[str, Any]: Updated state with the reformulated question and reformulated_question field.
        """
        original_question = state["question"]
        
        if not state["history"]:
            logger.info("No conversation history, skipping query reformulation")
            return {
                "question": original_question,
                "reformulated_question": None
            }

        try:
            history_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in state["history"]
            )
            prompt = (
                f"Given the following conversation history and a follow-up question, "
                f"rewrite the follow-up question as a standalone question that captures all necessary context.\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Follow-up question: {original_question}\n\n"
                f"Standalone question (output ONLY the reformulated question, nothing else):"
            )
            reformulated = self.llm.generate({"question": prompt, "history": []})
            reformulated_question = reformulated.strip()
            
            if not reformulated_question or len(reformulated_question) < 3:
                logger.warning(f"Reformulated question is empty or too short: '{reformulated_question}', using original")
                return {
                    "question": original_question,
                    "reformulated_question": None
                }
            
            logger.info(f"Query reformulated: '{original_question}' -> '{reformulated_question}'")
            return {
                "question": reformulated_question,
                "reformulated_question": reformulated_question
            }
            
        except Exception as e:
            logger.error(f"Query reformulation failed, using original question. Error: {e}")
            return {
                "question": original_question,
                "reformulated_question": None,
                "error_message": f"Reformulation failed: {str(e)}"
            }

    def _retrieve(self, state: State) -> Dict[str, Any]:
        """
        Retrieves relevant documents based on the input question.

        Args:
            state (Dict[str, str]): A dictionary containing the input question under the key 'question'.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved documents under the key 'context'.
        """
        question = state["question"]
        try:
            retrieved_docs = self.vector_store.similarity_search(
                question, k=self.k
            )
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: '{question}'")
            else:
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{question}'")
                
            return {
                "context": retrieved_docs, 
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed for query: '{question}'. Error: {e}")
            return {
                "context": [],
                "question": question,
                "error_message": f"Retrieval failed: {str(e)}"
            }

    def _build_prompt(self, state: Dict) -> str:
        """
        Builds a prompt that includes context with source citations and requires evidence-based answers.
        """
        context_docs = state.get("context", [])
        
        if not context_docs:
            return f"""
You are a helpful assistant. The user asked: {state["question"]}

IMPORTANT: No relevant documents were found in the knowledge base to answer this question.

Please respond exactly as follows:
"无法根据知识库中的内容回答此问题。知识库中没有找到与该问题相关的信息。"
"""

        context_sections = []
        for idx, doc in enumerate(context_docs, 1):
            metadata = doc.metadata if doc.metadata else {}
            source = metadata.get("source", "Unknown")
            score_info = []
            
            if "rerank_score" in metadata:
                score_info.append(f"rerank_score={metadata['rerank_score']:.4f}")
            if "bm25_score" in metadata:
                score_info.append(f"bm25_score={metadata['bm25_score']:.4f}")
            if "rrf_score" in metadata:
                score_info.append(f"rrf_score={metadata['rrf_score']:.4f}")
            if "rrf_combined_score" in metadata:
                score_info.append(f"rrf_combined={metadata['rrf_combined_score']:.4f}")
            
            score_str = ", ".join(score_info) if score_info else "N/A"
            
            context_sections.append(f"""---
[Document {idx}]
Source: {source}
Relevance Scores: {score_str}
Content:
{doc.page_content}
---""")

        context_str = "\n".join(context_sections)
        
        return f"""
You are an evidence-based assistant. Your answers MUST be strictly based on the provided context documents.

## Retrieved Context Documents:
{context_str}

## User Question:
{state["question"]}

## Instructions:
1. **ONLY use information explicitly stated in the retrieved context documents**
2. **Cite your sources** using [n] notation where n is the document number (e.g., "According to [1], ...")
3. If the context does not contain enough information to answer the question, respond EXACTLY with:
   "无法根据知识库中的内容回答此问题。知识库中没有找到与该问题相关的信息。"
4. Do NOT guess, fabricate, or use outside knowledge
5. If multiple documents contain relevant information, cite all relevant sources

## Final Answer (based only on the context):
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
        Preserves original metadata and adds rerank scores.

        Args:
            state (Dict[str, List[Document]]): A dictionary containing the list of retrieved documents under the key 'context'.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the reranked documents under the key 'context'.
        """
        try:
            question = state["question"]
            docs = state["context"]
            
            if not docs:
                logger.warning("No documents to rerank, returning empty context")
                return {"context": [], "question": state["question"]}

            doc_texts = [doc.page_content for doc in docs]
            top_k = max(1, int(self.k / 4))

            ranked_results = self.cross_encoder.predict(
                question, doc_texts, top_k
            )

            ranked_docs = []
            for result in ranked_results:
                original_doc = docs[result.corpus_id]
                new_metadata = dict(original_doc.metadata) if original_doc.metadata else {}
                new_metadata["rerank_score"] = result.score
                new_metadata["original_index"] = result.corpus_id
                
                original_stage = new_metadata.get("retrieval_stage", "unknown")
                retrieval_stages = new_metadata.get("retrieval_stages", [])
                if original_stage not in retrieval_stages:
                    retrieval_stages.append(original_stage)
                new_metadata["retrieval_stages"] = retrieval_stages
                new_metadata["retrieval_stage"] = "reranked"
                
                ranked_docs.append(Document(
                    page_content=result.text,
                    metadata=new_metadata
                ))

            logger.info(f"Rerank: {len(docs)} -> {len(ranked_docs)} documents preserved with metadata")

        except Exception as e:
            logger.error(f"Reranking failed, falling back to original context. Error: {e}")
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
                "Langfuse is not installed. Install it with: pip install 'langfuse==4.0.0'"
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
        result = self.generate_with_result(question)
        return result.answer

    def generate_with_result(self, question: str) -> RAGResult:
        """
        Executes the RAG pipeline for a given question and returns a complete RAGResult.

        Args:
            question (str): The input question.

        Returns:
            RAGResult: Complete result containing answer, retrieved documents, and metadata.
        """
        self.state["question"] = question
        original_question = question

        if self.max_history is not None:
            self.state["history"] = self.state["history"][-self.max_history :]

        try:
            if self.langfuse_config:
                callback = self._build_langfuse_callback()
                response = self.graph.invoke(self.state, config={"callbacks": [callback]})
            else:
                response = self.graph.invoke(self.state)

            answer = response["answer"]
            context_docs = response.get("context", [])
            reformulated_question = response.get("reformulated_question")
            error_message = response.get("error_message")

            retrieved_docs = [
                RetrievedDocument.from_langchain_doc(doc) for doc in context_docs
            ]

            has_evidence = len(retrieved_docs) > 0

            self.state["history"].extend(
                [
                    {"role": "user", "content": original_question},
                    {"role": "assistant", "content": answer},
                ]
            )

            return RAGResult(
                answer=answer,
                retrieved_docs=retrieved_docs,
                reformulated_question=reformulated_question,
                original_question=original_question,
                has_evidence=has_evidence,
                error_message=error_message,
            )

        except Exception as e:
            logger.error(f"RAG pipeline failed for question: '{question}'. Error: {e}")
            
            fallback_answer = "无法根据知识库中的内容回答此问题。处理过程中发生错误。"
            
            self.state["history"].extend(
                [
                    {"role": "user", "content": original_question},
                    {"role": "assistant", "content": fallback_answer},
                ]
            )

            return RAGResult(
                answer=fallback_answer,
                retrieved_docs=[],
                reformulated_question=None,
                original_question=original_question,
                has_evidence=False,
                error_message=f"Pipeline failed: {str(e)}",
            )

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

        reformulated_question = None
        error_messages = []

        if self.reformulation:
            reform_result = self._reformulate(state)
            state.update(reform_result)
            if reform_result.get("reformulated_question"):
                reformulated_question = reform_result["reformulated_question"]
            if reform_result.get("error_message"):
                error_messages.append(reform_result["error_message"])

        retrieve_result = self._retrieve(state)
        state.update(retrieve_result)
        if retrieve_result.get("error_message"):
            error_messages.append(retrieve_result["error_message"])

        if self.cross_encoder:
            state.update(self._rerank(state))

        prompt = self._build_prompt(state)

        callbacks = [self._build_langfuse_callback()] if self.langfuse_config else None

        try:
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

        except Exception as e:
            logger.error(f"Streaming generation failed. Error: {e}")
            fallback_answer = "无法根据知识库中的内容回答此问题。处理过程中发生错误。"
            yield fallback_answer

            self.state["history"].extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": fallback_answer},
                ]
            )
