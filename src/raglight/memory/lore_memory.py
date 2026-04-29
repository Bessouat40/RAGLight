"""
Lore Context memory integration for RAGLight.

Persists conversation history to Lore Context via its REST API,
enabling RAG pipelines to recall relevant past context across sessions.

Requires:
    - A running Lore Context API instance (default: http://127.0.0.1:3000)
    - LORE_API_KEY environment variable (or pass api_key directly)

Example:
    from raglight.memory.lore_memory import LoreMemory

    memory = LoreMemory(project_id="my-project")
    memory.save_conversation(messages, session_id="s1")
    context = memory.recall_context("What did we discuss about embeddings?")
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on requests
_requests = None


def _get_requests():
    global _requests
    if _requests is None:
        import requests
        _requests = requests
    return _requests


class LoreMemory:
    """
    Persists RAGLight conversation history to Lore Context.

    Uses Lore's REST API to store conversations and retrieve relevant
    past context, enabling cross-session memory for RAG pipelines.

    Args:
        api_url: Lore API base URL. Falls back to LORE_API_URL env var,
            then http://127.0.0.1:3000.
        api_key: Lore API key. Falls back to LORE_API_KEY env var.
        project_id: Project namespace in Lore. Defaults to "raglight".
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: str = "raglight",
    ):
        self.api_url = (
            api_url
            or os.environ.get("LORE_API_URL", "http://127.0.0.1:3000")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("LORE_API_KEY", "")
        self.project_id = project_id
        self._session = None

    def _get_session(self):
        if self._session is None:
            req = _get_requests()
            self._session = req.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    def _post(self, path: str, body: dict) -> dict:
        session = self._get_session()
        resp = session.post(f"{self.api_url}{path}", json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def save_conversation(
        self,
        messages: list,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save conversation history to Lore memory.

        Args:
            messages: List of LangChain message objects (HumanMessage, AIMessage).
            session_id: Optional session identifier for grouping.

        Returns:
            Memory ID if saved successfully, None on failure.
        """
        if not messages:
            return None

        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            logger.warning("langchain_core not available; skipping Lore save")
            return None

        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"AI: {msg.content}")

        if not parts:
            return None

        content = "\n\n".join(parts)
        session_tag = f" [session:{session_id}]" if session_id else ""

        try:
            result = self._post(
                "/v1/memory/write",
                {
                    "content": f"RAGLight conversation{session_tag}:\n\n{content}",
                    "memory_type": "conversation",
                    "project_id": self.project_id,
                    "scope": "project",
                },
            )
            memory_id = result.get("memory", {}).get("id")
            logger.info("Saved conversation to Lore memory: %s", memory_id)
            return memory_id
        except Exception as e:
            logger.warning("Failed to save conversation to Lore: %s", e)
            return None

    def recall_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant past context from Lore for the given query.

        Args:
            query: The search query (typically the user's current question).
            top_k: Maximum number of memory hits to retrieve.

        Returns:
            Concatenated relevant context strings, or empty string on failure.
        """
        try:
            result = self._post(
                "/v1/memory/search",
                {
                    "query": query,
                    "project_id": self.project_id,
                    "top_k": top_k,
                },
            )
            hits = result.get("hits", [])
            if not hits:
                return ""

            contexts = []
            for hit in hits:
                content = hit.get("content", "")
                if content:
                    contexts.append(content)

            return "\n\n---\n\n".join(contexts)
        except Exception as e:
            logger.warning("Failed to recall context from Lore: %s", e)
            return ""
