"""Tests for the Lore Context memory integration."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from raglight.memory.lore_memory import LoreMemory


class TestLoreMemoryInit(unittest.TestCase):
    """Test LoreMemory initialization."""

    def test_default_values(self):
        mem = LoreMemory()
        self.assertEqual(mem.api_url, "http://127.0.0.1:3000")
        self.assertEqual(mem.api_key, "")
        self.assertEqual(mem.project_id, "raglight")

    def test_custom_values(self):
        mem = LoreMemory(
            api_url="http://custom:9000",
            api_key="test-key",
            project_id="my-project",
        )
        self.assertEqual(mem.api_url, "http://custom:9000")
        self.assertEqual(mem.api_key, "test-key")
        self.assertEqual(mem.project_id, "my-project")

    def test_strips_trailing_slash(self):
        mem = LoreMemory(api_url="http://localhost:3000/")
        self.assertEqual(mem.api_url, "http://localhost:3000")

    @patch.dict(os.environ, {"LORE_API_URL": "http://env-url:4000", "LORE_API_KEY": "env-key"})
    def test_env_var_fallback(self):
        mem = LoreMemory()
        self.assertEqual(mem.api_url, "http://env-url:4000")
        self.assertEqual(mem.api_key, "env-key")


class TestLoreMemorySaveConversation(unittest.TestCase):
    """Test save_conversation method."""

    def setUp(self):
        self.mem = LoreMemory(api_key="test-key")
        self.mem._session = MagicMock()

    def test_returns_none_for_empty_messages(self):
        result = self.mem.save_conversation([])
        self.assertIsNone(result)

    def test_saves_conversation(self):
        from langchain_core.messages import HumanMessage, AIMessage

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"memory": {"id": "mem-123"}}
        self.mem._session.post.return_value = mock_resp

        messages = [
            HumanMessage(content="What is RAG?"),
            AIMessage(content="RAG is Retrieval-Augmented Generation."),
        ]
        result = self.mem.save_conversation(messages, session_id="s1")
        self.assertEqual(result, "mem-123")
        self.mem._session.post.assert_called_once()


class TestLoreMemoryRecallContext(unittest.TestCase):
    """Test recall_context method."""

    def setUp(self):
        self.mem = LoreMemory(api_key="test-key")
        self.mem._session = MagicMock()

    def test_returns_empty_on_no_hits(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"hits": []}
        self.mem._session.post.return_value = mock_resp

        result = self.mem.recall_context("test query")
        self.assertEqual(result, "")

    def test_returns_concatenated_context(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "hits": [
                {"content": "Context 1"},
                {"content": "Context 2"},
            ]
        }
        self.mem._session.post.return_value = mock_resp

        result = self.mem.recall_context("test query")
        self.assertIn("Context 1", result)
        self.assertIn("Context 2", result)

    def test_returns_empty_on_error(self):
        self.mem._session.post.side_effect = Exception("Connection error")
        result = self.mem.recall_context("test query")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
