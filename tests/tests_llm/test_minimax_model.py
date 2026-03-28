import unittest
from unittest.mock import MagicMock, patch

# Import directly to avoid the broken agentic_rag import chain
from raglight.llm.minimax_model import MiniMaxModel
from raglight.config.settings import Settings


MINIMAX_LLM_MODEL = "MiniMax-M2.7"


class TestMiniMaxModel(unittest.TestCase):
    _MOCK_RESPONSE = "Hello! This is a test MiniMax response."

    @patch("raglight.llm.minimax_model.ChatOpenAI")
    def setUp(self, mock_chat_openai: MagicMock):
        mock_chat_openai.return_value = MagicMock()
        self.model = MiniMaxModel(
            model_name=MINIMAX_LLM_MODEL,
        )

        mock_response = MagicMock()
        mock_response.content = self._MOCK_RESPONSE
        self.model.model.invoke = MagicMock(return_value=mock_response)

    def test_generate_response(self):
        response = self.model.generate({"question": "Say hello."})
        self.assertIsInstance(response, str)
        self.assertEqual(response, self._MOCK_RESPONSE)

    def test_generate_calls_invoke(self):
        self.model.generate({"question": "Test question"})
        self.model.model.invoke.assert_called_once()

    def test_system_prompt_included(self):
        """System prompt should be prepended as a SystemMessage."""
        self.model.generate({"question": "Test"})
        call_args = self.model.model.invoke.call_args[0][0]
        from langchain_core.messages import SystemMessage, HumanMessage

        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertIsInstance(call_args[-1], HumanMessage)

    def test_strip_think_tags(self):
        """Think tags should be stripped from output."""
        mock_response = MagicMock()
        mock_response.content = "<think>reasoning here</think>Final answer"
        self.model.model.invoke = MagicMock(return_value=mock_response)

        response = self.model.generate({"question": "Test"})
        self.assertEqual(response, "Final answer")

    def test_strip_multiline_think_tags(self):
        """Multi-line think tags should be stripped."""
        mock_response = MagicMock()
        mock_response.content = (
            "<think>\nstep 1\nstep 2\n</think>\nThe final answer is 42."
        )
        self.model.model.invoke = MagicMock(return_value=mock_response)

        response = self.model.generate({"question": "What is the answer?"})
        self.assertEqual(response, "The final answer is 42.")

    def test_no_think_tags_unchanged(self):
        """Output without think tags should remain unchanged."""
        mock_response = MagicMock()
        mock_response.content = "Just a normal response."
        self.model.model.invoke = MagicMock(return_value=mock_response)

        response = self.model.generate({"question": "Test"})
        self.assertEqual(response, "Just a normal response.")

    def test_temperature_clamping(self):
        """Temperature should be clamped to > 0."""
        with patch("raglight.llm.minimax_model.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = MiniMaxModel(
                model_name=MINIMAX_LLM_MODEL,
                temperature=0.0,
            )
            self.assertGreater(model.temperature, 0)

    def test_default_api_base(self):
        """Default API base should be the MiniMax endpoint."""
        with patch("raglight.llm.minimax_model.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = MiniMaxModel(model_name=MINIMAX_LLM_MODEL)
            self.assertEqual(model.api_base, "https://api.minimax.io/v1")

    def test_custom_api_base(self):
        """Custom API base should be used when provided."""
        with patch("raglight.llm.minimax_model.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = MiniMaxModel(
                model_name=MINIMAX_LLM_MODEL,
                api_base="https://custom.minimax.io/v1",
            )
            self.assertEqual(model.api_base, "https://custom.minimax.io/v1")

    def test_history_included_in_messages(self):
        """Conversation history should be included in messages."""
        self.model.generate(
            {
                "question": "Follow up?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            }
        )
        call_args = self.model.model.invoke.call_args[0][0]
        from langchain_core.messages import AIMessage, HumanMessage

        # system + user history + assistant history + question
        self.assertEqual(len(call_args), 4)
        self.assertIsInstance(call_args[1], HumanMessage)
        self.assertIsInstance(call_args[2], AIMessage)

    def test_streaming(self):
        """Streaming should yield chunks from the model."""
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = " world"
        self.model.model.stream = MagicMock(return_value=iter([chunk1, chunk2]))

        chunks = list(
            self.model.generate_streaming({"question": "Say hello."})
        )
        self.assertEqual(chunks, ["Hello", " world"])

    def test_streaming_skips_empty_chunks(self):
        """Streaming should skip chunks with empty content."""
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = ""
        chunk3 = MagicMock()
        chunk3.content = " world"
        self.model.model.stream = MagicMock(
            return_value=iter([chunk1, chunk2, chunk3])
        )

        chunks = list(
            self.model.generate_streaming({"question": "Test"})
        )
        self.assertEqual(chunks, ["Hello", " world"])

    def test_image_support(self):
        """Image input should be formatted correctly."""
        self.model.generate(
            {
                "question": "Describe this image",
                "images": [{"base64": "abc123"}],
            }
        )
        call_args = self.model.model.invoke.call_args[0][0]
        from langchain_core.messages import HumanMessage

        last_msg = call_args[-1]
        self.assertIsInstance(last_msg, HumanMessage)
        # Content should be a list with text + image_url
        self.assertIsInstance(last_msg.content, list)
        self.assertEqual(len(last_msg.content), 2)
        self.assertEqual(last_msg.content[0]["type"], "text")
        self.assertEqual(last_msg.content[1]["type"], "image_url")

    def test_custom_system_prompt(self):
        """Custom system prompt should be used."""
        with patch("raglight.llm.minimax_model.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = MiniMaxModel(
                model_name=MINIMAX_LLM_MODEL,
                system_prompt="You are a helpful assistant.",
            )
            self.assertEqual(model.system_prompt, "You are a helpful assistant.")


class TestMiniMaxModelBuilder(unittest.TestCase):
    """Integration tests for MiniMax registration in the Builder."""

    @patch("raglight.llm.minimax_model.ChatOpenAI")
    def test_builder_with_minimax(self, mock_chat_openai):
        """Builder should accept Settings.MINIMAX as provider type."""
        mock_chat_openai.return_value = MagicMock()
        from raglight.rag.builder import Builder

        builder = Builder()
        builder.with_llm(
            Settings.MINIMAX,
            model_name=MINIMAX_LLM_MODEL,
        )
        self.assertIsInstance(builder.llm, MiniMaxModel)

    @patch("raglight.llm.minimax_model.ChatOpenAI")
    def test_builder_build_llm(self, mock_chat_openai):
        """Builder.build_llm() should return the MiniMax instance."""
        mock_chat_openai.return_value = MagicMock()
        from raglight.rag.builder import Builder

        builder = Builder()
        builder.with_llm(
            Settings.MINIMAX,
            model_name=MINIMAX_LLM_MODEL,
        )
        llm = builder.build_llm()
        self.assertIsInstance(llm, MiniMaxModel)
        self.assertEqual(llm.model_name, MINIMAX_LLM_MODEL)

    def test_builder_rejects_unknown_provider(self):
        """Builder should raise ValueError for unknown provider types."""
        from raglight.rag.builder import Builder

        builder = Builder()
        with self.assertRaises(ValueError):
            builder.with_llm("UnknownProvider", model_name="test-model")


class TestMiniMaxSettings(unittest.TestCase):
    """Test that MiniMax settings are properly defined."""

    def test_minimax_constant_exists(self):
        self.assertEqual(Settings.MINIMAX, "MiniMax")

    def test_minimax_default_client_url(self):
        self.assertEqual(
            Settings.DEFAULT_MINIMAX_CLIENT, "https://api.minimax.io/v1"
        )

    def test_minimax_llm_model(self):
        self.assertEqual(Settings.MINIMAX_LLM_MODEL, "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
