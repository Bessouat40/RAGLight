import unittest
from unittest.mock import MagicMock, patch

from raglight.llm.ollama_model import OllamaModel
from ..test_config import TestsConfig


class TestOllamaModel(unittest.TestCase):
    @patch("raglight.llm.ollama_model.ChatOllama")
    def setUp(self, MockChatOllama):
        mock_lc_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Machine learning (ML) is a subset of artificial intelligence"
        )
        mock_lc_client.invoke.return_value = mock_response
        MockChatOllama.return_value = mock_lc_client

        self.model = OllamaModel(
            model_name=TestsConfig.OLLAMA_MODEL,
            system_prompt_file=TestsConfig.TEST_SYSTEM_PROMPT,
            preload_model=False,
            options={"temperature": 0.3},
            headers={"x-some-header": "some-value"},
        )

    def test_generate_response(self):
        question = "Define machine learning."
        response = self.model.generate({"question": question})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(
            response, "Machine learning (ML) is a subset of artificial intelligence"
        )
        self.model.model.invoke.assert_called_once()


class TestOllamaModelStreaming(unittest.TestCase):
    @patch("raglight.llm.ollama_model.ChatOllama")
    def setUp(self, MockChatOllama):
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = " world"

        self.mock_lc_client = MagicMock()
        self.mock_lc_client.stream.return_value = iter([chunk1, chunk2])
        MockChatOllama.return_value = self.mock_lc_client

        self.model = OllamaModel(
            model_name=TestsConfig.OLLAMA_MODEL,
            system_prompt="You are helpful.",
            preload_model=False,
        )

    def test_generate_streaming_yields_chunks(self):
        result = list(self.model.generate_streaming({"question": "Say hello."}))
        self.assertEqual(result, ["Hello", " world"])

    def test_generate_streaming_builds_correct_messages(self):
        from langchain_core.messages import SystemMessage, HumanMessage

        list(self.model.generate_streaming({"question": "Say hello."}))
        call_args = self.mock_lc_client.stream.call_args
        messages = call_args[0][0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, "You are helpful.")
        self.assertIsInstance(messages[-1], HumanMessage)
        self.assertEqual(messages[-1].content, "Say hello.")

    def test_generate_streaming_includes_history(self):
        from langchain_core.messages import AIMessage, HumanMessage

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        list(
            self.model.generate_streaming(
                {"question": "Follow-up.", "history": history}
            )
        )
        call_args = self.mock_lc_client.stream.call_args
        messages = call_args[0][0]
        # system + 2 history + user question = 4
        self.assertEqual(len(messages), 4)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "Previous question")
        self.assertIsInstance(messages[2], AIMessage)

    def test_generate_streaming_passes_callbacks(self):
        cb = MagicMock()
        list(self.model.generate_streaming({"question": "Say hello."}, callbacks=[cb]))
        call_kwargs = self.mock_lc_client.stream.call_args.kwargs
        self.assertIn("callbacks", call_kwargs.get("config", {}))


if __name__ == "__main__":
    unittest.main()
