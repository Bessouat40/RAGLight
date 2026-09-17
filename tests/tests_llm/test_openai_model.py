import unittest
from unittest.mock import MagicMock, patch

from raglight.llm.openai_model import OpenAIModel
from ..test_config import TestsConfig


class TestOpenAIModel(unittest.TestCase):
    _MOCK_RESPONSE = "Hello! This is a test OpenAI response."

    @patch("raglight.llm.openai_model.ChatOpenAI")
    @patch("raglight.Settings.OPENAI_API_KEY", "DUMMY_KEY")
    def setUp(self, MockChatOpenAI):
        mock_lc_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self._MOCK_RESPONSE
        mock_lc_client.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_lc_client

        self.model = OpenAIModel(model_name="gpt-4")

    def test_generate_response(self):
        prompt = "Say hello."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(response, self._MOCK_RESPONSE)
        self.model.model.invoke.assert_called_once()


class TestOpenAIModelStreaming(unittest.TestCase):
    @patch("raglight.llm.openai_model.ChatOpenAI")
    @patch("raglight.Settings.OPENAI_API_KEY", "DUMMY_KEY")
    def setUp(self, MockChatOpenAI):
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = " world"

        self.mock_lc_client = MagicMock()
        self.mock_lc_client.stream.return_value = iter([chunk1, chunk2])
        MockChatOpenAI.return_value = self.mock_lc_client

        self.model = OpenAIModel(
            model_name="gpt-4",
            system_prompt="You are helpful.",
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
        list(self.model.generate_streaming({"question": "Follow-up.", "history": history}))
        call_args = self.mock_lc_client.stream.call_args
        messages = call_args[0][0]
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
