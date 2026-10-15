import unittest
from unittest.mock import MagicMock, patch

from raglight.integrations.llms.bedrock_model import BedrockLLM
from ..test_config import TestsConfig


class TestBedrockLLM(unittest.TestCase):
    _MOCK_RESPONSE = "Hello! This is a test Bedrock response."

    @patch("raglight.integrations.llms.bedrock_model.ChatBedrock")
    def setUp(self, mock_chat_bedrock: MagicMock):
        mock_chat_bedrock.return_value = MagicMock()
        self.model = BedrockLLM(
            model_name=TestsConfig.BEDROCK_LLM_MODEL,
            region_name="us-east-1",
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
        # First message should be the system prompt, last should be human message
        from langchain_core.messages import SystemMessage, HumanMessage

        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertIsInstance(call_args[-1], HumanMessage)


if __name__ == "__main__":
    unittest.main()
