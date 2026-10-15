import unittest
from unittest.mock import MagicMock, patch

from raglight.llm.gemini_model import GeminiModel
from ..test_config import TestsConfig


class TestGeminiModel(unittest.TestCase):
    _MOCK_RESPONSE = "Hello! This is a test response."

    @patch("raglight.llm.gemini_model.ChatGoogleGenerativeAI")
    @patch("raglight.llm.gemini_model.Settings.GEMINI_API_KEY", "DUMMY_KEY")
    def setUp(self, MockChatGemini):
        mock_lc_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self._MOCK_RESPONSE
        mock_lc_client.invoke.return_value = mock_response
        MockChatGemini.return_value = mock_lc_client

        self.model = GeminiModel(model_name=TestsConfig.GEMINI_LLM_MODEL)

    def test_generate_response(self):
        prompt = "Say hello."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(response, self._MOCK_RESPONSE)
        self.model.model.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
