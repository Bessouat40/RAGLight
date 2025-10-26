import unittest
from unittest.mock import MagicMock, patch

from ollama import ChatResponse, Message
from raglight.llm.ollama_model import OllamaModel
from ..test_config import TestsConfig


class TestOllamaModel(unittest.TestCase):
    def setUp(self):
        mock_ollama_client = MagicMock()

        self.model = OllamaModel(
            model_name=TestsConfig.OLLAMA_MODEL,
            system_prompt_file=TestsConfig.TEST_SYSTEM_PROMPT,
            options={"temperature": 0.3},
        )

        message: Message = Message(
            role="assistant",
            content="Machine learning (ML) is a subset of artificial intelligence",
        )
        chat_response: ChatResponse = ChatResponse(message=message)
        mock_ollama_client.chat = MagicMock(return_value=chat_response)
        self.model.model = mock_ollama_client

    def test_generate_response(self):
        prompt = "Define machine learning."
        response = self.model.generate({"question": prompt})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(
            response, "Machine learning (ML) is a subset of artificial intelligence"
        )
        self.model.model.chat.assert_called_once_with(
            model=TestsConfig.OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": '{"question": "Define machine learning.", "system prompt": "This is just a test prompt"}',
                },
            ],
            options={"temperature": 0.3},
        )


if __name__ == "__main__":
    unittest.main()
