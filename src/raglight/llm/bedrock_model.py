from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
import logging

try:
    from typing_extensions import override
    from langchain_aws import ChatBedrock
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ModuleNotFoundError as error:
    from .._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_aws", "langchain_core"),
        extra="bedrock",
        feature="AWS Bedrock support",
    )

from ..config.settings import Settings
from .llm import LLM


class BedrockModel(LLM):
    """
    A subclass of LLM that uses AWS Bedrock as the backend for text generation.

    Authentication relies on the standard boto3 credential chain:
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    - AWS config/credentials files (~/.aws/)
    - IAM instance role (when running on EC2/ECS/Lambda)

    Attributes:
        model_name (str): The Bedrock model ID (e.g. 'anthropic.claude-3-5-sonnet-20241022-v2:0').
        region_name (str): AWS region where Bedrock is available.
        system_prompt (str): The system prompt used for generation.
        model (ChatBedrock): The LangChain ChatBedrock client.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        """
        Initializes a BedrockModel instance.

        Args:
            model_name (str): The Bedrock model ID.
            system_prompt (Optional[str]): System prompt text. Falls back to default if not provided.
            system_prompt_file (Optional[str]): Path to a file containing the system prompt.
            region_name (Optional[str]): AWS region. Defaults to AWS_DEFAULT_REGION env var or 'us-east-1'.
        """
        self.region_name = region_name or Settings.AWS_DEFAULT_REGION
        super().__init__(model_name, system_prompt, system_prompt_file)
        logging.info(f"Using AWS Bedrock with {model_name} model 🤖")

    @override
    def load(self) -> ChatBedrock:
        """
        Loads the ChatBedrock client.

        Returns:
            ChatBedrock: The LangChain Bedrock chat client.
        """
        return ChatBedrock(
            model_id=self.model_name,
            region_name=self.region_name,
        )

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        """
        Generates text using the Bedrock model via LangChain.

        Args:
            input (Dict[str, Any]): Must contain a 'question' key with the user query.

        Returns:
            str: The generated response text.
        """
        history = input.get("history", [])
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for msg in history:
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=input.get("question", "")))
        response = self.model.invoke(messages)
        return response.content

    @override
    def generate_streaming(
        self, input: Dict[str, Any], callbacks=None
    ) -> Iterable[str]:
        history = input.get("history", [])
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        for msg in history:
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=input.get("question", "")))

        stream_config = {"callbacks": callbacks} if callbacks else {}
        for chunk in self.model.stream(messages, config=stream_config):
            if chunk.content:
                yield chunk.content
