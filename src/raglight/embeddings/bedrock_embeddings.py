from __future__ import annotations
from typing import Optional, List

try:
    from typing_extensions import override
    from langchain_aws import BedrockEmbeddings
except ModuleNotFoundError as error:
    from .._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_aws"),
        extra="bedrock",
        feature="AWS Bedrock embeddings support",
    )

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel


class BedrockEmbeddingsModel(EmbeddingsModel):
    """
    Concrete implementation of EmbeddingsModel for AWS Bedrock embedding models.

    Authentication relies on the standard boto3 credential chain:
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    - AWS config/credentials files (~/.aws/)
    - IAM instance role (when running on EC2/ECS/Lambda)

    Attributes:
        model_name (str): The Bedrock embedding model ID (e.g. 'amazon.titan-embed-text-v2:0').
        region_name (str): AWS region where Bedrock is available.
        model (BedrockEmbeddings): The LangChain BedrockEmbeddings client.
    """

    def __init__(self, model_name: str, region_name: Optional[str] = None) -> None:
        """
        Initializes a BedrockEmbeddingsModel instance.

        Args:
            model_name (str): The Bedrock embedding model ID.
            region_name (Optional[str]): AWS region. Defaults to AWS_DEFAULT_REGION env var or 'us-east-1'.
        """
        self.region_name = region_name or Settings.AWS_DEFAULT_REGION
        super().__init__(model_name)

    @override
    def load(self) -> BedrockEmbeddings:
        """
        Loads the BedrockEmbeddings client.

        Returns:
            BedrockEmbeddings: The LangChain Bedrock embeddings client.
        """
        return BedrockEmbeddings(
            model_id=self.model_name,
            region_name=self.region_name,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the Bedrock embedding model.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        return self.model.embed_query(text)
