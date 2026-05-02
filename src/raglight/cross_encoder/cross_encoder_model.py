from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class RerankResult:
    """
    Represents a single reranked document result with metadata and score.
    
    Attributes:
        text (str): The document text content.
        score (float): The relevance score from the cross encoder.
        corpus_id (int): The original index in the input document list.
        metadata (Optional[dict]): Original document metadata (source, etc.).
    """
    text: str
    score: float
    corpus_id: int
    metadata: Optional[dict] = None


class CrossEncoderModel(ABC):
    """
    Abstract base class for cross encoder models.

    This class defines a blueprint for implementing cross encoder models with a consistent interface for
    loading and retrieving the model.

    Attributes:
        model_name (str): The name of the model.
        model (Any): The loaded model instance.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes an CrossEncoderModel instance.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.model_name: str = model_name
        self.model: Any = self.load()

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to load the cross encoder model.

        This method must be implemented by any concrete subclass to define the loading process
        for the specific model.

        Returns:
            Any: The loaded model instance.
        """
        pass

    def get_model(self) -> CrossEncoderModel:
        """
        Retrieves the loaded cross encoder model.

        Returns:
            Any: The loaded model instance.
        """
        return self.model

    @abstractmethod
    def predict(self, query: str, documents: List[str], top_k: int) -> List[RerankResult]:
        """
        Re-ranks the given documents against the query and returns the top_k most relevant.

        Args:
            query (str): The input query.
            documents (List[str]): The list of document texts to rank.
            top_k (int): The number of top results to return.

        Returns:
            List[RerankResult]: The top_k re-ranked results with scores and corpus IDs.
        """
        pass
