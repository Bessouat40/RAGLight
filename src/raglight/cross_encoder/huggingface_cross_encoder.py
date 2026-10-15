from __future__ import annotations
from typing import List
from .cross_encoder_model import CrossEncoderModel

try:
    from typing_extensions import override
    from sentence_transformers import CrossEncoder
except ModuleNotFoundError as error:
    from .._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "sentence_transformers"),
        extra="huggingface",
        feature="Hugging Face reranking support",
    )


class HuggingfaceCrossEncoderModel(CrossEncoderModel):
    """
    Concrete implementation of the CrossEncoderModel for HuggingFace models.

    This class provides a specific implementation of the abstract `CrossEncoderModel` for
    loading and using HuggingFace cross encoder.

    Attributes:
        model_name (str): The name of the HuggingFace model to be loaded.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes a HuggingfaceCrossEncoderModel instance.

        Args:
            model_name (str): The name of the HuggingFace model to load.
        """
        super().__init__(model_name)

    @override
    def load(self) -> HuggingfaceCrossEncoderModel:
        """
        Loads the HuggingFace cross encoder model.

        This method overrides the abstract `load` method from the `CrossEncoderModel` class
        and initializes the HuggingFace cross encoder model with the specified `model_name`.

        Returns:
            HuggingfaceCrossEncoderModel: The loaded HuggingFace cross encoder model.
        """
        return CrossEncoder(self.model_name)

    @override
    def predict(self, query: str, documents: List[str], top_k: int) -> List[str]:
        """
        Predicts the similarity scores and returns the list of most relevant document texts.

        Args:
            query (str): The input query.
            documents (List[str]): The list of document texts to rank.
            top_k (int): The number of top results to return.

        Returns:
            List[str]: The list of top_k re-ranked document texts.
        """
        # rank returns a list of dicts: [{'corpus_id': int, 'score': float, 'text': str}, ...]
        results = self.model.rank(
            query=query, documents=documents, top_k=top_k, return_documents=True
        )

        # We extract and return only the text strings
        return [res["text"] for res in results]
