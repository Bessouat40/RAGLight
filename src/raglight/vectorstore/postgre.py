import logging
from typing import List, Dict
from typing_extensions import override
from langchain_core.documents import Document

from ..config.settings import Settings

from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel

from enum import auto
from langchain_postgres import PGVector

class PostgreVS(VectorStore):
    """
    Concrete implementation for PostgreVDB.

    It inherits the main ingestion logic from the base VectorStore class and
    only implements the Postgre-specific methods for adding documents and
    performing searches.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        host: str = Settings.DEFAULT_POSTGRE_HOST,
        port: int = Settings.DEFAULT_POSTGRE_PORT,
        database: str = Settings.DEFAULT_POSTGRE_DATABASE,
        user: str = Settings.DEFAULT_POSTGRE_DATABASE,
        password: str = Settings.DEFAULT_POSTGRE_DATABASE
        
    ) -> None:
        """
        Initializes a PostgreVS instance.
        """
        super().__init__(persist_directory, embeddings_model)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.database = database
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        if host and port:
            
            self.vector_store = PGVector(
                collection_name=collection_name,
                connection=connection_string,
                embeddings=embeddings_model,
                use_jsonb=True,
            )
            
            self.vector_store_classes = PGVector(
                collection_name=f"{collection_name}_classes",
                connection=connection_string,
                embeddings=self.embeddings_model,
                use_jsonb=True,
            )
        else:
            raise ValueError(
                "Invalid configuration for PostgreVS: "
                "You must either:\n"
                "  • Provide both host and port (for PostgreDB), OR\n"
                "  • Provide both user and password  (for PostgreDB).\n"
                "  • Provide exist database   (for PostgreDB).\n"

                f"Received -> host={host}, port={port}, user={user}, password={'*' * len(password) if password else None}, database={database}"
            )

    @override
    def add_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add documents specifically to the main PostgreDB collection,
        using batching for efficiency.
        """
        if not documents:
            return

        # logging.info(
        #     f"⏳ Adding {len(documents)} document chunks to PostgreDB collection '{self.vector_store._collection_name}'..."
        # )
        self.vector_store.add_documents(documents=documents)
        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add class signature documents to the dedicated PostgreDB
        collection for classes.
        """
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to PostgreDB collection '{self.vector_store_classes._collection_name}'..."
        )
        self.vector_store_classes.add_documents(documents=documents)
        logging.info("✅ Class documents successfully added to the class collection.")

    @override
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Dict[str, str] = None,
        collection_name: str = None,
    ) -> List[Document]:
        """
        Implements similarity search using the main PostgreDB client.
        """
        return self.vector_store.similarity_search(question, k=k, filter=filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Dict[str, str] = None,
        collection_name: str = None,
    ) -> List[Document]:
        """
        Implements similarity search using the dedicated class PostgreDB client.
        """
        pass