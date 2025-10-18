import logging
from typing import List, Dict
from typing_extensions import override
from langchain_core.documents import Document

from ..config.settings import Settings

from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel

from enum import auto
from langchain_postgres import PGVector

class PostgresVS(VectorStore):
    """
    Concrete implementation for PostgresVDB.

    It inherits the main ingestion logic from the base VectorStore class and
    only implements the Postgres-specific methods for adding documents and
    performing searches.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        host: str = Settings.POSTGRES_HOST,
        port: int = Settings.POSTGRES_PORT,
        database: str = Settings.POSTGRES_DATABASE,
        user: str = Settings.POSTGRES_USER,
        password: str = Settings.POSTGRES_PASSWORD
        
    ) -> None:
        """
        Initializes a PostgresVS instance.
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
                "Invalid configuration for PostgresVS: "
                "You must either:\n"
                "  • Provide both host and port (for PostgresDB), OR\n"
                "  • Provide both user and password  (for PostgreDsB).\n"
                "  • Provide exist database   (for PostgresDB).\n"

                f"Received -> host={host}, port={port}, user={user}, password={'*' * len(password) if password else None}, database={database}"
            )

    @override
    def add_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add documents specifically to the main PostgresDB collection,
        using batching for efficiency.
        """
        if not documents:
            return

        # logging.info(
        #     f"⏳ Adding {len(documents)} document chunks to PostgresDB collection '{self.vector_store._collection_name}'..."
        # )
        self.vector_store.add_documents(documents=documents)
        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        """
        Implements the logic to add class signature documents to the dedicated PostgresDB
        collection for classes.
        """
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to PostgresDB collection '{self.vector_store_classes._collection_name}'..."
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
        Implements similarity search using the main PostgresDB client.
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
        Implements similarity search using the dedicated class PostgresDB client.
        """
        pass