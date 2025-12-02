from ..config.graph_rag_config import GraphRAGConfig
from ..rag.builder import Builder
from ..rag.graph_rag import GraphRAG


class GraphRAGPipeline:
    """Pipeline wrapper to run Graph RAG with Neo4j."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.graph_rag: GraphRAG = (
            Builder()
            .with_llm(
                config.provider,
                model_name=config.llm,
                system_prompt=config.system_prompt,
                api_base=config.api_base,
            )
            .with_graph_store(
                uri=config.uri,
                username=config.username,
                password=config.password,
                database=config.database,
            )
            .build_graph_rag(cypher_prompt=config.cypher_prompt, stream=config.stream)
        )

    def generate(self, question: str):
        return self.graph_rag.generate(question)
