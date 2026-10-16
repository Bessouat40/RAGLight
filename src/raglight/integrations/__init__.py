"""Lazy registries and optional built-in integrations."""

from ..registry import IntegrationRegistries, IntegrationRegistry


def _builtins() -> IntegrationRegistries:
    registries = IntegrationRegistries(
        llms=IntegrationRegistry("LLM"),
        embeddings=IntegrationRegistry("embedding"),
        vector_stores=IntegrationRegistry("vector store"),
        rerankers=IntegrationRegistry("reranker"),
        document_processors=IntegrationRegistry("document processor"),
    )

    for name, module, attribute in (
        ("bedrock", "raglight.integrations.bedrock", "BedrockLLM"),
        ("gemini", "raglight.integrations.gemini", "GeminiLLM"),
        ("lmstudio", "raglight.integrations.lmstudio", "LMStudioLLM"),
        ("mistral", "raglight.integrations.mistral", "MistralLLM"),
        ("ollama", "raglight.integrations.ollama", "OllamaLLM"),
        ("openai", "raglight.integrations.openai", "OpenAILLM"),
    ):
        registries.llms.register_lazy(name, module, attribute)

    for name, module, attribute in (
        ("bedrock", "raglight.integrations.bedrock", "BedrockEmbeddings"),
        ("gemini", "raglight.integrations.gemini", "GeminiEmbeddings"),
        (
            "huggingface",
            "raglight.integrations.huggingface",
            "HuggingFaceEmbeddings",
        ),
        ("ollama", "raglight.integrations.ollama", "OllamaEmbeddings"),
        ("openai", "raglight.integrations.openai", "OpenAIEmbeddings"),
    ):
        registries.embeddings.register_lazy(name, module, attribute)

    registries.vector_stores.register_lazy(
        "chroma", "raglight.integrations.chroma", "ChromaVectorStore"
    )
    registries.vector_stores.register_lazy(
        "qdrant", "raglight.integrations.qdrant", "QdrantVectorStore"
    )
    registries.rerankers.register_lazy(
        "huggingface",
        "raglight.integrations.huggingface",
        "HuggingFaceReranker",
    )

    for name, attribute in (
        ("pdf", "PDFProcessor"),
        ("c", "CodeProcessor"),
        ("cpp", "CodeProcessor"),
        ("cs", "CodeProcessor"),
        ("go", "CodeProcessor"),
        ("hs", "CodeProcessor"),
        ("java", "CodeProcessor"),
        ("js", "CodeProcessor"),
        ("lua", "CodeProcessor"),
        ("php", "CodeProcessor"),
        ("pl", "CodeProcessor"),
        ("py", "CodeProcessor"),
        ("rb", "CodeProcessor"),
        ("rs", "CodeProcessor"),
        ("scala", "CodeProcessor"),
        ("sol", "CodeProcessor"),
        ("swift", "CodeProcessor"),
        ("ts", "CodeProcessor"),
        ("html", "TextProcessor"),
        ("md", "TextProcessor"),
        ("txt", "TextProcessor"),
    ):
        registries.document_processors.register_lazy(
            name, "raglight.integrations.documents", attribute
        )
    return registries


default_registries = _builtins()

__all__ = ["IntegrationRegistries", "IntegrationRegistry", "default_registries"]
