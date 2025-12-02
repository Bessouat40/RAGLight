from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import Settings


@dataclass(kw_only=True)
class GraphRAGConfig:
    uri: str
    username: str
    password: str
    database: Optional[str] = None
    cypher_prompt: Optional[str] = None
    api_base: str = field(default=Settings.DEFAULT_OLLAMA_CLIENT)
    llm: str = field(default=Settings.DEFAULT_LLM)
    provider: str = field(default=Settings.OLLAMA)
    system_prompt: str = field(default=Settings.DEFAULT_SYSTEM_PROMPT)
    stream: bool = field(default=False)
