from dataclasses import dataclass, field

from ..config.settings import Settings
from typing import Any, Dict, List, Optional
from ..models.data_source_model import DataSource


@dataclass(kw_only=True)
class AgenticRAGConfig:
    api_key: str = field(default="")
    api_base: Optional[str] = field(default=None)
    provider: str = field(default=Settings.OLLAMA)
    model: str = field(default=Settings.DEFAULT_LLM)
    num_ctx: int = field(default=8192)
    k: int = field(default=5)
    mcp_config: Optional[Dict[str, Dict[str, Any]]] = field(default=None)
    verbosity_level: int = field(default=2)
    max_steps: int = field(default=4)
    system_prompt: str = field(default=Settings.DEFAULT_AGENT_PROMPT)
    knowledge_base: Optional[List[DataSource]] = field(default=None)
    ignore_folders: list = field(
        default_factory=lambda: list(Settings.DEFAULT_IGNORE_FOLDERS)
    )
