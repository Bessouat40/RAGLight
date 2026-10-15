from __future__ import annotations
import base64
from typing import Iterable, Mapping, Optional, Dict, Any
from ..config.settings import Settings
from .llm import LLM
import logging

try:
    from typing_extensions import override
    from langchain_ollama import ChatOllama
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ModuleNotFoundError as error:
    from .._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_ollama", "langchain_core"),
        extra="ollama",
        feature="Ollama support",
    )

OLLAMA_DEFAULT_CONTEXT_SIZE = 4096
OLLAMA_OPTION_CONTEXT_SIZE = "num_ctx"


class OllamaModel(LLM):
    def __init__(
        self,
        model_name: str,
        options: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        preload_model: Optional[bool] = True,
        api_base: Optional[str] = None,
        role: str = "user",
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_base = api_base or Settings.DEFAULT_OLLAMA_CLIENT
        self.headers = headers
        self.preload_model = preload_model
        self.options = options or {}
        super().__init__(model_name, system_prompt, system_prompt_file, self.api_base)
        logging.info(f"Using Ollama with {model_name} model 🤖")
        self.role: str = role

    @override
    def load(self) -> ChatOllama:
        model = ChatOllama(
            model=self.model_name,
            base_url=self.api_base,
            headers=self.headers,
            **self.options,
        )
        if self.preload_model:
            try:
                model.invoke([HumanMessage(content="hi")])
            except Exception:
                pass
        return model

    def _build_messages(self, input: Dict[str, Any]):
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        for msg in input.get("history", []):
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))

        question = input.get("question", "")
        if "images" in input:
            content = [{"type": "text", "text": question}]
            for img in input["images"]:
                b64 = (
                    base64.b64encode(img["bytes"]).decode()
                    if isinstance(img.get("bytes"), bytes)
                    else img.get("base64", "")
                )
                content.append(
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
                )
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=question))
        return messages

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        response = self.model.invoke(self._build_messages(input))
        return response.content

    @override
    def generate_streaming(
        self, input: Dict[str, Any], callbacks=None
    ) -> Iterable[str]:
        config = {"callbacks": callbacks} if callbacks else {}
        for chunk in self.model.stream(self._build_messages(input), config=config):
            if chunk.content:
                yield chunk.content
