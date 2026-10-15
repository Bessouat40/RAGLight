from __future__ import annotations
import os
from typing import Iterable, Optional, Dict, Any
from ._base import LLMAdapter
import logging

try:
    from typing_extensions import override
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ModuleNotFoundError as error:
    from ..._optional import optional_import_error

    optional_import_error(
        error,
        dependency=("typing_extensions", "langchain_mistralai", "langchain_core"),
        extra="mistral",
        feature="Mistral support",
    )


class MistralLLM(LLMAdapter):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        api_base: str = None,
        api_key: Optional[str] = None,
        role: str = "user",
    ) -> None:
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        super().__init__(model_name, system_prompt, system_prompt_file)
        logging.info(f"Using Mistral with {model_name} model 🤖")
        self.role: str = role

    @override
    def load(self) -> ChatMistralAI:
        return ChatMistralAI(
            model=self.model_name,
            api_key=self.api_key,
        )

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
            for image in input["images"]:
                try:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image['base64']}",
                        }
                    )
                except Exception as e:
                    logging.error(f"Could not read image: {e}")
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
