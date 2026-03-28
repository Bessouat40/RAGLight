from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from typing_extensions import override
from ..config.settings import Settings
from .llm import LLM
import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class MiniMaxModel(LLM):
    """MiniMax LLM provider using the OpenAI-compatible API.

    MiniMax exposes an OpenAI-compatible chat completions endpoint at
    ``https://api.minimax.io/v1``.  This class reuses ``ChatOpenAI``
    from *langchain-openai* so that the MiniMax models (MiniMax-M2.7,
    MiniMax-M2.5, etc.) integrate seamlessly with the RAGLight pipeline.

    Temperature is clamped to ``(0, 1]`` as required by the MiniMax API,
    and ``<think>…</think>`` reasoning tags are automatically stripped
    from the output.
    """

    # Pattern used to remove chain-of-thought tags emitted by some
    # MiniMax reasoning models (e.g. MiniMax-M2.7).
    _THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        api_base: Optional[str] = None,
        role: str = "user",
        temperature: float = 0.7,
    ) -> None:
        self.api_base = api_base or Settings.DEFAULT_MINIMAX_CLIENT
        self.temperature = max(temperature, 0.01)  # MiniMax requires > 0
        super().__init__(model_name, system_prompt, system_prompt_file, self.api_base)
        logging.info(f"Using MiniMax with {model_name} model 🤖")
        self.role: str = role

    @override
    def load(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=Settings.MINIMAX_API_KEY,
            base_url=self.api_base,
            temperature=self.temperature,
        )

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove ``<think>…</think>`` blocks from model output."""
        return MiniMaxModel._THINK_PATTERN.sub("", text).strip()

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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image['base64']}"
                            },
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
        return self._strip_think_tags(response.content)

    @override
    def generate_streaming(
        self, input: Dict[str, Any], callbacks=None
    ) -> Iterable[str]:
        config = {"callbacks": callbacks} if callbacks else {}
        for chunk in self.model.stream(self._build_messages(input), config=config):
            if chunk.content:
                yield chunk.content
