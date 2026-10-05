"""Provider-agnostic chat completion via LiteLLM."""
from typing import Protocol

import litellm

from src.config import Settings

litellm.suppress_debug_info = True  # keep error output readable


class ChatModel(Protocol):
    """Anything that turns a (system, user) prompt pair into text."""

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class LiteLLMClient:
    """ChatModel backed by LiteLLM (Groq, Ollama, OpenAI, ... chosen by config)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Run one chat completion and return the reply text ("" if empty)."""
        s = self._settings
        response = litellm.completion(
            model=s.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=s.llm_temperature,
            max_tokens=max_tokens,
            timeout=s.llm_timeout,
            api_key=s.llm_api_key.get_secret_value() if s.llm_api_key else None,
            api_base=s.llm_api_base or None,
        )
        return response.choices[0].message.content or ""
