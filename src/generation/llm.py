"""Provider-agnostic chat completion via LiteLLM (blocking and streaming)."""
from collections.abc import Iterator
from typing import Any, Protocol

import litellm

from src.config import Settings

litellm.suppress_debug_info = True  # keep error output readable


class ChatModel(Protocol):
    """Anything that turns a (system, user) prompt pair into text."""

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]: ...


class LiteLLMClient:
    """ChatModel backed by LiteLLM (Groq, Ollama, OpenAI, ... chosen by config)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _request(self, system: str, user: str, max_tokens: int, stream: bool) -> Any:
        s = self._settings
        return litellm.completion(
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
            stream=stream,
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Run one chat completion and return the reply text ("" if empty)."""
        response = self._request(system, user, max_tokens, stream=False)
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """Yield reply text fragments as the provider produces them.

        The request is sent on the first ``next()``, so connection and auth
        errors surface there, not when the generator is created.
        """
        for chunk in self._request(system, user, max_tokens, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
