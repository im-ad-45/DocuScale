"""HyDE: Hypothetical Document Embeddings query expansion."""
import logging

from src.config import Settings
from src.generation.llm import ChatModel

logger = logging.getLogger(__name__)


class HydeExpander:
    """Asks the LLM to write a passage that *would* answer the query."""

    def __init__(self, llm: ChatModel, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def expand(self, query: str) -> str | None:
        """Return a hypothetical answer passage, or None if generation fails.

        Failure must never take retrieval down: callers fall back to the raw query.
        """
        try:
            passage = self._llm.complete(
                system=self._settings.hyde_prompt,
                user=query,
                max_tokens=self._settings.hyde_max_tokens,
            ).strip()
        except Exception:  # LiteLLM raises many provider-specific error types
            logger.warning("HyDE generation failed; using the raw query", exc_info=True)
            return None
        return passage or None
