"""Deterministic stand-ins for the embedder, reranker and LLM (no network, no model downloads)."""
import hashlib
import re
from collections.abc import Iterable, Iterator

import numpy as np

from src.config import DEFAULT_HYDE_PROMPT
from src.ingestion.chunker import Chunk

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts that share words land close together."""

    DIM = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, documents: list[str]) -> Iterable[np.ndarray]:
        return (self._vec(t) for t in documents)

    def query_embed(self, query: str) -> Iterable[np.ndarray]:
        return iter([self._vec(query)])


class FakeReranker:
    """Scores by query-word overlap and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        self.calls.append((query, len(chunks)))
        q = set(words(query))
        return [float(len(q & set(words(c.text)))) for c in chunks]


class FakeLLM:
    """Scriptable ChatModel. By default it answers by citing the first source label."""

    def __init__(
        self,
        answer: str | None = None,
        hyde: str = "sourdough starter flour yeast",
        fail_hyde: bool = False,
        fail_answer: bool = False,
        fail_stream_after: int | None = None,
    ) -> None:
        self.answer = answer
        self.hyde = hyde
        self.fail_hyde = fail_hyde
        self.fail_answer = fail_answer
        self.fail_stream_after = fail_stream_after
        self.calls: list[tuple[str, str, str]] = []  # (kind, system, user)

    def _text(self, user: str) -> str:
        if self.answer is not None:
            return self.answer
        match = re.search(r"SOURCES:\n\[([^\]]+)\]", user)
        return f"The sources state the relevant facts [{match.group(1)}]." if match else "No sources."

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if system == DEFAULT_HYDE_PROMPT:
            self.calls.append(("hyde", system, user))
            if self.fail_hyde:
                raise RuntimeError("hyde unavailable")
            return self.hyde
        self.calls.append(("answer", system, user))
        if self.fail_answer:
            raise RuntimeError("llm unavailable")
        return self._text(user)

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        self.calls.append(("answer", system, user))
        if self.fail_answer:
            raise RuntimeError("llm unavailable")
        for i, token in enumerate(re.findall(r"\S+\s*", self._text(user))):
            if self.fail_stream_after is not None and i >= self.fail_stream_after:
                raise RuntimeError("stream broke")
            yield token
