"""Cross-encoder reranking via FastEmbed (ONNX, CPU-only)."""
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.ingestion.chunker import Chunk


class Reranker(Protocol):
    """Anything that scores chunks against a query (higher = more relevant)."""

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly, which is slower but sharper than vectors."""

    def __init__(self, model_name: str) -> None:
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        """Return one relevance score per chunk, aligned with the input order."""
        if not chunks:
            return []
        return [float(s) for s in self._model.rerank(query, [c.text for c in chunks])]
