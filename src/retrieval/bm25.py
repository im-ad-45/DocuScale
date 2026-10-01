"""In-memory BM25 lexical index over chunks."""
import re

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.storage.vector_store import SearchHit

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters (no stemming, no stopwords)."""
    return _WORD.findall(text.lower())


class BM25Index:
    """Keyword index; rebuild it whenever the underlying chunks change."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self._model: BM25Okapi | None = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return up to ``top_k`` chunks sharing at least one term with ``query``."""
        if self._model is None:
            return []
        scores = self._model.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:top_k]
        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0  # zero means no query term matched at all
        ]
