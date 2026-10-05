"""Hybrid retrieval: [HyDE] -> dense + BM25 -> RRF -> optional cross-encoder rerank."""
from pydantic import BaseModel

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hyde import HydeExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.vector_store import VectorStore


class HybridHit(BaseModel):
    """A result plus the evidence for why it ranked where it did."""

    chunk: Chunk
    rrf_score: float
    dense_rank: int | None = None  # 1-based rank in dense results, None if absent
    bm25_rank: int | None = None  # 1-based rank in BM25 results, None if absent
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """Ranked hits plus the HyDE passage that was used (None if HyDE was off/failed)."""

    hits: list[HybridHit]
    hyde_passage: str | None = None


class HybridRetriever:
    """Combines semantic and lexical search, then reranks the fused shortlist."""

    def __init__(
        self,
        store: VectorStore,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
        hyde: HydeExpander | None = None,
    ) -> None:
        if settings.use_hyde and hyde is None:
            raise ValueError("settings.use_hyde is true but no HydeExpander was provided")
        self._store = store
        self._settings = settings
        self._reranker = reranker or CrossEncoderReranker(settings.reranker_model)
        self._hyde = hyde
        self._bm25 = BM25Index(store.get_all_chunks())

    def refresh_index(self) -> None:
        """Rebuild the BM25 index. Call after ingesting new documents."""
        self._bm25 = BM25Index(self._store.get_all_chunks())

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = True,
        use_hyde: bool | None = None,
    ) -> RetrievalResult:
        """Return the ``top_k`` best chunks for ``query``.

        ``use_hyde=None`` defers to ``settings.use_hyde``. HyDE only changes the
        *dense* query; BM25 and the reranker always see the user's original words.
        """
        limit = self._settings.final_top_k if top_k is None else top_k
        n = self._settings.candidate_k
        want_hyde = self._settings.use_hyde if use_hyde is None else use_hyde

        passage: str | None = None
        if want_hyde:
            if self._hyde is None:
                raise ValueError("HyDE requested but no HydeExpander was provided")
            passage = self._hyde.expand(query)
        dense_query = f"{query}\n\n{passage}" if passage else query

        dense = self._store.dense_search(dense_query, top_k=n)
        lexical = self._bm25.search(query, top_k=n)

        chunks = {h.chunk.chunk_id: h.chunk for h in (*dense, *lexical)}
        dense_rank = {h.chunk.chunk_id: r for r, h in enumerate(dense, start=1)}
        bm25_rank = {h.chunk.chunk_id: r for r, h in enumerate(lexical, start=1)}
        fused = reciprocal_rank_fusion(
            [list(dense_rank), list(bm25_rank)], k=self._settings.rrf_k
        )

        hits = [
            HybridHit(
                chunk=chunks[cid],
                rrf_score=score,
                dense_rank=dense_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
            )
            for cid, score in fused[:n]  # shortlist size caps reranker cost
        ]
        if not rerank or not hits:
            return RetrievalResult(hits=hits[:limit], hyde_passage=passage)

        scores = self._reranker.score(query, [h.chunk for h in hits])
        ranked = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
        return RetrievalResult(
            hits=[h.model_copy(update={"rerank_score": s}) for s, h in ranked[:limit]],
            hyde_passage=passage,
        )
