"""Hybrid retrieval: dense + BM25 -> RRF -> optional cross-encoder rerank."""
from pydantic import BaseModel

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.storage.vector_store import VectorStore


class HybridHit(BaseModel):
    """A result plus the evidence for why it ranked where it did."""

    chunk: Chunk
    rrf_score: float
    dense_rank: int | None = None  # 1-based rank in dense results, None if absent
    bm25_rank: int | None = None  # 1-based rank in BM25 results, None if absent
    rerank_score: float | None = None


class HybridRetriever:
    """Combines semantic and lexical search, then reranks the fused shortlist."""

    def __init__(
        self,
        store: VectorStore,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._reranker = reranker or CrossEncoderReranker(settings.reranker_model)
        self._bm25 = BM25Index(store.get_all_chunks())

    def refresh_index(self) -> None:
        """Rebuild the BM25 index. Call after ingesting new documents."""
        self._bm25 = BM25Index(self._store.get_all_chunks())

    def retrieve(
        self, query: str, top_k: int | None = None, rerank: bool = True
    ) -> list[HybridHit]:
        """Return the ``top_k`` best chunks for ``query``."""
        limit = self._settings.final_top_k if top_k is None else top_k
        n = self._settings.candidate_k

        dense = self._store.dense_search(query, top_k=n)
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
            return hits[:limit]

        scores = self._reranker.score(query, [h.chunk for h in hits])
        ranked = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
        return [h.model_copy(update={"rerank_score": s}) for s, h in ranked[:limit]]
