"""Local Qdrant storage manager with a pluggable, FastEmbed-compatible embedder."""
import threading
from collections.abc import Iterable
from typing import Any, Protocol

from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import Settings
from src.ingestion.chunker import Chunk

_BATCH_SIZE = 64
_SCROLL_PAGE = 256


class Embedder(Protocol):
    """The slice of FastEmbed's TextEmbedding API that VectorStore relies on."""

    def embed(self, documents: list[str]) -> Iterable[Any]: ...

    def query_embed(self, query: str) -> Iterable[Any]: ...


class SearchHit(BaseModel):
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Owns the Qdrant client and the embedding model for one collection.

    Local Qdrant mode is single-process and not built for concurrent callers, so
    every client call runs under one re-entrant lock. LLM calls never run under it.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        if embedder is not None and embedding_dim is None:
            raise ValueError("embedding_dim is required when a custom embedder is supplied")
        self._settings = settings
        self._lock = threading.RLock()
        self._closed = False
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(settings.storage_path))  # on-disk, no Docker
        if embedder is None:
            self._embedder: Embedder = TextEmbedding(model_name=settings.embedding_model)
            self._dim = TextEmbedding.get_embedding_size(settings.embedding_model)
        else:
            self._embedder, self._dim = embedder, int(embedding_dim or 0)

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection if missing; with ``recreate=True`` wipe it first."""
        name = self._settings.collection_name
        with self._lock:
            if not self._client.collection_exists(name):
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
            elif recreate:
                self._client.delete(
                    collection_name=name,
                    points_selector=FilterSelector(filter=Filter()),
                )

    def upsert_documents(self, chunks: list[Chunk]) -> int:
        """Embed chunks and upsert them in batches. Returns the number stored."""
        with self._lock:
            for start in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[start : start + _BATCH_SIZE]
                vectors = self._embedder.embed([c.text for c in batch])  # lazy generator
                points = [
                    PointStruct(
                        id=c.chunk_id,
                        vector=vec.tolist(),
                        payload=c.model_dump(exclude={"chunk_id"}),
                    )
                    for c, vec in zip(batch, vectors)
                ]
                self._client.upsert(self._settings.collection_name, points=points)
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of ``doc_id`` so re-ingesting a document replaces it."""
        with self._lock:
            self._client.delete(
                self._settings.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                ),
            )

    def dense_search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Embed ``query`` and return the ``top_k`` most similar chunks."""
        with self._lock:
            query_vec = next(iter(self._embedder.query_embed(query)))
            result = self._client.query_points(
                collection_name=self._settings.collection_name,
                query=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
            )
        return [
            SearchHit(chunk=Chunk(chunk_id=str(p.id), **(p.payload or {})), score=p.score)
            for p in result.points
        ]

    def get_all_chunks(self) -> list[Chunk]:
        """Scroll through the whole collection (used to build the BM25 index)."""
        chunks: list[Chunk] = []
        offset = None
        with self._lock:
            while True:
                points, offset = self._client.scroll(
                    self._settings.collection_name,
                    limit=_SCROLL_PAGE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                chunks.extend(Chunk(chunk_id=str(p.id), **(p.payload or {})) for p in points)
                if offset is None:
                    return chunks

    def count_chunks(self) -> int:
        """Exact number of chunks currently stored."""
        with self._lock:
            return self._client.count(self._settings.collection_name, exact=True).count

    def close(self) -> None:
        """Release the on-disk lock so another process can open the store (idempotent)."""
        with self._lock:
            if not self._closed:
                self._client.close()
                self._closed = True