"""Local Qdrant storage manager with built-in FastEmbed embedding."""
from typing import Iterator

from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import Settings
from src.ingestion.chunker import Chunk

_BATCH_SIZE = 64


class SearchHit(BaseModel):
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Owns the Qdrant client and the embedding model for one collection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(settings.storage_path))  # on-disk, no Docker
        self._embedder = TextEmbedding(model_name=settings.embedding_model)
        self._dim = TextEmbedding.get_embedding_size(settings.embedding_model)

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection if missing; with ``recreate=True`` wipe it first."""
        name = self._settings.collection_name
        if recreate and self._client.collection_exists(name):
            self._client.delete_collection(name)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def upsert_documents(self, chunks: list[Chunk]) -> int:
        """Embed chunks and upsert them in batches. Returns the number stored."""
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

    def dense_search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Embed ``query`` and return the ``top_k`` most similar chunks."""
        query_vec: Iterator = self._embedder.query_embed(query)
        result = self._client.query_points(
            collection_name=self._settings.collection_name,
            query=next(query_vec).tolist(),
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
        while True:
            points, offset = self._client.scroll(
                self._settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(Chunk(chunk_id=str(p.id), **(p.payload or {})) for p in points)
            if offset is None:
                return chunks

    def close(self) -> None:
        """Release the on-disk lock so another process can open the store."""
        self._client.close()
