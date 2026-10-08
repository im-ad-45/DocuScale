from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.services import Services
from src.config import Settings
from src.generation.synthesizer import Synthesizer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hyde import HydeExpander
from src.storage.vector_store import VectorStore
from tests.corpus import index_corpus
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings: temp storage, tiny chunks, no env / .env involvement."""
    return Settings(
        storage_path=tmp_path / "qdrant",
        collection_name="test",
        chunk_size=40,
        chunk_overlap=10,
        candidate_k=10,
        final_top_k=3,
    )


@pytest.fixture
def store(settings) -> Iterator[VectorStore]:
    s = VectorStore(settings, embedder=FakeEmbedder(), embedding_dim=FakeEmbedder.DIM)
    s.create_collection()
    yield s
    s.close()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def indexed_store(store, settings) -> VectorStore:
    index_corpus(store, settings)
    return store


@pytest.fixture
def retriever(indexed_store, settings, llm, reranker) -> HybridRetriever:
    return HybridRetriever(indexed_store, settings, reranker=reranker, hyde=HydeExpander(llm, settings))


@pytest.fixture
def synthesizer(llm, settings) -> Synthesizer:
    return Synthesizer(llm, settings)


@pytest.fixture
def services(settings, store, llm, reranker) -> Services:
    """Pipeline over an EMPTY store (API tests ingest through /ingest)."""
    retriever = HybridRetriever(store, settings, reranker=reranker, hyde=HydeExpander(llm, settings))
    return Services(settings, store, retriever, Synthesizer(llm, settings))


@pytest.fixture
def client(services) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as c:
        yield c
