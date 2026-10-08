import pytest

from src.ingestion.chunker import chunk_document
from src.storage.vector_store import VectorStore
from tests.corpus import chunk_corpus, index_corpus
from tests.fakes import FakeEmbedder


def test_custom_embedder_requires_dimension(settings):
    with pytest.raises(ValueError):
        VectorStore(settings, embedder=FakeEmbedder())


def test_create_collection_is_idempotent_and_recreate_wipes(store, settings):
    index_corpus(store, settings)
    store.create_collection()  # no-op, data survives
    assert store.count_chunks() == len(chunk_corpus(settings))
    store.create_collection(recreate=True)
    assert store.count_chunks() == 0


def test_upsert_is_idempotent(store, settings):
    chunks = chunk_corpus(settings)
    assert store.upsert_documents(chunks) == len(chunks)
    store.upsert_documents(chunks)
    assert store.count_chunks() == len(chunks)


def test_dense_search_finds_relevant_chunk_and_round_trips_payload(store, settings):
    chunks = index_corpus(store, settings)
    hits = store.dense_search("travel rule originator beneficiary wire transfers", top_k=3)
    assert hits[0].chunk.doc_id == "fatf"
    original = next(c for c in chunks if c.chunk_id == hits[0].chunk.chunk_id)
    assert hits[0].chunk == original
    assert hits[0].score >= hits[-1].score


def test_dense_search_respects_top_k_and_handles_empty_collection(store, settings):
    assert store.dense_search("anything", top_k=3) == []
    index_corpus(store, settings)
    assert len(store.dense_search("bank", top_k=2)) == 2


def test_delete_document_removes_only_that_document(store, settings):
    index_corpus(store, settings)
    before = store.count_chunks()
    store.delete_document("kyc")
    remaining = {c.doc_id for c in store.get_all_chunks()}
    assert remaining == {"fatf", "sourdough", "vector-db"}
    assert store.count_chunks() < before


def test_get_all_chunks_pages_through_large_collections(store):
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_document(text, "big", chunk_size=1, chunk_overlap=0)  # 300 one-word chunks
    store.upsert_documents(chunks)
    assert len(store.get_all_chunks()) == 300  # > one scroll page (256)


def test_close_is_idempotent(settings):
    s = VectorStore(settings, embedder=FakeEmbedder(), embedding_dim=FakeEmbedder.DIM)
    s.close()
    s.close()
