import pytest

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hyde import HydeExpander
from tests.corpus import FATF_QUESTION, index_corpus
from tests.fakes import FakeLLM, FakeReranker


def test_hybrid_surfaces_the_exact_term_document(retriever):
    result = retriever.retrieve(FATF_QUESTION)
    assert result.hits[0].chunk.doc_id == "fatf"
    assert result.hyde_passage is None


def test_hits_carry_evidence_from_every_stage(retriever):
    for hit in retriever.retrieve(FATF_QUESTION, top_k=5).hits:
        assert hit.rrf_score > 0
        assert hit.dense_rank is not None or hit.bm25_rank is not None
        assert hit.rerank_score is not None


def test_rerank_false_skips_the_cross_encoder(retriever, reranker):
    hits = retriever.retrieve(FATF_QUESTION, rerank=False).hits
    assert reranker.calls == []
    assert all(h.rerank_score is None for h in hits)
    assert [h.rrf_score for h in hits] == sorted((h.rrf_score for h in hits), reverse=True)


def test_reranker_decides_the_final_order(indexed_store, settings):
    class Reversing:
        def score(self, query, chunks):
            return [float(i) for i in range(len(chunks))]  # last candidate wins

    r = HybridRetriever(indexed_store, settings, reranker=Reversing())
    by_rrf = r.retrieve(FATF_QUESTION, top_k=10, rerank=False).hits
    by_rerank = r.retrieve(FATF_QUESTION, top_k=3).hits
    assert [h.chunk.chunk_id for h in by_rerank] == [h.chunk.chunk_id for h in reversed(by_rrf)][:3]


def test_top_k_defaults_to_settings(retriever, settings):
    assert len(retriever.retrieve("bank").hits) == settings.final_top_k
    assert len(retriever.retrieve("bank", top_k=1).hits) == 1


def test_hyde_feeds_dense_only_and_reranker_sees_raw_query(retriever, indexed_store, llm, reranker, monkeypatch):
    dense_queries = []
    original = indexed_store.dense_search
    monkeypatch.setattr(
        indexed_store, "dense_search", lambda q, top_k: (dense_queries.append(q), original(q, top_k=top_k))[1]
    )
    result = retriever.retrieve(FATF_QUESTION, top_k=10, use_hyde=True)

    assert result.hyde_passage == llm.hyde
    assert dense_queries == [f"{FATF_QUESTION}\n\n{llm.hyde}"]  # raw query prepended
    assert reranker.calls[0][0] == FATF_QUESTION  # reranker judges the user's words
    # the passage mentions "sourdough"; if BM25 had seen it, sourdough chunks would have a bm25_rank
    sourdough = [h for h in result.hits if h.chunk.doc_id == "sourdough"]
    assert sourdough and all(h.bm25_rank is None for h in sourdough)


def test_hyde_follows_settings_flag_and_per_call_override(indexed_store, settings, llm, reranker):
    on = settings.model_copy(update={"use_hyde": True})
    r = HybridRetriever(indexed_store, on, reranker=reranker, hyde=HydeExpander(llm, on))
    assert r.retrieve(FATF_QUESTION).hyde_passage is not None
    assert r.retrieve(FATF_QUESTION, use_hyde=False).hyde_passage is None


def test_hyde_failure_falls_back_to_the_raw_query(indexed_store, settings, reranker):
    broken = FakeLLM(fail_hyde=True)
    r = HybridRetriever(indexed_store, settings, reranker=reranker, hyde=HydeExpander(broken, settings))
    result = r.retrieve(FATF_QUESTION, use_hyde=True)
    assert result.hyde_passage is None
    assert result.hits[0].chunk.doc_id == "fatf"


def test_hyde_without_an_expander_is_a_configuration_error(indexed_store, settings, reranker):
    with pytest.raises(ValueError):
        HybridRetriever(indexed_store, settings.model_copy(update={"use_hyde": True}), reranker=reranker)
    plain = HybridRetriever(indexed_store, settings, reranker=reranker)
    with pytest.raises(ValueError):
        plain.retrieve("q", use_hyde=True)


def test_empty_store_returns_no_hits(store, settings, reranker):
    assert HybridRetriever(store, settings, reranker=reranker).retrieve("anything").hits == []


def test_refresh_index_makes_new_documents_visible_to_bm25(store, settings, reranker):
    r = HybridRetriever(store, settings, reranker=reranker)  # BM25 built while empty
    index_corpus(store, settings)
    assert all(h.bm25_rank is None for h in r.retrieve(FATF_QUESTION, rerank=False).hits)
    r.refresh_index()
    assert any(h.bm25_rank is not None for h in r.retrieve(FATF_QUESTION, rerank=False).hits)


class TestHydeExpander:
    def test_returns_stripped_passage(self, settings):
        assert HydeExpander(FakeLLM(hyde="  passage \n"), settings).expand("q") == "passage"

    def test_returns_none_on_error_or_blank_output(self, settings):
        assert HydeExpander(FakeLLM(fail_hyde=True), settings).expand("q") is None
        assert HydeExpander(FakeLLM(hyde="   "), settings).expand("q") is None

    def test_sends_the_raw_query_with_the_hyde_prompt(self, settings):
        llm = FakeLLM()
        HydeExpander(llm, settings).expand("my question")
        kind, system, user = llm.calls[0]
        assert (kind, system, user) == ("hyde", settings.hyde_prompt, "my question")
