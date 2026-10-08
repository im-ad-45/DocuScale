import pytest

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from tests.corpus import DOCS, FATF_QUESTION, chunk_corpus


def test_rrf_rewards_agreement_and_orders_by_score():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    order = [i for i, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)]
    assert order == ["b", "a", "d", "c"]


def test_rrf_edge_cases():
    assert reciprocal_rank_fusion([]) == []
    assert [i for i, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Wire-Transfers, FATF #16!") == ["wire", "transfers", "fatf", "16"]


def test_bm25_ranks_exact_term_match_first(settings):
    index = BM25Index(chunk_corpus(settings))
    hits = index.search(FATF_QUESTION, top_k=5)
    assert hits[0].chunk.doc_id == "fatf"
    assert all(h.score > 0 for h in hits)


def test_bm25_returns_only_matching_chunks_and_respects_top_k(settings):
    index = BM25Index(chunk_corpus(settings))
    assert index.search("zzzxqv", top_k=5) == []
    assert len(index.search("the", top_k=1)) <= 1


def test_bm25_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", top_k=5) == []
