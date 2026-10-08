import json

import pytest

from src.ingestion.chunker import chunk_document
from tests.corpus import DOCS, FATF_QUESTION

INGEST_ALL = {"documents": [{"doc_id": k, "text": v} for k, v in DOCS.items()]}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    frames = [f for f in body.strip().split("\n\n") if f]
    return [
        (f.splitlines()[0].removeprefix("event: "), json.loads(f.splitlines()[1].removeprefix("data: ")))
        for f in frames
    ]


@pytest.fixture
def loaded(client):
    assert client.post("/ingest", json=INGEST_ALL).status_code == 200
    return client


# ---------- health ----------
def test_health_reports_status_collection_and_llm_config(client, settings):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["collection_size"] == 0
    assert body["collection"] == "test" and body["llm_model"] == settings.llm_model
    assert body["hyde_enabled"] is False
    assert "key" not in json.dumps(body).lower()


def test_health_degrades_with_503_when_the_store_fails(client, services, monkeypatch):
    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(services.store, "count_chunks", boom)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"
    assert r.json()["collection_size"] is None


# ---------- ingest ----------
def test_ingest_chunks_indexes_and_reports_sizes(client, settings):
    expected = sum(len(chunk_document(t, k, settings.chunk_size, settings.chunk_overlap)) for k, t in DOCS.items())
    body = client.post("/ingest", json=INGEST_ALL).json()
    assert body == {"documents": len(DOCS), "chunks_indexed": expected, "collection_size": expected}
    assert client.get("/health").json()["collection_size"] == expected


def test_ingest_refreshes_bm25_so_new_docs_are_lexically_searchable(loaded):
    hits = loaded.post("/query", json={"question": FATF_QUESTION, "rerank": False}).json()["retrieval"]["hits"]
    assert any(h["bm25_rank"] is not None for h in hits)


def test_reingesting_a_document_replaces_it_without_stale_chunks(client, settings):
    long_text = " ".join(f"alpha{i}" for i in range(200))
    n_long = len(chunk_document(long_text, "doc", settings.chunk_size, settings.chunk_overlap))
    assert n_long > 1
    assert client.post("/ingest", json={"documents": [{"doc_id": "doc", "text": long_text}]}).json()["chunks_indexed"] == n_long
    r = client.post("/ingest", json={"documents": [{"doc_id": "doc", "text": "short replacement text"}]}).json()
    assert r["chunks_indexed"] == 1 and r["collection_size"] == 1
    hits = client.post("/query", json={"question": "alpha5", "top_k": 10, "rerank": False}).json()["retrieval"]["hits"]
    assert [h["chunk"]["chunk_index"] for h in hits] == [0]


@pytest.mark.parametrize(
    "payload",
    [
        {"documents": []},
        {"documents": [{"doc_id": "bad#id", "text": "x"}]},
        {"documents": [{"doc_id": "a,b", "text": "x"}]},
        {"documents": [{"doc_id": "   ", "text": "x"}]},
        {"documents": [{"doc_id": "ok", "text": "   "}]},
        {"documents": [{"doc_id": "dup", "text": "x"}, {"doc_id": "dup", "text": "y"}]},
        {},
    ],
)
def test_ingest_rejects_invalid_payloads(client, payload):
    assert client.post("/ingest", json=payload).status_code == 422


# ---------- query ----------
def test_query_returns_grounded_answer_with_retrieval_evidence(loaded):
    r = loaded.post("/query", json={"question": FATF_QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == FATF_QUESTION
    assert body["answer"]["grounded"] is True and body["answer"]["citations"][0].startswith("fatf#")
    assert body["answer"]["sources"][0]["doc_id"] == "fatf"
    assert body["retrieval"]["hits"][0]["chunk"]["doc_id"] == "fatf"


def test_query_on_empty_store_refuses_without_calling_the_llm(client, llm, settings):
    body = client.post("/query", json={"question": "anything"}).json()
    assert body["answer"]["refused"] is True and body["answer"]["answer"] == settings.refusal_message
    assert llm.calls == []


def test_query_refusal_from_the_model_is_reported(loaded, llm, settings):
    llm.answer = settings.refusal_message
    a = loaded.post("/query", json={"question": "who won the world cup?"}).json()["answer"]
    assert a["refused"] is True and a["grounded"] is False


def test_query_flags_hallucinated_citations(loaded, llm):
    llm.answer = "Invented claim [nope#7]."
    a = loaded.post("/query", json={"question": FATF_QUESTION}).json()["answer"]
    assert a["grounded"] is False and a["unknown_citations"] == ["nope#7"]


def test_query_hyde_override_is_honoured(loaded, llm):
    on = loaded.post("/query", json={"question": FATF_QUESTION, "use_hyde": True}).json()
    off = loaded.post("/query", json={"question": FATF_QUESTION, "use_hyde": False}).json()
    assert on["retrieval"]["hyde_passage"] == llm.hyde and off["retrieval"]["hyde_passage"] is None


def test_query_llm_failure_is_a_502_without_leaking_details(loaded, llm):
    llm.fail_answer = True
    r = loaded.post("/query", json={"question": FATF_QUESTION})
    assert r.status_code == 502 and r.json() == {"detail": "LLM generation failed"}


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "   "}, {"question": "q", "top_k": 0}, {"question": "q", "top_k": 51}]
)
def test_query_rejects_invalid_payloads(client, payload):
    assert client.post("/query", json=payload).status_code == 422
    assert client.post("/query/stream", json=payload).status_code == 422


# ---------- streaming ----------
def test_stream_emits_tokens_then_a_final_audit_event(loaded):
    r = loaded.post("/query/stream", json={"question": FATF_QUESTION})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"

    events = parse_sse(r.text)
    names = [name for name, _ in events]
    assert names[-1] == "done" and set(names[:-1]) == {"token"} and len(names) > 2

    done = events[-1][1]
    assert set(done) == {"answer", "refused", "grounded", "citations", "unknown_citations", "sources"}
    assert done["grounded"] is True and done["refused"] is False
    assert done["citations"][0].startswith("fatf#")
    assert "".join(data["text"] for name, data in events if name == "token").strip() == done["answer"]


def test_stream_audit_matches_the_blocking_endpoint(loaded):
    streamed = parse_sse(loaded.post("/query/stream", json={"question": FATF_QUESTION}).text)[-1][1]
    blocking = loaded.post("/query", json={"question": FATF_QUESTION}).json()["answer"]
    assert streamed == blocking


def test_stream_flags_hallucinated_citations_in_the_final_event(loaded, llm):
    llm.answer = "Invented claim [nope#7]."
    done = parse_sse(loaded.post("/query/stream", json={"question": FATF_QUESTION}).text)[-1][1]
    assert done["grounded"] is False and done["unknown_citations"] == ["nope#7"]


def test_stream_on_empty_store_streams_the_refusal_without_calling_the_llm(client, llm, settings):
    events = parse_sse(client.post("/query/stream", json={"question": "anything"}).text)
    assert events[0] == ("token", {"text": settings.refusal_message})
    assert events[-1][0] == "done" and events[-1][1]["refused"] is True
    assert llm.calls == []


def test_stream_failure_before_first_token_is_a_502(loaded, llm):
    llm.fail_answer = True
    r = loaded.post("/query/stream", json={"question": FATF_QUESTION})
    assert r.status_code == 502 and r.json() == {"detail": "LLM generation failed"}


def test_stream_failure_mid_response_is_reported_in_band(loaded, llm):
    llm.fail_stream_after = 2
    events = parse_sse(loaded.post("/query/stream", json={"question": FATF_QUESTION}).text)
    assert [n for n, _ in events] == ["token", "token", "error"]
    assert events[-1][1] == {"detail": "LLM generation failed"}


def test_openapi_lists_every_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/ingest", "/query", "/query/stream", "/health"} <= set(paths)
