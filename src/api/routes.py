"""HTTP routes: ingestion, querying (blocking + SSE streaming) and health."""
import itertools
import json
import logging
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.api.services import Services, get_services
from src.ingestion.chunker import chunk_document
from src.retrieval.hybrid import RetrievalResult

logger = logging.getLogger(__name__)
router = APIRouter()


def sse(event: str, data: BaseModel | dict[str, Any]) -> str:
    """Format one Server-Sent Events frame. ``data`` is JSON on a single line."""
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _retrieve(req: QueryRequest, svc: Services) -> RetrievalResult:
    return svc.retriever.retrieve(
        req.question, top_k=req.top_k, rerank=req.rerank, use_hyde=req.use_hyde
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, svc: Services = Depends(get_services)) -> IngestResponse:
    """Chunk, embed and index documents, then refresh the BM25 index."""
    s = svc.settings
    total = 0
    with svc.ingest_lock:  # serialize writers; delete+upsert per doc must not interleave
        for doc in req.documents:
            chunks = chunk_document(doc.text, doc.doc_id, s.chunk_size, s.chunk_overlap, doc.metadata)
            svc.store.delete_document(doc.doc_id)  # re-ingesting replaces, never leaves stale chunks
            total += svc.store.upsert_documents(chunks)
        svc.retriever.refresh_index()
    return IngestResponse(
        documents=len(req.documents), chunks_indexed=total, collection_size=svc.store.count_chunks()
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, svc: Services = Depends(get_services)) -> QueryResponse:
    """Retrieve, then return one complete, citation-audited answer."""
    result = _retrieve(req, svc)
    try:
        answer = svc.synthesizer.answer(req.question, result.hits)
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail="LLM generation failed") from exc
    return QueryResponse(question=req.question, answer=answer, retrieval=result)


@router.post("/query/stream", response_class=StreamingResponse)
def query_stream(req: QueryRequest, svc: Services = Depends(get_services)) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Events: ``token`` {"text"} repeated, then one ``done`` (the full audited
    ``Answer``), or ``error`` {"detail"} if generation breaks mid-stream.
    """
    result = _retrieve(req, svc)
    tokens = iter(svc.synthesizer.stream(req.question, result.hits))
    try:
        first = next(tokens, None)  # wait for token 1 so auth/connection errors become a real 502
    except Exception as exc:
        logger.exception("LLM stream failed before the first token")
        raise HTTPException(status_code=502, detail="LLM generation failed") from exc

    def events() -> Iterator[str]:
        parts: list[str] = []
        try:
            for token in itertools.chain([] if first is None else [first], tokens):
                parts.append(token)
                yield sse("token", {"text": token})
            yield sse("done", svc.synthesizer.audit("".join(parts), result.hits))
        except Exception:
            logger.exception("LLM stream failed mid-response")
            yield sse("error", {"detail": "LLM generation failed"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthResponse)
def health(response: Response, svc: Services = Depends(get_services)) -> HealthResponse:
    """Report pipeline status, collection size and the active model configuration."""
    s = svc.settings
    status: Literal["ok", "degraded"] = "ok"
    size: int | None = None
    try:
        size = svc.store.count_chunks()
    except Exception:
        logger.exception("health check could not reach the vector store")
        status = "degraded"
        response.status_code = 503
    return HealthResponse(
        status=status,
        collection=s.collection_name,
        collection_size=size,
        embedding_model=s.embedding_model,
        reranker_model=s.reranker_model,
        llm_model=s.llm_model,
        hyde_enabled=s.use_hyde,
    )
