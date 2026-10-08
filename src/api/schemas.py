"""Request/response schemas for the DocuScale HTTP API."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from src.generation.synthesizer import Answer
from src.retrieval.hybrid import RetrievalResult

# doc_id becomes part of a citation label "doc_id#n", so it must not contain
# the characters the citation parser treats as syntax: # , ; [ ]
DocId = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\w.\- ]+$", min_length=1, max_length=200)
]
DocText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000_000)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class IngestDocument(BaseModel):
    """One document to chunk, embed and index. Re-using a doc_id replaces it."""

    doc_id: DocId
    text: DocText
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1, max_length=100)

    @field_validator("documents")
    @classmethod
    def _unique_doc_ids(cls, docs: list[IngestDocument]) -> list[IngestDocument]:
        ids = [d.doc_id for d in docs]
        if len(ids) != len(set(ids)):
            raise ValueError("doc_id values must be unique within one request")
        return docs


class IngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    collection_size: int


class QueryRequest(BaseModel):
    question: Question
    top_k: int | None = Field(None, ge=1, le=50, description="Defaults to settings.final_top_k")
    use_hyde: bool | None = Field(None, description="Defaults to settings.use_hyde")
    rerank: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: Answer
    retrieval: RetrievalResult


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    collection: str
    collection_size: int | None
    embedding_model: str
    reranker_model: str
    llm_model: str
    hyde_enabled: bool
