"""Sliding-window text chunker with overlap and metadata tracking."""
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

_TOKEN = re.compile(r"\S+")


class Chunk(BaseModel):
    """A slice of a source document, plus everything needed to trace it back."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    start_token: int
    end_token: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of ``chunk_size`` tokens.

    Tokens are whitespace-delimited words (a cheap approximation of model
    tokens). Each window starts ``chunk_size - chunk_overlap`` tokens after the
    previous one, so neighbouring chunks share ``chunk_overlap`` tokens.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = _TOKEN.findall(text)
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        chunks.append(
            Chunk(
                # Deterministic ID: re-ingesting the same doc overwrites, never duplicates.
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}")),
                doc_id=doc_id,
                text=" ".join(tokens[start:end]),
                chunk_index=index,
                start_token=start,
                end_token=end,
                metadata=metadata or {},
            )
        )
        if end == len(tokens):  # window reached the end; don't emit a redundant tail
            break
    return chunks
