"""Typed, immutable configuration for DocuScale."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    """Validated settings. Frozen so nothing can mutate config at runtime."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    storage_path: Path = Path("./qdrant_data")
    collection_name: str = "docuscale"
    chunk_size: int = Field(200, gt=0, description="Tokens per chunk")
    chunk_overlap: int = Field(40, ge=0, description="Tokens shared between neighbours")
    reranker_model: str = "BAAI/bge-reranker-base"
    candidate_k: int = Field(20, gt=0, description="Candidates fetched per retriever")
    rrf_k: int = Field(60, gt=0, description="RRF smoothing constant")
    final_top_k: int = Field(5, gt=0, description="Results returned to the caller")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def load_settings() -> Settings:
    """Build Settings from defaults, overridden by DOCUSCALE_* env vars / .env."""
    load_dotenv()
    overrides = {
        name: value
        for name in Settings.model_fields
        if (value := os.getenv(f"DOCUSCALE_{name.upper()}")) is not None
    }
    return Settings(**overrides)  # pydantic coerces "200" -> int, "./x" -> Path
