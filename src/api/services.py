"""Application services: one wired-up pipeline shared by all requests."""
import threading
from dataclasses import dataclass, field

from fastapi import Request

from src.config import Settings
from src.generation.llm import LiteLLMClient
from src.generation.synthesizer import Synthesizer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hyde import HydeExpander
from src.storage.vector_store import VectorStore


@dataclass
class Services:
    """Everything a request handler needs. Tests build this with fakes."""

    settings: Settings
    store: VectorStore
    retriever: HybridRetriever
    synthesizer: Synthesizer
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)


def build_services(settings: Settings) -> Services:
    """Construct the production pipeline (loads the embedding and reranker models)."""
    llm = LiteLLMClient(settings)
    store = VectorStore(settings)
    store.create_collection()
    retriever = HybridRetriever(store, settings, hyde=HydeExpander(llm, settings))
    return Services(settings, store, retriever, Synthesizer(llm, settings))


def get_services(request: Request) -> Services:
    """FastAPI dependency: the Services instance created at startup."""
    return request.app.state.services
