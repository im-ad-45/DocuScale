"""FastAPI application factory.

Run:  uvicorn src.api.app:app --host 127.0.0.1 --port 8000
(single worker: local Qdrant and the in-memory BM25 index are per-process)
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.api.services import Services, build_services
from src.config import load_settings


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Pass ``services`` (e.g. with fakes) to skip loading real models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = services if services is not None else build_services(load_settings())
        active.store.create_collection()
        app.state.services = active
        try:
            yield
        finally:
            if services is None:  # only close what this lifespan created
                active.store.close()

    app = FastAPI(title="DocuScale", version="0.4.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
