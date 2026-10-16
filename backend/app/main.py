"""HTTP API: add documents, list them, and ask questions with a streamed answer."""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import build_agent
from .llm import embedder_from_env, model_from_env
from .retrieval import DocumentStore, load_directory

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "data"


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


def create_app(store: DocumentStore | None = None, load_samples: bool = True) -> FastAPI:
    store = store or DocumentStore(embedder=embedder_from_env())
    model = model_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_samples and SAMPLE_DATA.is_dir():
            load_directory(store, SAMPLE_DATA)
        app.state.agent = build_agent(store, model)
        yield

    app = FastAPI(title="Agentic RAG Platform", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model.name, "retrieval": store.mode}

    @app.get("/documents")
    def list_documents() -> list[dict[str, str | int]]:
        return store.list()

    @app.post("/documents", status_code=201)
    def add_document(doc: DocumentIn) -> dict[str, str | int]:
        doc_id = _slug(doc.title)
        try:
            chunks = store.add(doc_id, doc.title, doc.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"id": doc_id, "title": doc.title, "chunks": chunks}

    @app.post("/chat")
    async def chat(body: Question) -> StreamingResponse:
        return StreamingResponse(
            _events(app.state.agent, body.question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


async def _events(agent, question: str) -> AsyncIterator[str]:
    """Server-sent events: one `sources` event, then `token` events, then `done`."""
    try:
        async for event in agent.astream({"question": question}, stream_mode="custom"):
            yield _sse(event["type"], event)
    except Exception:  # surface a clean error to the client, keep detail in logs
        import logging

        logging.getLogger(__name__).exception("chat failed")
        yield _sse("error", {"message": "The answer could not be generated."})
        return
    yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "document"


app = create_app()
