"""HTTP API: add documents, list them, and ask questions with a streamed answer."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from .agent import build_agent
from .llm import embedder_from_env, model_from_env
from .rerank import Reranker, reranker_from_env
from .retrieval import DocumentStore, load_directory

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "data"


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    # Continue an earlier conversation; omit to start a new one.
    thread_id: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


def create_app(
    store: DocumentStore | None = None,
    load_samples: bool = True,
    reranker: Reranker | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> FastAPI:
    store = store or DocumentStore(embedder=embedder_from_env())
    model = model_from_env()
    reranker = reranker or reranker_from_env()
    # Conversations live in memory; pass a persistent saver (SQLite, Postgres)
    # to keep them across restarts.
    checkpointer = checkpointer or InMemorySaver()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_samples and SAMPLE_DATA.is_dir():
            load_directory(store, SAMPLE_DATA)
        app.state.agent = build_agent(store, model, reranker=reranker, checkpointer=checkpointer)
        yield

    app = FastAPI(title="Agentic RAG Platform", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Thread-Id"],
    )

    @app.get("/health")
    def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "model": model.name,
            "retrieval": store.mode,
            "reranker": reranker.name if reranker else None,
        }

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
        thread_id = body.thread_id or uuid.uuid4().hex
        return StreamingResponse(
            _events(app.state.agent, body.question, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Thread-Id": thread_id},
        )

    @app.delete("/threads/{thread_id}", status_code=204)
    def forget_thread(thread_id: str) -> Response:
        checkpointer.delete_thread(thread_id)
        return Response(status_code=204)

    return app


async def _events(agent, question: str, thread_id: str) -> AsyncIterator[str]:
    """Server-sent events: one `sources` event, then `token` events, then `done`."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async for event in agent.astream({"question": question}, config, stream_mode="custom"):
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
