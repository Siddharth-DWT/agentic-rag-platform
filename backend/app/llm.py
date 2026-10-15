"""The answering model.

With `OPENAI_API_KEY` set, answers come from any OpenAI-compatible endpoint
(OpenAI, Azure, a local Ollama or vLLM server via `OPENAI_BASE_URL`). Without
it, an extractive fallback answers by quoting the best passages, so the app
and its tests run with no keys at all.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from .retrieval import Passage

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "Cite sources inline as [1], [2]. If the sources do not contain the answer, "
    "say so plainly instead of guessing."
)


def format_sources(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] {p.title}\n{p.text}" for i, p in enumerate(passages, start=1))


class AnswerModel(Protocol):
    name: str

    def stream(self, question: str, passages: list[Passage]) -> AsyncIterator[str]: ...


class ExtractiveModel:
    """Answers by quoting the top passages. No network, deterministic."""

    name = "extractive"

    async def stream(self, question: str, passages: list[Passage]) -> AsyncIterator[str]:
        yield "From the documents:\n\n"
        for i, passage in enumerate(passages[:2], start=1):
            snippet = " ".join(passage.text.split())
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
            yield f"- {snippet} [{i}]\n"


class ChatModel:
    """Streams an answer from an OpenAI-compatible chat model."""

    def __init__(self, model: str, base_url: str | None = None, llm: BaseChatModel | None = None) -> None:
        self.name = model
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=model, base_url=base_url, temperature=0, streaming=True)
        self._llm = llm

    async def stream(self, question: str, passages: list[Passage]) -> AsyncIterator[str]:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Sources:\n\n{format_sources(passages)}\n\nQuestion: {question}"),
        ]
        async for chunk in self._llm.astream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content


def model_from_env() -> AnswerModel:
    if os.environ.get("OPENAI_API_KEY"):
        return ChatModel(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
    return ExtractiveModel()
