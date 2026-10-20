"""Reranking: reorder retrieved passages by reading the question and each passage together.

Retrieval scores every chunk on its own: by shared terms (BM25) or by one
embedding per chunk. A cross-encoder scores the question and a passage as a
pair, which is slower but more accurate, so it runs only on a short candidate
list and picks the final top_k from it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from .retrieval import Passage


class Reranker(Protocol):
    name: str

    def score(self, question: str, texts: list[str]) -> list[float]: ...


def rerank(reranker: Reranker, question: str, passages: list[Passage], top_k: int) -> list[Passage]:
    """Return the `top_k` passages the reranker scores highest, carrying its scores."""
    if not passages:
        return []
    scores = reranker.score(question, [p.text for p in passages])
    # sorted() is stable, so passages the reranker ties keep their retrieval order
    ranked = sorted(zip(passages, scores, strict=True), key=lambda pair: -pair[1])
    return [replace(p, score=float(s)) for p, s in ranked[:top_k]]


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model
        self._model = CrossEncoder(model)

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(question, t) for t in texts])]


class FunctionReranker:
    """Wraps a plain scoring function, for tests and custom rerankers."""

    def __init__(self, fn: Callable[[str, str], float], name: str = "function") -> None:
        self.name = name
        self._fn = fn

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [self._fn(question, t) for t in texts]


def reranker_from_env() -> Reranker | None:
    """Reranking is off unless RERANKER_MODEL names a cross-encoder."""
    if model := os.environ.get("RERANKER_MODEL"):
        return CrossEncoderReranker(model)
    return None
