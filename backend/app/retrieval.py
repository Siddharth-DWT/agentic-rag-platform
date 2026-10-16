"""Document store and retrieval.

Documents are split into overlapping chunks with LlamaIndex's SentenceSplitter.
Every store searches with BM25, which needs no model and is strong on exact
names and rare terms. Given an embedder, the store also searches by vector
similarity, which finds passages that say the same thing in other words, and
merges the two rankings with reciprocal rank fusion (hybrid retrieval).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever

Vector = list[float]


class Embedder:
    """Turns text into vectors. Documents and queries may be embedded differently."""

    name: str

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> Vector:
        raise NotImplementedError


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with enough to cite where it came from."""

    doc_id: str
    title: str
    text: str
    score: float
    matched_by: Literal["keyword", "vector", "both"] = "keyword"


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Merge ranked lists of ids. Each list adds 1 / (k + rank) for every id in it.

    RRF uses only ranks, so it needs no tuning between BM25 scores and cosine
    similarities, which are on unrelated scales. k = 60 is the value from the
    original paper and damps the advantage of the very top ranks.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class DocumentStore:
    """An in-memory corpus. The BM25 index is rebuilt lazily after each change."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        embedder: Embedder | None = None,
        min_similarity: float = 0.3,
    ) -> None:
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = embedder
        # Every chunk has some similarity to every question; below this a vector
        # match is noise, and keeping it would stop the agent saying "not found".
        self._min_similarity = min_similarity
        self._documents: dict[str, Document] = {}
        self._nodes: list[BaseNode] = []
        self._vectors: dict[str, Vector] = {}
        self._retriever: BM25Retriever | None = None
        self._lock = threading.Lock()

    @property
    def mode(self) -> Literal["bm25", "hybrid"]:
        return "hybrid" if self._embedder else "bm25"

    def add(self, doc_id: str, title: str, text: str) -> int:
        """Add or replace a document. Returns the number of chunks it produced."""
        if not text.strip():
            raise ValueError("document text is empty")
        document = Document(text=text, id_=doc_id, metadata={"title": title})
        chunks = self._splitter.get_nodes_from_documents([document])
        vectors = self._embedder.embed_documents([c.get_content() for c in chunks]) if self._embedder else []
        with self._lock:
            stale = {n.node_id for n in self._nodes if n.ref_doc_id == doc_id}
            self._documents[doc_id] = document
            self._nodes = [n for n in self._nodes if n.node_id not in stale] + chunks
            for node_id in stale:
                self._vectors.pop(node_id, None)
            self._vectors.update({c.node_id: v for c, v in zip(chunks, vectors, strict=False)})
            self._retriever = None
        return len(chunks)

    def list(self) -> list[dict[str, str | int]]:
        with self._lock:
            counts: dict[str, int] = {}
            for node in self._nodes:
                counts[node.ref_doc_id] = counts.get(node.ref_doc_id, 0) + 1
            return [
                {"id": doc_id, "title": doc.metadata["title"], "chunks": counts.get(doc_id, 0)}
                for doc_id, doc in self._documents.items()
            ]

    def search(self, query: str, top_k: int = 4) -> list[Passage]:
        """Return up to `top_k` relevant chunks, best first."""
        with self._lock:
            if not self._nodes:
                return []
            if self._retriever is None:
                # BM25 refuses a top_k larger than the corpus, so ask for all of it.
                self._retriever = BM25Retriever.from_defaults(
                    nodes=self._nodes, similarity_top_k=len(self._nodes)
                )
            retriever = self._retriever
            nodes = {n.node_id: n for n in self._nodes}
            vectors = dict(self._vectors)

        keyword = {r.node.node_id: float(r.score or 0.0) for r in retriever.retrieve(query)}
        keyword_ranking = [i for i, s in sorted(keyword.items(), key=lambda x: -x[1]) if s > 0.0]

        if not self._embedder:
            return [self._passage(nodes[i], keyword[i], "keyword") for i in keyword_ranking[:top_k]]

        q = self._embedder.embed_query(query)
        similarity = {i: _cosine(q, v) for i, v in vectors.items()}
        vector_ranking = [
            i for i, s in sorted(similarity.items(), key=lambda x: -x[1]) if s >= self._min_similarity
        ]

        fused = reciprocal_rank_fusion([keyword_ranking, vector_ranking])
        in_keyword, in_vector = set(keyword_ranking), set(vector_ranking)
        best = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [
            self._passage(
                nodes[i],
                fused[i],
                "both" if i in in_keyword and i in in_vector else "keyword" if i in in_keyword else "vector",
            )
            for i in best
        ]

    @staticmethod
    def _passage(node: BaseNode, score: float, matched_by) -> Passage:
        return Passage(
            doc_id=node.ref_doc_id or "",
            title=node.metadata.get("title", ""),
            text=node.get_content(),
            score=score,
            matched_by=matched_by,
        )


class OpenAIEmbedder(Embedder):
    """Embeddings from any OpenAI-compatible endpoint."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.name = model
        self._client = OpenAIEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        return self._client.embed_query(text)


class FunctionEmbedder(Embedder):
    """Wraps a plain function, for tests and for local models."""

    def __init__(self, fn: Callable[[str], Vector], name: str = "function") -> None:
        self.name = name
        self._fn = fn

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._fn(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._fn(text)


def load_directory(store: DocumentStore, directory: Path) -> int:
    """Load every markdown file in `directory`, titled by its first heading."""
    loaded = 0
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first = text.lstrip().splitlines()[0] if text.strip() else path.stem
        title = first.lstrip("# ").strip() or path.stem
        store.add(path.stem, title, text)
        loaded += 1
    return loaded
