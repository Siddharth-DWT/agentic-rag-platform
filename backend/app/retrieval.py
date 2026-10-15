"""Document store and retrieval.

Documents are split into overlapping chunks with LlamaIndex's SentenceSplitter
and searched with BM25. BM25 needs no API key and no embedding model, so the
whole pipeline runs offline; vector search is added alongside it as the
second half of hybrid retrieval (see the roadmap in the README).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with enough to cite where it came from."""

    doc_id: str
    title: str
    text: str
    score: float


class DocumentStore:
    """An in-memory corpus with a BM25 index that is rebuilt on every change."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._documents: dict[str, Document] = {}
        self._nodes: list[BaseNode] = []
        self._retriever: BM25Retriever | None = None
        self._lock = threading.Lock()

    def add(self, doc_id: str, title: str, text: str) -> int:
        """Add or replace a document. Returns the number of chunks it produced."""
        if not text.strip():
            raise ValueError("document text is empty")
        document = Document(text=text, id_=doc_id, metadata={"title": title})
        chunks = self._splitter.get_nodes_from_documents([document])
        with self._lock:
            self._documents[doc_id] = document
            self._nodes = [n for n in self._nodes if n.ref_doc_id != doc_id] + chunks
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
        """Return up to `top_k` chunks that share terms with the query."""
        with self._lock:
            if not self._nodes:
                return []
            if self._retriever is None:
                # BM25 refuses a top_k larger than the corpus, so cap it here
                # and again per query below.
                self._retriever = BM25Retriever.from_defaults(
                    nodes=self._nodes, similarity_top_k=len(self._nodes)
                )
            retriever = self._retriever
        results = retriever.retrieve(query)
        passages = [
            Passage(
                doc_id=r.node.ref_doc_id or "",
                title=r.node.metadata.get("title", ""),
                text=r.node.get_content(),
                score=float(r.score or 0.0),
            )
            for r in results
            if (r.score or 0.0) > 0.0
        ]
        return passages[:top_k]


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
