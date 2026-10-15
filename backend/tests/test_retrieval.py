from pathlib import Path

import pytest

from app.retrieval import DocumentStore, load_directory

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def store() -> DocumentStore:
    s = DocumentStore()
    load_directory(s, DATA)
    return s


def test_samples_load_with_titles(store):
    titles = {d["title"] for d in store.list()}
    assert titles == {"Retrieval-augmented generation", "Model Context Protocol", "Agents with LangGraph"}
    assert all(d["chunks"] >= 1 for d in store.list())


@pytest.mark.parametrize(
    ("question", "doc_id"),
    [
        ("How does BM25 score a chunk?", "rag"),
        ("Which transport does a remote MCP server use?", "mcp"),
        ("How do I pause a graph for human approval?", "langgraph"),
    ],
)
def test_the_best_passage_comes_from_the_right_document(store, question, doc_id):
    assert store.search(question)[0].doc_id == doc_id


def test_top_k_is_respected(store):
    assert len(store.search("retrieval model server graph", top_k=2)) == 2


def test_unrelated_question_finds_nothing(store):
    assert store.search("xylophone zeppelin") == []


def test_empty_store_finds_nothing():
    assert DocumentStore().search("anything") == []


def test_replacing_a_document_drops_its_old_chunks():
    s = DocumentStore()
    s.add("note", "Note", "The launch code is alpha.")
    s.add("note", "Note", "The launch code is bravo.")
    assert [p.text for p in s.search("launch code")] == ["The launch code is bravo."]


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        DocumentStore().add("x", "X", "   ")
