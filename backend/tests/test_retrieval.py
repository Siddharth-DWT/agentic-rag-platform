from pathlib import Path

import pytest

from app.retrieval import DocumentStore, FunctionEmbedder, load_directory, reciprocal_rank_fusion

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


# --- hybrid retrieval -----------------------------------------------------

# A tiny embedder where synonyms share a dimension, so "passphrase" and
# "password" land close together while sharing no keyword. Real embedding
# models learn this; here it is spelled out so the test needs no network.
CONCEPTS = [
    {"password", "passphrase", "credential"},
    {"wifi", "wi-fi", "wireless", "network"},
    {"parking", "car", "garage"},
    {"lunch", "food", "canteen"},
]


def concept_vector(text: str) -> list[float]:
    words = {w.strip(".,?!").lower() for w in text.split()}
    return [float(len(words & c)) for c in CONCEPTS]


def hybrid_store() -> DocumentStore:
    s = DocumentStore(embedder=FunctionEmbedder(concept_vector))
    s.add("wifi", "Office Wi-Fi", "The guest wireless credential is sunflower42.")
    s.add("parking", "Parking", "Visitors park in the garage on level two.")
    s.add("lunch", "Lunch", "The canteen serves food from noon.")
    return s


def test_rrf_rewards_agreement_between_rankings():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]])
    assert scores["a"] == scores["b"] > scores["c"]
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_a_store_without_an_embedder_is_keyword_only():
    assert DocumentStore().mode == "bm25"
    assert hybrid_store().mode == "hybrid"


def test_vector_search_finds_a_paraphrase_keyword_search_misses():
    keyword_only = DocumentStore()
    keyword_only.add("wifi", "Office Wi-Fi", "The guest wireless credential is sunflower42.")
    assert keyword_only.search("wifi passphrase") == []

    [hit, *_] = hybrid_store().search("wifi passphrase")
    assert hit.doc_id == "wifi" and hit.matched_by == "vector"


def test_a_match_on_both_is_labelled_both_and_ranked_first():
    [hit, *_] = hybrid_store().search("garage parking")
    assert hit.doc_id == "parking" and hit.matched_by == "both"


def test_weak_vector_matches_are_dropped():
    assert hybrid_store().search("quantum entanglement") == []


def test_replacing_a_document_drops_its_old_vectors():
    s = hybrid_store()
    s.add("wifi", "Office Wi-Fi", "Ask reception for the guest credential.")
    hits = s.search("passphrase")
    assert [h.text for h in hits] == ["Ask reception for the guest credential."]
