import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rerank import FunctionReranker, rerank, reranker_from_env
from app.retrieval import DocumentStore, Passage


def passage(doc_id: str, text: str, score: float = 1.0) -> Passage:
    return Passage(doc_id=doc_id, title=doc_id.title(), text=text, score=score)


def contains(word: str) -> FunctionReranker:
    return FunctionReranker(lambda question, text: float(word in text.lower()), name=f"contains-{word}")


def test_rerank_reorders_by_the_reranker_and_keeps_its_scores():
    passages = [passage("a", "about cats", 3.0), passage("b", "about dogs", 2.0)]
    result = rerank(contains("dogs"), "q", passages, top_k=2)
    assert [p.doc_id for p in result] == ["b", "a"]
    assert [p.score for p in result] == [1.0, 0.0]


def test_rerank_cuts_to_top_k():
    passages = [passage(str(i), f"text {i}") for i in range(5)]
    assert len(rerank(contains("text"), "q", passages, top_k=2)) == 2


def test_ties_keep_retrieval_order():
    passages = [passage("a", "x"), passage("b", "y"), passage("c", "z")]
    assert [p.doc_id for p in rerank(contains("nothing"), "q", passages, top_k=3)] == ["a", "b", "c"]


def test_rerank_of_nothing_is_empty():
    assert rerank(contains("x"), "q", [], top_k=3) == []


def test_reranking_is_off_without_a_model(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    assert reranker_from_env() is None


@pytest.fixture
def store() -> DocumentStore:
    s = DocumentStore()
    # BM25 ranks the first document higher for "refund policy": it repeats both terms.
    s.add("faq", "FAQ", "Refund policy. Our refund policy page lists the refund policy.")
    s.add("terms", "Terms", "Refunds are paid within five days under this policy.")
    return s


def test_the_reranker_picks_the_final_sources_in_the_api(store, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    without = TestClient(create_app(store=store, load_samples=False))
    with without as c:
        plain = c.post("/chat", json={"question": "refund policy"}).text
    assert plain.index('"doc_id": "faq"') < plain.index('"doc_id": "terms"')

    with TestClient(create_app(store=store, load_samples=False, reranker=contains("five days"))) as c:
        assert c.get("/health").json()["reranker"] == "contains-five days"
        reranked = c.post("/chat", json={"question": "refund policy"}).text
    assert reranked.index('"doc_id": "terms"') < reranked.index('"doc_id": "faq"')
