import json

import pytest
from fastapi.testclient import TestClient

from app.agent import NO_ANSWER
from app.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(create_app()) as c:
        yield c


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health_reports_the_model(client):
    assert client.get("/health").json() == {"status": "ok", "model": "extractive"}


def test_sample_documents_are_listed(client):
    assert len(client.get("/documents").json()) == 3


def test_chat_streams_sources_then_tokens_then_done(client):
    r = client.post("/chat", json={"question": "What does a reranker do?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    evs = events(r)
    names = [name for name, _ in evs]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert evs[0][1]["sources"][0]["doc_id"] == "rag"
    answer = "".join(data["text"] for name, data in evs if name == "token")
    assert "[1]" in answer


def test_chat_says_so_when_nothing_matches(client):
    evs = events(client.post("/chat", json={"question": "xylophone zeppelin"}))
    assert evs[0] == ("sources", {"type": "sources", "sources": []})
    assert "".join(d["text"] for n, d in evs if n == "token") == NO_ANSWER


def test_an_added_document_can_be_asked_about(client):
    r = client.post(
        "/documents", json={"title": "Office Wi-Fi", "text": "The guest network password is sunflower42."}
    )
    assert r.status_code == 201 and r.json()["id"] == "office-wi-fi"
    evs = events(client.post("/chat", json={"question": "guest network password"}))
    assert evs[0][1]["sources"][0]["title"] == "Office Wi-Fi"


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 2001}])
def test_bad_questions_are_rejected(client, body):
    assert client.post("/chat", json=body).status_code == 422
