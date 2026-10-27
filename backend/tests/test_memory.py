import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import NO_ANSWER
from app.llm import ChatModel
from app.main import create_app
from app.retrieval import Passage


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(create_app()) as c:
        yield c


def ask(client, question, thread_id=None):
    body = {"question": question} if thread_id is None else {"question": question, "thread_id": thread_id}
    r = client.post("/chat", json=body)
    assert r.status_code == 200
    events = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    answer = "".join(d["text"] for name, d in events if name == "token")
    return r.headers["x-thread-id"], events[0][1]["sources"], answer


def test_each_new_conversation_gets_its_own_thread(client):
    first, _, _ = ask(client, "How does BM25 score a chunk?")
    second, _, _ = ask(client, "How does BM25 score a chunk?")
    assert first and second and first != second


def test_a_follow_up_is_read_with_the_question_before_it(client):
    # "tell me more" matches nothing on its own...
    _, sources, answer = ask(client, "tell me more")
    assert sources == [] and answer == NO_ANSWER

    # ...but in a conversation it is retrieved together with the previous question.
    thread, _, _ = ask(client, "How does BM25 score a chunk?")
    same_thread, sources, answer = ask(client, "tell me more", thread_id=thread)
    assert same_thread == thread
    assert sources and sources[0]["doc_id"] == "rag"
    assert answer != NO_ANSWER


def test_the_conversation_history_is_kept_per_thread(client):
    thread, _, _ = ask(client, "How does BM25 score a chunk?")
    ask(client, "Which transport does a remote MCP server use?", thread_id=thread)

    state = client.app.state.agent.get_state({"configurable": {"thread_id": thread}})
    assert [turn["question"] for turn in state.values["history"]] == [
        "How does BM25 score a chunk?",
        "Which transport does a remote MCP server use?",
    ]


def test_a_forgotten_thread_starts_over(client):
    thread, _, _ = ask(client, "How does BM25 score a chunk?")
    assert client.delete(f"/threads/{thread}").status_code == 204

    _, sources, answer = ask(client, "tell me more", thread_id=thread)
    assert sources == [] and answer == NO_ANSWER


@pytest.mark.parametrize("thread_id", ["", "has space", "x" * 101, "../etc"])
def test_invalid_thread_ids_are_rejected(client, thread_id):
    assert client.post("/chat", json={"question": "hi", "thread_id": thread_id}).status_code == 422


def test_the_chat_model_sees_earlier_turns():
    class RecordingLLM:
        def __init__(self):
            self.messages = None

        async def astream(self, messages):
            self.messages = messages
            yield type("Chunk", (), {"content": "ok"})()

    llm = RecordingLLM()
    model = ChatModel("fake", llm=llm)
    passage = Passage(doc_id="a", title="A", text="Alpha.", score=1.0)

    async def run():
        return "".join(
            [
                t
                async for t in model.stream(
                    "And then?", [passage], [{"question": "First?", "answer": "First."}]
                )
            ]
        )

    assert asyncio.run(run()) == "ok"
    roles = [role for role, _ in llm.messages]
    assert roles == ["system", "human", "ai", "human"]
    assert llm.messages[1:3] == [("human", "First?"), ("ai", "First.")]
    assert llm.messages[-1][1].endswith("Question: And then?")
