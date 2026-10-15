import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.llm import ChatModel, ExtractiveModel, format_sources, model_from_env
from app.retrieval import Passage

PASSAGES = [
    Passage(doc_id="a", title="Alpha", text="Alpha text.", score=2.0),
    Passage(doc_id="b", title="Beta", text="Beta text.", score=1.0),
]


async def collect(stream) -> str:
    return "".join([t async for t in stream])


def test_sources_are_numbered_from_one():
    assert format_sources(PASSAGES) == "[1] Alpha\nAlpha text.\n\n[2] Beta\nBeta text."


def test_chat_model_streams_the_model_output():
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="Alpha is first [1].")]))
    model = ChatModel("fake", llm=fake)
    assert asyncio.run(collect(model.stream("Which is first?", PASSAGES))) == "Alpha is first [1]."


def test_extractive_model_cites_what_it_quotes():
    answer = asyncio.run(collect(ExtractiveModel().stream("q", PASSAGES)))
    assert "Alpha text. [1]" in answer and "Beta text. [2]" in answer


def test_no_key_means_the_extractive_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert model_from_env().name == "extractive"


def test_a_key_selects_the_configured_chat_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    assert model_from_env().name == "my-model"
