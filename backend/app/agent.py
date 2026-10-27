"""The RAG agent as a LangGraph graph.

    START → retrieve ─┬─ passages found → generate → END
                      └─ nothing found  → no_answer → END

With a reranker, `retrieve` fetches a longer candidate list and the reranker
picks the final top_k from it.

Each conversation is a LangGraph thread: a checkpointer keeps the state between
questions, and every answer appends one turn to `history`. A follow-up that
finds nothing on its own ("tell me more about it") is retried together with
the previous question, and the model sees the earlier turns.

Answer tokens are emitted on LangGraph's `custom` stream as they are produced,
so the API can forward them to the browser while the graph is still running.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .llm import AnswerModel, Turn
from .rerank import Reranker, rerank
from .retrieval import DocumentStore, Passage

NO_ANSWER = "I couldn't find anything about that in the documents."


class AgentState(TypedDict, total=False):
    question: str
    history: Annotated[list[Turn], operator.add]
    passages: list[Passage]
    answer: str


def build_agent(
    store: DocumentStore,
    model: AnswerModel,
    top_k: int = 4,
    reranker: Reranker | None = None,
    candidates: int = 12,
    checkpointer: BaseCheckpointSaver | None = None,
):
    def find(query: str) -> list[Passage]:
        if reranker:
            return rerank(reranker, query, store.search(query, top_k=candidates), top_k)
        return store.search(query, top_k=top_k)

    def retrieve(state: AgentState) -> AgentState:
        question = state["question"]
        passages = find(question)
        history = state.get("history") or []
        if not passages and history:
            # A follow-up often shares no terms with the documents; read it
            # together with the question it follows.
            passages = find(f"{history[-1]['question']} {question}")
        get_stream_writer()({"type": "sources", "sources": [_source(p) for p in passages]})
        return {"passages": passages}

    def route(state: AgentState) -> Literal["generate", "no_answer"]:
        return "generate" if state.get("passages") else "no_answer"

    async def generate(state: AgentState) -> AgentState:
        write = get_stream_writer()
        parts: list[str] = []
        async for token in model.stream(state["question"], state["passages"], state.get("history") or []):
            parts.append(token)
            write({"type": "token", "text": token})
        answer = "".join(parts)
        return {"answer": answer, "history": [{"question": state["question"], "answer": answer}]}

    def no_answer(state: AgentState) -> AgentState:
        get_stream_writer()({"type": "token", "text": NO_ANSWER})
        return {"answer": NO_ANSWER, "history": [{"question": state["question"], "answer": NO_ANSWER}]}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route)
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)
    return graph.compile(checkpointer=checkpointer)


def _source(passage: Passage) -> dict[str, str | float]:
    return {
        "doc_id": passage.doc_id,
        "title": passage.title,
        "text": passage.text,
        "score": round(passage.score, 4),
        "matched_by": passage.matched_by,
    }
