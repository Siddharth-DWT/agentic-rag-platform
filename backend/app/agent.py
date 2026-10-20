"""The RAG agent as a LangGraph graph.

    START → retrieve ─┬─ passages found → generate → END
                      └─ nothing found  → no_answer → END

With a reranker, `retrieve` fetches a longer candidate list and the reranker
picks the final top_k from it.

Answer tokens are emitted on LangGraph's `custom` stream as they are produced,
so the API can forward them to the browser while the graph is still running.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .llm import AnswerModel
from .rerank import Reranker, rerank
from .retrieval import DocumentStore, Passage

NO_ANSWER = "I couldn't find anything about that in the documents."


class AgentState(TypedDict, total=False):
    question: str
    passages: list[Passage]
    answer: str


def build_agent(
    store: DocumentStore,
    model: AnswerModel,
    top_k: int = 4,
    reranker: Reranker | None = None,
    candidates: int = 12,
):
    def retrieve(state: AgentState) -> AgentState:
        question = state["question"]
        if reranker:
            passages = rerank(reranker, question, store.search(question, top_k=candidates), top_k)
        else:
            passages = store.search(question, top_k=top_k)
        get_stream_writer()({"type": "sources", "sources": [_source(p) for p in passages]})
        return {"passages": passages}

    def route(state: AgentState) -> Literal["generate", "no_answer"]:
        return "generate" if state.get("passages") else "no_answer"

    async def generate(state: AgentState) -> AgentState:
        write = get_stream_writer()
        parts: list[str] = []
        async for token in model.stream(state["question"], state["passages"]):
            parts.append(token)
            write({"type": "token", "text": token})
        return {"answer": "".join(parts)}

    def no_answer(state: AgentState) -> AgentState:
        get_stream_writer()({"type": "token", "text": NO_ANSWER})
        return {"answer": NO_ANSWER}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route)
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)
    return graph.compile()


def _source(passage: Passage) -> dict[str, str | float]:
    return {
        "doc_id": passage.doc_id,
        "title": passage.title,
        "text": passage.text,
        "score": round(passage.score, 4),
        "matched_by": passage.matched_by,
    }
