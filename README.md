# Agentic RAG Platform

[![CI](https://github.com/Siddharth-DWT/agentic-rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddharth-DWT/agentic-rag-platform/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An open reference app for retrieval-augmented generation with an agent in the loop. Ask a question, watch the sources arrive, then watch the answer stream in with citations.

It is built from the libraries production AI teams use: a **LangGraph** agent, **LlamaIndex** chunking and retrieval, a **FastAPI** server that streams over server-sent events, and a **Next.js** chat UI. It runs fully offline with no API keys; set `OPENAI_API_KEY` to answer with a real model.

## How it works

```mermaid
flowchart LR
    UI["Next.js chat UI"] -- "POST /chat" --> API["FastAPI"]
    API -- "SSE: sources, tokens, done" --> UI
    API --> G

    subgraph G["LangGraph agent"]
        direction LR
        R["retrieve"] -->|passages found| GEN["generate"]
        R -->|nothing found| NA["no_answer"]
    end

    R --> IDX[("LlamaIndex chunks + BM25 index")]
    GEN --> LLM["OpenAI-compatible model or offline extractive model"]
```

1. **Ingest.** Documents are split into overlapping chunks with LlamaIndex's `SentenceSplitter` and indexed with BM25.
2. **Retrieve.** The agent's `retrieve` node finds the best chunks and immediately streams them to the UI as sources.
3. **Route.** A conditional edge sends the run to `generate`, or to `no_answer` when nothing relevant was found, so the model is never asked to answer from nothing.
4. **Generate.** The answer streams token by token, citing sources as `[1]`, `[2]`. Tokens travel on LangGraph's `custom` stream, so the browser sees them while the graph is still running.

## Quick start

**Docker**

```bash
docker compose up --build
```

Open http://localhost:3000. The backend loads three sample documents (RAG, MCP and LangGraph) so you can ask questions straight away.

**Local**

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

# frontend, in a second terminal
cd frontend
npm install
npm run dev
```

**Use a real model.** Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Any OpenAI-compatible endpoint works through `OPENAI_BASE_URL`, including a local Ollama or vLLM server.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/health` | Status and the model in use |
| `GET` | `/documents` | Documents in the index, with chunk counts |
| `POST` | `/documents` | Add a document: `{"title": "...", "text": "..."}` |
| `POST` | `/chat` | Ask a question: `{"question": "..."}`. Streams `sources`, `token` and `done` events |

```bash
curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "How does an MCP server check a token?"}'
```

## Project layout

```
backend/
  app/retrieval.py   chunking, BM25 index, document store
  app/agent.py       the LangGraph graph
  app/llm.py         OpenAI-compatible model and offline fallback
  app/main.py        FastAPI app and SSE streaming
  data/              sample documents
  tests/             pytest suite, runs offline
frontend/            Next.js chat UI
```

## Roadmap

- [x] LangGraph agent with conditional routing and streamed answers
- [x] LlamaIndex chunking with BM25 retrieval
- [x] Next.js chat UI with live sources and citations
- [x] Offline test suite and CI
- [ ] Hybrid retrieval: vector search alongside BM25, merged with reciprocal rank fusion
- [ ] Reranking with a cross-encoder
- [ ] Document ingestion from PDFs, Office files and web pages
- [ ] Conversation memory with a LangGraph checkpointer
- [ ] Human-in-the-loop approval for tool calls
- [ ] The agent's tools exposed as a remote MCP server with OAuth
- [ ] Tracing with Langfuse
- [ ] Retrieval and answer-quality evaluation running in CI
- [ ] Chat UI on the Vercel AI SDK

## License

MIT
