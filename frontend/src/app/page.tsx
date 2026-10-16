"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

import { addDocument, ask, DocumentInfo, listDocuments, Source } from "@/lib/api";

type Turn = {
  question: string;
  answer: string;
  sources: Source[];
  status: "streaming" | "done" | "error";
};

const SUGGESTIONS = [
  "How does BM25 score a chunk?",
  "How does an MCP server check a token?",
  "How do I pause a graph for human approval?",
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [backendError, setBackendError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const busy = turns.at(-1)?.status === "streaming";

  useEffect(() => {
    listDocuments()
      .then(setDocuments)
      .catch(() => setBackendError("Can't reach the backend. Start it with `uvicorn app.main:app`."));
  }, []);

  function update(patch: (turn: Turn) => Turn) {
    setTurns((all) => [...all.slice(0, -1), patch(all[all.length - 1])]);
  }

  async function submit(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setQuestion("");
    setTurns((all) => [...all, { question: q, answer: "", sources: [], status: "streaming" }]);
    abort.current = new AbortController();
    try {
      for await (const ev of ask(q, abort.current.signal)) {
        if (ev.event === "sources") update((t) => ({ ...t, sources: ev.sources }));
        else if (ev.event === "token") update((t) => ({ ...t, answer: t.answer + ev.text }));
        else if (ev.event === "error") update((t) => ({ ...t, answer: ev.message, status: "error" }));
      }
      update((t) => (t.status === "streaming" ? { ...t, status: "done" } : t));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      update((t) => ({ ...t, answer: t.answer || message, status: "error" }));
    }
  }

  return (
    <div className="mx-auto grid w-full max-w-6xl flex-1 gap-6 px-4 py-8 md:grid-cols-[1fr_280px]">
      <main className="flex min-w-0 flex-col gap-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Agentic RAG</h1>
          <p className="text-sm text-neutral-500">
            Answers come only from the indexed documents, with every claim cited.
          </p>
        </header>

        {backendError && (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
            {backendError}
          </p>
        )}

        {turns.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => submit(s)}
                className="rounded-full border border-neutral-300 px-3 py-1.5 text-sm hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-900"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <ol className="flex flex-col gap-8">
          {turns.map((turn, i) => (
            <li key={i} className="flex flex-col gap-3">
              <p className="self-end rounded-2xl bg-neutral-900 px-4 py-2 text-white dark:bg-neutral-100 dark:text-neutral-900">
                {turn.question}
              </p>
              {turn.sources.length > 0 && <Sources sources={turn.sources} />}
              <div
                className={`whitespace-pre-wrap leading-relaxed ${turn.status === "error" ? "text-red-600" : ""}`}
              >
                {turn.answer}
                {turn.status === "streaming" && <span className="animate-pulse">▍</span>}
              </div>
            </li>
          ))}
        </ol>

        <form
          onSubmit={(e: FormEvent) => {
            e.preventDefault();
            submit(question);
          }}
          className="sticky bottom-4 mt-auto flex gap-2 rounded-xl border border-neutral-300 bg-[var(--background)] p-2 shadow-sm dark:border-neutral-700"
        >
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <input
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about the documents…"
            className="min-w-0 flex-1 bg-transparent px-2 outline-none"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="rounded-lg bg-neutral-900 px-4 py-2 text-sm text-white disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>
      </main>

      <aside className="flex flex-col gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">Documents</h2>
        <ul className="flex flex-col gap-1 text-sm">
          {documents.map((d) => (
            <li key={d.id} className="flex justify-between gap-2">
              <span className="truncate">{d.title}</span>
              <span className="shrink-0 tabular-nums text-neutral-500">{d.chunks} chunks</span>
            </li>
          ))}
        </ul>
        <AddDocument onAdded={(d) => setDocuments((all) => [...all.filter((x) => x.id !== d.id), d])} />
      </aside>
    </div>
  );
}

const MATCH_LABEL: Record<Source["matched_by"], string> = {
  keyword: "keyword match",
  vector: "meaning match",
  both: "keyword + meaning",
};

function Sources({ sources }: { sources: Source[] }) {
  return (
    <ol className="grid gap-2 sm:grid-cols-2">
      {sources.map((s, i) => (
        <li key={i} className="rounded-lg border border-neutral-200 p-3 text-sm dark:border-neutral-800">
          <p className="flex items-baseline justify-between gap-2 font-medium">
            <span className="min-w-0 truncate">
              <span className="mr-1 text-neutral-500">[{i + 1}]</span>
              {s.title}
            </span>
            <span className="shrink-0 rounded bg-neutral-100 px-1.5 py-0.5 text-xs font-normal text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300">
              {MATCH_LABEL[s.matched_by]}
            </span>
          </p>
          <p className="mt-1 line-clamp-3 text-neutral-600 dark:text-neutral-400">{s.text}</p>
        </li>
      ))}
    </ol>
  );
}

function AddDocument({ onAdded }: { onAdded: (d: DocumentInfo) => void }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    try {
      const doc = await addDocument(title.trim(), text);
      onAdded(doc);
      setTitle("");
      setText("");
      setMessage(`Added “${doc.title}” in ${doc.chunks} chunks.`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "The document could not be added.");
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-2 border-t border-neutral-200 pt-4 dark:border-neutral-800">
      <h3 className="text-sm font-medium">Add a document</h3>
      <label htmlFor="doc-title" className="sr-only">
        Title
      </label>
      <input
        id="doc-title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
        className="rounded-md border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
      />
      <label htmlFor="doc-text" className="sr-only">
        Text
      </label>
      <textarea
        id="doc-text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste text here"
        rows={5}
        className="rounded-md border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
      />
      <button
        type="submit"
        disabled={!title.trim() || !text.trim()}
        className="rounded-md border border-neutral-300 px-3 py-1.5 text-sm disabled:opacity-40 dark:border-neutral-700"
      >
        Add to index
      </button>
      {message && <p className="text-xs text-neutral-500">{message}</p>}
    </form>
  );
}
