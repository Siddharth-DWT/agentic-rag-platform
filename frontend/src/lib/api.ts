export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = {
  doc_id: string;
  title: string;
  text: string;
  score: number;
  matched_by: "keyword" | "vector" | "both";
};

export type DocumentInfo = {
  id: string;
  title: string;
  chunks: number;
};

export type ChatEvent =
  | { event: "sources"; sources: Source[] }
  | { event: "token"; text: string }
  | { event: "error"; message: string }
  | { event: "done" };

/** Parse complete server-sent-event blocks out of `buffer`; return them and the unparsed rest. */
export function parseEvents(buffer: string): { events: ChatEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    let name = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!name) continue;
    const payload = data ? JSON.parse(data) : {};
    events.push({ ...payload, event: name } as ChatEvent);
  }
  return { events, rest };
}

/** POST a question and yield events as the answer streams in. */
export async function* ask(question: string, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`The server answered ${response.status}.`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseEvents(buffer + value);
    buffer = parsed.rest;
    yield* parsed.events;
  }
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const response = await fetch(`${API_URL}/documents`);
  if (!response.ok) throw new Error(`The server answered ${response.status}.`);
  return response.json();
}

export async function addDocument(title: string, text: string): Promise<DocumentInfo> {
  const response = await fetch(`${API_URL}/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, text }),
  });
  if (!response.ok) throw new Error(`The document could not be added (${response.status}).`);
  return response.json();
}
