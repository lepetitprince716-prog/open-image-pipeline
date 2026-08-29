// The sidebar's data source and the transcript store. Conversations live in
// one local JSON file (data/sessions.json): the bridge appends the user
// message and every assistant bubble/fence exactly as posted, so /api/history
// replays byte-identical transcripts after a restart. The Agent SDK keeps the
// model-side context in its own transcript (resumed via sdk_session_id); this
// store only owns what the page renders.

import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { rename, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

export const DEFAULT_SESSION_TITLE = "New chat";

export type SessionSummary = {
  id: string;
  title: string;
  status: string;
  created_at: string;
};

// What useChat expects as initial messages: id + role + text parts.
export type UIMessageJSON = {
  id: string;
  role: "user" | "assistant";
  parts: { type: "text"; text: string }[];
};

type StoredSession = {
  id: string;
  title: string;
  created_at: string;
  sdk_session_id?: string;
  messages: UIMessageJSON[];
};

const DATA_DIR = fileURLToPath(new URL("../data", import.meta.url));
const STORE_PATH = `${DATA_DIR}/sessions.json`;

// Single-user local tool: load once, mutate in memory, persist on change.
// Writes are atomic (tmp + rename) and chained so they never interleave;
// turns are already serialized per thread by enqueueTurn in src/agent.ts.
const sessions = new Map<string, StoredSession>();

function load(): void {
  mkdirSync(DATA_DIR, { recursive: true });
  let raw: string;
  try {
    raw = readFileSync(STORE_PATH, "utf8");
  } catch {
    return; // first boot: empty store
  }
  try {
    const parsed = JSON.parse(raw) as StoredSession[];
    for (const session of parsed) {
      if (typeof session?.id === "string") sessions.set(session.id, session);
    }
  } catch (err) {
    // A corrupt store must fail loudly, not silently wipe every conversation
    // on the next persist.
    throw new Error(`sessions store unreadable (${STORE_PATH}): ${String(err)}`);
  }
}
load();

let persisting: Promise<void> = Promise.resolve();

function persist(): void {
  const snapshot = JSON.stringify([...sessions.values()], null, 1);
  persisting = persisting
    .catch(() => {})
    .then(async () => {
      const tmp = `${STORE_PATH}.tmp`;
      await writeFile(tmp, snapshot, "utf8");
      await rename(tmp, STORE_PATH);
    });
  persisting.catch((err) => console.error("[sessions] persist failed:", err));
}

function toSummary(session: StoredSession): SessionSummary {
  return {
    id: session.id,
    title: session.title || DEFAULT_SESSION_TITLE,
    status: "ready",
    created_at: session.created_at,
  };
}

export async function listSessions(): Promise<SessionSummary[]> {
  return [...sessions.values()]
    .map(toSummary)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
}

// "New chat" in the sidebar. The session is created before the first message
// is sent, so its ID can be the useChat conversation ID from the start; the
// first message retitles it (see runTurn in src/agent.ts).
export async function createSession(): Promise<SessionSummary> {
  const session: StoredSession = {
    id: randomUUID(),
    title: DEFAULT_SESSION_TITLE,
    created_at: new Date().toISOString(),
    messages: [],
  };
  sessions.set(session.id, session);
  persist();
  console.log(`[agent] new session ${session.id}`);
  return toSummary(session);
}

// Conversation IDs come from the browser, so never trust one blindly: it must
// look like an ID and resolve to a session this store created. This is the
// gate for every route that touches a session.
export async function ownedSession(sessionId: string): Promise<StoredSession | null> {
  if (!/^[A-Za-z0-9_-]{4,128}$/.test(sessionId)) return null;
  return sessions.get(sessionId) ?? null;
}

export async function historyOf(sessionId: string): Promise<UIMessageJSON[]> {
  return sessions.get(sessionId)?.messages ?? [];
}

// Bridge-side mutations. All of them assume ownedSession() already passed.

export function appendMessage(sessionId: string, message: UIMessageJSON): void {
  const session = sessions.get(sessionId);
  if (!session) return;
  session.messages.push(message);
  persist();
}

export function setSdkSessionId(sessionId: string, sdkSessionId: string): void {
  const session = sessions.get(sessionId);
  if (!session || session.sdk_session_id === sdkSessionId) return;
  session.sdk_session_id = sdkSessionId;
  persist();
}

export function retitle(sessionId: string, title: string): void {
  const session = sessions.get(sessionId);
  if (!session || session.title !== DEFAULT_SESSION_TITLE) return;
  session.title = title;
  persist();
}
