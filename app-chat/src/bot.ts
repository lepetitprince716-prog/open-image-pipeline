// Chat SDK wiring: the web adapter turns each browser POST into a DM on a
// thread (`web:{userId}:{conversationId}`) and pumps `thread.post()` back
// onto that request's response stream, so the handler surface is one
// callback and the reply channel is the HTTP response itself. The
// conversation ID inside the thread ID is a local session ID -- the page
// creates the session first (POST /api/sessions) and chats into it.

import { Chat } from "chat";
import { createWebAdapter } from "@chat-adapter/web";
import { createMemoryState } from "@chat-adapter/state-memory";
import { publishActivity } from "./activity";
import { enqueueTurn, runTurn } from "./agent";

// The web adapter's whole security boundary: this callback decides who the
// caller is for every route that exposes a conversation. Returning null
// produces a 401. This build trusts anyone who can reach the loopback port
// and gives them the same identity -- fine on 127.0.0.1, wrong anywhere
// else. Swap in a real session lookup before changing HOST.
function getUser(_request: Request) {
  return { id: "local", name: "you" };
}

// The same identity check, for the routes main.ts serves directly.
export { getUser as authenticate };

const adapters = {
  web: createWebAdapter({
    userName: "workbench-assistant",
    getUser,
    // The browser holds the live transcript and the local store holds the
    // durable one; don't have the Chat SDK cache message bodies too.
    persistMessageHistory: false,
  }),
};

// The activity feed subscribes by conversation ID; the bridge publishes by
// thread ID. Resolve the caller with the same getUser the chat route uses,
// then let the adapter build the ID.
export async function activityThreadId(
  request: Request,
  conversationId: string,
): Promise<string | null> {
  const user = await getUser(request);
  // Mirror the adapter's own rule: a ':' in the user ID would corrupt the
  // `web:{userId}:{conversationId}` round-trip, so it never names a thread.
  if (!user || user.id.includes(":")) return null;
  return adapters.web.encodeThreadId({ userId: user.id, conversationId });
}

export const bot = new Chat<typeof adapters>({
  userName: "workbench-assistant",
  adapters,
  // The Chat SDK requires a state adapter for its internals (locks, message
  // dedup); no conversation state lives there. Memory state is fine.
  state: createMemoryState(),
  // Don't use the SDK's per-thread lock for serialization: its TTL (30s) is
  // far shorter than an agent turn. Let every message through; the
  // per-thread queue in agent.ts runs the turns in order.
  concurrency: "concurrent",
});

bot.onDirectMessage(async (thread, message) => {
  const text = message.text?.trim();
  if (!text || message.author.isMe) return;
  // runTurn re-checks the ID really is one of this store's sessions before
  // touching it.
  const { conversationId } = adapters.web.decodeThreadId(thread.id);
  // The page tags the first message of a fresh conversation with a title
  // hint; the bridge writes it onto the session so the sidebar shows what
  // the chat is about.
  const metadata = (message.raw as { metadata?: { title?: string } }).metadata;
  const title = typeof metadata?.title === "string" ? metadata.title.trim() : undefined;
  // Await the whole turn. thread.post() writes onto this request's response
  // stream, which the adapter closes as soon as the handler returns.
  await enqueueTurn(thread.id, () =>
    runTurn(thread, conversationId, text, {
      title: title || undefined,
      activity: (item) => publishActivity(thread.id, item),
    }),
  );
});
