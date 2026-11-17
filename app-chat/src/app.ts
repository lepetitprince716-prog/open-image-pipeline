// The platform-neutral core: the four /api routes as one fetch-native Hono
// app. Every import is Web-standard; src/main.ts mounts this behind the Node
// server that also serves the page and the /api/jobs proxy.

import { Hono, type Context } from "hono";
import { subscribeActivity } from "./activity";
import { activityThreadId, authenticate, bot } from "./bot";
import { createSession, historyOf, listSessions, ownedSession } from "./sessions";

// Same identity boundary as /api/chat: an anonymous caller gets a 401 before
// any session data is touched.
function withUser(handler: (request: Request) => Promise<Response>) {
  return async (c: Context): Promise<Response> => {
    if (!(await authenticate(c.req.raw))) return new Response("Unauthorized", { status: 401 });
    return handler(c.req.raw);
  };
}

// Routes carry their full /api/... paths, so a plain route("/", api) mount
// matches with no prefix juggling.
export const api = new Hono();

api.post("/api/chat", (c) => bot.webhooks.web(c.req.raw));

// The sidebar: conversations are local store sessions, nothing more.
api.get(
  "/api/sessions",
  withUser(async () => Response.json(await listSessions())),
);
api.post(
  "/api/sessions",
  withUser(async () => Response.json(await createSession())),
);

// Transcript replay from the local store. The ownership check matters: the
// conversation ID comes from the browser.
api.get(
  "/api/history",
  withUser(async (request) => {
    const conversation = new URL(request.url).searchParams.get("conversation");
    if (!conversation) return new Response("missing ?conversation", { status: 400 });
    if (!(await ownedSession(conversation))) return new Response("Not found", { status: 404 });
    return Response.json(await historyOf(conversation));
  }),
);

// Server-sent activity for one conversation: tool calls and progress notes
// published by the bridge while its turn runs. Progress only -- message text
// travels exclusively on /api/chat.
api.get("/api/activity", (c) => activityTail(c.req.raw));

async function activityTail(request: Request): Promise<Response> {
  const conversation = new URL(request.url).searchParams.get("conversation");
  if (!conversation) return new Response("missing ?conversation", { status: 400 });
  const threadId = await activityThreadId(request, conversation);
  if (!threadId) return new Response("Unauthorized", { status: 401 });
  if (!(await ownedSession(conversation))) return new Response("Not found", { status: 404 });
  const encoder = new TextEncoder();
  let unsubscribe = () => {};
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // A first byte so EventSource reports the connection open immediately.
      controller.enqueue(encoder.encode(": connected\n\n"));
      unsubscribe = subscribeActivity(threadId, (item) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(item)}\n\n`));
        } catch {
          // The tab is gone but cancel() has not run yet -- drop the subscription.
          unsubscribe();
        }
      });
      // A closed tab that never receives another publish would otherwise
      // leave its subscriber registered until one arrives and throws.
      request.signal.addEventListener("abort", () => unsubscribe());
    },
    cancel() {
      unsubscribe();
    },
  });
  return new Response(stream, {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
  });
}
