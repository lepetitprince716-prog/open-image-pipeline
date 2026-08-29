// Serves the chat page and its assets, the /api routes from src/app.ts, and
// a narrow same-origin proxy to the loramake backend for the two paths the
// proposal card needs. Everything binds to loopback only.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { serve } from "@hono/node-server";
import * as esbuild from "esbuild";
import { Hono, type Context } from "hono";
import { api } from "./app";

const PORT = Number(process.env.PORT) || 8874;

// The demo getUser trusts every caller, so the default only listens on the
// loopback interface; set HOST only after wiring real auth into src/bot.ts.
const HOST = process.env.HOST || "127.0.0.1";

// Where the loramake FastAPI lives. The proposal card's POST goes through the
// proxy below so the page stays same-origin (the backend has no CORS config
// and needs none -- same pattern as scripts/panel_server.py).
const LORAMAKE = process.env.LORAMAKE_API || "http://127.0.0.1:8873";

const webFile = (name: string) => fileURLToPath(new URL(`../web/${name}`, import.meta.url));

// Dev rebuilds the bundle on each request (~100ms), so web/ edits show on
// reload -- production caches the first build.
const NODE_ENV = process.env.NODE_ENV || "development";
const PRODUCTION = NODE_ENV === "production";
let cachedBundle: string | undefined;
async function bundleApp(): Promise<string> {
  if (PRODUCTION && cachedBundle) return cachedBundle;
  const result = await esbuild.build({
    entryPoints: [webFile("app.tsx")],
    bundle: true,
    format: "esm",
    sourcemap: PRODUCTION ? false : "inline",
    minify: PRODUCTION,
    // React's entry points branch on this at require time; without the
    // define, the browser bundle would reference a `process` that isn't there.
    define: { "process.env.NODE_ENV": JSON.stringify(NODE_ENV) },
    write: false,
  });
  const text = result.outputFiles![0].text;
  if (PRODUCTION) cachedBundle = text;
  return text;
}

// Narrow proxy: exactly the two loramake paths the proposal card uses. No
// wildcard forward -- that would swallow the sidecar's own /api/* routes.
// Only the headers the backend cares about survive the hop.
async function proxyJobs(c: Context): Promise<Response> {
  const url = new URL(c.req.raw.url);
  const headers: Record<string, string> = {};
  const contentType = c.req.raw.headers.get("content-type");
  const idempotency = c.req.raw.headers.get("idempotency-key");
  if (contentType) headers["content-type"] = contentType;
  if (idempotency) headers["idempotency-key"] = idempotency;
  const body = c.req.raw.method === "POST" ? await c.req.raw.text() : undefined;
  const upstream = await fetch(`${LORAMAKE}${url.pathname}${url.search}`, {
    method: c.req.raw.method,
    headers,
    body,
    signal: AbortSignal.timeout(30_000),
  });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

const app = new Hono();

app.get("/", async (c) => c.html(await readFile(webFile("index.html"), "utf8")));
app.get("/app.css", async (c) =>
  c.body(await readFile(webFile("app.css"), "utf8"), 200, { "content-type": "text/css" }),
);
app.get("/app.js", async (c) =>
  c.body(await bundleApp(), 200, { "content-type": "text/javascript" }),
);

// Binary-safe passthrough for rendered images (the card's thumbnails).
async function proxyFiles(c: Context): Promise<Response> {
  const url = new URL(c.req.raw.url);
  const upstream = await fetch(`${LORAMAKE}${url.pathname}${url.search}`, {
    signal: AbortSignal.timeout(30_000),
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/octet-stream" },
  });
}

app.post("/api/jobs", proxyJobs);
app.get("/api/jobs/:id", proxyJobs);
app.post("/api/providers/whatif", proxyJobs);
app.get("/files", proxyFiles);

app.route("/", api);

serve(
  {
    fetch: app.fetch,
    port: PORT,
    hostname: HOST,
    // Agent turns hold the response open for minutes -- requestTimeout: 0
    // disables Node's 5-minute reap.
    serverOptions: { requestTimeout: 0 },
  },
  () => console.log(`Z3Comfy chat running at http://${HOST}:${PORT}`),
);
