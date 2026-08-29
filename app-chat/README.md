# app-chat — workbench chat agent sidecar

A browser chat for Z3Comfy Studio, adapted from Anthropic's
[Chat SDK × Claude Managed Agents quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/managed-agents/chat-sdk)
with the cloud bridge replaced by the local
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-typescript):
Vercel's Chat SDK owns the chat surface, `src/agent.ts` runs one SDK `query()`
per turn against the claude CLI's subscription credentials (no API key), and
the agent's only capabilities are ten in-process MCP tools wrapping the
loramake FastAPI on 127.0.0.1:8873 (health, jobs, gallery, LoRA/base catalog,
diagnostics, prompt assist, and propose_render).

The permission model is the design-console one: the agent may query freely
(health, jobs, gallery, diagnostics, prompt assist) but can never submit a
render. `propose_render` only produces a confirmation card; the browser POSTs
`/api/jobs` (proxied same-origin to 8873) when the user clicks Submit, with
the proposal id as the Idempotency-Key.

## Run

```sh
./run.sh start     # loopback :8874; requires the ACL proxy on 127.0.0.1:11882
./run.sh status
./run.sh stop
```

Dev mode (rebuilds the page bundle per request): `pnpm dev`.
Portless alias: `portless alias loramake-chat 8874` → https://loramake-chat.localhost

## Configuration (.env, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8874` | Listen port (loopback only) |
| `LORAMAKE_API` | `http://127.0.0.1:8873` | The workbench backend |
| `CHAT_MODEL` | `sonnet` | Model alias/id passed to the Agent SDK |

`ANTHROPIC_API_KEY` is deliberately unset by run.sh: the SDK discovers the
claude CLI's subscription credentials, so usage bills against the plan quota.

## Layout

- `src/agent.ts` — Agent SDK bridge: toolset, lockdown options, turn loop
- `src/bot.ts` / `src/app.ts` / `src/main.ts` — Chat SDK wiring, /api routes, server + jobs proxy
- `src/sessions.ts` — conversation store (`data/sessions.json`)
- `src/fences.ts` — fence contracts shared with the page (tools trace, proposal)
- `web/` — the chat page (React, bundled by esbuild at serve time)

Conversations are local: `data/sessions.json` holds the transcripts, and the
SDK's own session transcripts (under `~/.claude`) hold the model context,
resumed per turn via `sdk_session_id`.
