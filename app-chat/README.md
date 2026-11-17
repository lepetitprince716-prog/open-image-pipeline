# app-chat — workbench chat agent sidecar

A browser chat for BoxaComfy Studio, adapted from Anthropic's
[Chat SDK × Claude Managed Agents quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/managed-agents/chat-sdk)
with the cloud bridge replaced by the local
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-typescript).
Vercel's Chat SDK owns the chat surface; `src/agent.ts` runs one SDK `query()`
per turn against the claude CLI's subscription credentials (no API key).

The agent sees exactly ten in-process MCP tools:

1. `health`
2. `jobs_list`
3. `job_get`
4. `jobs_search`
5. `gallery_search`
6. `resource_search`
7. `render_plan`
8. `diagnostics`
9. `assist`
10. `propose_render`

`resource_search` calls the bounded SFW endpoint
`GET /api/v1/resources/search` (`loramake.resource-search/v1`); it no longer
fetches or filters the full `/api/meta` payload. `assist` calls
`POST /api/assist` (fill + compose + validate) and is the first step of a
generation. `render_plan` calls `POST /api/v1/render-plans` with
`{brief, contexts?, per_role?, ir?}` and returns `loramake.render-plan/v1`.
Both planning responses come from the shared `loramake/model_resources.py`
seam, stay compact, and do not submit GPU work. The app-chat toolset does not
expose the NSFW planning routes. Agents do not write Danbooru tags from memory.

The permission model is the design-console one. Read/search, diagnostics,
prompt assist and render planning may call their narrow backend endpoints, but
no agent tool POSTs `/api/jobs`. `propose_render` only emits a strictly parsed
confirmation card; the browser POSTs `/api/jobs` through the same-origin proxy
when the user clicks Submit, using the proposal id as `Idempotency-Key`. A plan
or proposal therefore never means a job was queued or rendered.

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

`ANTHROPIC_API_KEY` is deliberately unset by `run.sh`: the SDK discovers the
claude CLI's subscription credentials, so usage bills against the plan quota.

## Layout

- `src/agent.ts` — Agent SDK bridge: ten-tool surface, lockdown options, turn loop
- `src/bot.ts` / `src/app.ts` / `src/main.ts` — Chat SDK wiring, `/api` routes, server + jobs proxy
- `src/sessions.ts` — conversation store (`data/sessions.json`)
- `src/fences.ts` — fence contracts shared with the page (tools trace, proposal)
- `web/` — the chat page (React, bundled by esbuild at serve time)

Conversations are local: `data/sessions.json` holds the transcripts, and the
SDK's own session transcripts (under `~/.claude`) hold the model context,
resumed per turn via `sdk_session_id`.

This README describes the code contract only; it does not assert a browser or
live Studio/GPU verification for the current checkout.
