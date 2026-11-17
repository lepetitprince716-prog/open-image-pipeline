// The chat <-> Claude Agent SDK bridge. Each conversation maps to an SDK
// session resumed turn by turn (sdk_session_id in the local store); the
// agent's only capability surface is the in-process MCP toolset below --
// read-only HTTP wrappers over the loramake backend plus propose_render,
// which drafts a job proposal the user must confirm in the UI. The agent
// physically cannot submit a job: no tool of its POSTs /api/jobs.

import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import {
  createSdkMcpServer,
  query,
  tool,
  type CanUseTool,
  type Options,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import {
  PROPOSAL_BOXES,
  PROPOSAL_PROVIDERS,
  PROPOSAL_TIERS,
  proposalFence,
  toolLabel,
  toolsFence,
  truncate,
  TITLE_MAX,
  type Proposal,
  type ToolCall,
} from "./fences";
import {
  appendMessage,
  ownedSession,
  retitle,
  setSdkSessionId,
  DEFAULT_SESSION_TITLE,
} from "./sessions";

// Resolves to the app-chat/ root. The SDK stores its transcripts per cwd, so
// this must stay stable or `resume` silently loses the conversation context.
const APP_CHAT_DIR = fileURLToPath(new URL("..", import.meta.url));

const BASE = process.env.LORAMAKE_API || "http://127.0.0.1:8873";

// One line of turn progress: a tool call starting or finishing, a thinking
// block, the model writing. Surfaces decide how to show them (the web page
// streams them to a live feed). The user-facing reply never travels here.
export type ActivityItem = {
  kind: "model" | "thinking" | "tool" | "tool_done" | "tool_error" | "writing" | "retry" | "waiting";
  label: string;
};

export type TurnHooks = {
  activity?: (item: ActivityItem) => void;
  // First message of a fresh conversation: becomes the session title.
  title?: string;
};

// The minimal thread surface this module needs; satisfied by the Chat SDK's
// real thread object.
export interface BotThread {
  id: string;
  post(message: string | AsyncIterable<string>): Promise<unknown>;
}

const SESSION_ENDED = "This conversation is gone from the local store. Start a new chat from the sidebar.";

// Serialize turns per thread: replies in order even if the user double-sends.
const queues = new Map<string, Promise<void>>();

export function enqueueTurn(threadId: string, turn: () => Promise<void>): Promise<void> {
  const prev = queues.get(threadId) ?? Promise.resolve();
  const next = prev.catch(() => {}).then(turn);
  queues.set(threadId, next);
  next
    .catch(() => {})
    .finally(() => {
      if (queues.get(threadId) === next) queues.delete(threadId);
    });
  return next;
}

// ---------------------------------------------------------------------------
// The toolset: HTTP wrappers over the loramake FastAPI on 127.0.0.1:8873.

async function http(path: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<string> {
  const response = await fetch(`${BASE}${path}`, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`${init.method ?? "GET"} ${path} -> ${response.status}: ${truncate(body, 300)}`);
  }
  return body;
}

type ToolResult = { content: { type: "text"; text: string }[]; isError?: boolean };

const ok = (text: string): ToolResult => ({ content: [{ type: "text", text }] });

async function wrap(run: () => Promise<string>): Promise<ToolResult> {
  try {
    return ok(await run());
  } catch (err) {
    return { content: [{ type: "text", text: `ERROR: ${String(err)}` }], isError: true };
  }
}

function qs(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

// Shared brief fields for render_plan and propose_render. They describe the
// same generation surface; render_plan adds the compact planning envelope.
const BRIEF_COMMON = {
  provider: z
    .enum(PROPOSAL_PROVIDERS)
    .describe("local = on-prem boxes; civitai = paid cloud generation (best tier only; the user must confirm the spend on the card)"),
  tier: z
    .enum(PROPOSAL_TIERS)
    .describe("draft = fast turbo batch, best = quality, artistic = styled, sequence = identity-anchored (needs ref)"),
  prompt: z
    .string()
    .min(1)
    .max(4000)
    .describe("Composed prompt from assist. Do not invent Danbooru tags from memory."),
  box: z.enum(PROPOSAL_BOXES).describe("Render box; auto lets the backend pick"),
  n: z.number().int().min(1).max(16).optional().describe("Image count, 1..16"),
  seed: z.number().int().min(0).optional(),
  steps: z.number().int().min(1).max(60).optional(),
  cfg: z.number().min(1).max(15).optional(),
  res: z.string().max(20).optional().describe("WxH like 896x1152; draft also accepts portrait/square/landscape/tall"),
  ref: z.string().max(500).optional().describe("Existing image path (required by sequence tier)"),
  lora: z
    .string()
    .max(200)
    .optional()
    .describe("Primary LoRA spec FILE[:WEIGHT] — use the canonical_name from resource_search or render_plan"),
  loras: z.array(z.string().max(200)).max(4).optional().describe("Extra LoRA specs, at most 4"),
  style_lora: z.string().max(200).optional().describe("Style LoRA spec (artistic tier)"),
  base: z
    .string()
    .max(100)
    .optional()
    .describe("Registered base-model id (from render_plan selectable_bases or resource_search)"),
};

// propose_render's input contract, mirrored from app/brief.py (the server's
// brief.validate remains the real gate). `provider` is a literal today so the
// union can widen per channel later without reshaping the contract. Paid
// confirmation fields are deliberately absent -- see fences.ts.
const PROPOSE_SHAPE = {
  ...BRIEF_COMMON,
  box: BRIEF_COMMON.box.default("auto"),
  note: z.string().max(200).optional().describe("One-line rationale shown on the confirmation card"),
};

export const proposeInput = z.object(PROPOSE_SHAPE);

const RESOURCE_KINDS = [
  "lora",
  "base",
  "checkpoint",
  "vae",
  "controlnet",
  "upscale_models",
  "clip_vision",
  "diffusion_models",
  "ipadapter",
] as const;

const RESOURCE_FAMILIES = ["sdxl", "anima", "sd15", "any"] as const;
const RESOURCE_ROLES = ["character", "style", "concept", "pose", "detail"] as const;

const RENDER_PLAN_SHAPE = {
  provider: BRIEF_COMMON.provider.default("local"),
  tier: BRIEF_COMMON.tier.default("draft"),
  prompt: BRIEF_COMMON.prompt,
  box: BRIEF_COMMON.box.default("auto"),
  n: BRIEF_COMMON.n,
  seed: BRIEF_COMMON.seed,
  steps: BRIEF_COMMON.steps,
  cfg: BRIEF_COMMON.cfg,
  res: BRIEF_COMMON.res,
  ref: BRIEF_COMMON.ref,
  lora: BRIEF_COMMON.lora,
  loras: BRIEF_COMMON.loras,
  style_lora: BRIEF_COMMON.style_lora,
  base: BRIEF_COMMON.base,
  ir: z
    .record(z.any())
    .optional()
    .describe("Prompt IR from assist. Pass through; do not invent tags."),
  contexts: z
    .array(z.string().min(1).max(160))
    .min(1)
    .max(3)
    .optional()
    .describe("1..3 short search contexts, each <=160 characters"),
  per_role: z
    .number()
    .int()
    .min(1)
    .max(3)
    .optional()
    .describe("Candidates per role (1..3); default 3 when omitted"),
};

export const TOOL_NAMES = [
  "health",
  "jobs_list",
  "job_get",
  "jobs_search",
  "gallery_search",
  "resource_search",
  "render_plan",
  "diagnostics",
  "assist",
  "propose_render",
] as const;

// Pure request builders, exported so tests can verify encoding and body shape
// without spinning up the agent or the network.
export function buildResourceSearchPath(args: Record<string, unknown>): string {
  const params: Record<string, string | number | undefined> = {
    q: typeof args.q === "string" ? args.q : undefined,
    kind: typeof args.kind === "string" ? args.kind : undefined,
    family: typeof args.family === "string" ? args.family : undefined,
    role: typeof args.role === "string" ? args.role : undefined,
    box: typeof args.box === "string" ? args.box : undefined,
    limit: typeof args.limit === "number" ? args.limit : undefined,
  };
  return `/api/v1/resources/search${qs(params)}`;
}

export function buildRenderPlanBody(args: Record<string, unknown>): Record<string, unknown> {
  const { contexts, per_role, note, id, ir, ...rest } = args;
  const brief: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(rest)) {
    if (value !== undefined && value !== "") brief[key] = value;
  }
  const body: Record<string, unknown> = { brief };
  if (Array.isArray(contexts) && contexts.length > 0) body.contexts = contexts;
  if (typeof per_role === "number") body.per_role = per_role;
  if (ir && typeof ir === "object" && !Array.isArray(ir)) body.ir = ir;
  return body;
}

const loramake = createSdkMcpServer({
  name: "loramake",
  version: "1.0.0",
  tools: [
    tool("health", "Workbench backend and render-box health.", {}, () =>
      wrap(() => http("/api/health")),
    ),
    tool(
      "jobs_list",
      "Most recent render jobs, newest first.",
      { limit: z.number().int().min(1).max(200).default(20) },
      (args) => wrap(() => http(`/api/jobs${qs({ limit: args.limit })}`)),
    ),
    tool(
      "job_get",
      "One render job by id: brief, status, outputs.",
      { job_id: z.string().min(1).max(128) },
      (args) => wrap(() => http(`/api/jobs/${encodeURIComponent(args.job_id)}`)),
    ),
    tool(
      "jobs_search",
      "Paged job history; q substring-matches the brief, status filters exactly.",
      {
        q: z.string().max(200).default(""),
        status: z.string().max(20).default(""),
        limit: z.number().int().min(1).max(100).default(20),
        offset: z.number().int().min(0).default(0),
      },
      (args) => wrap(() => http(`/api/jobs/search${qs(args)}`)),
    ),
    tool(
      "gallery_search",
      "Search the SFW gallery catalog (curated renders).",
      {
        q: z.string().max(160).default(""),
        limit: z.number().int().min(1).max(100).default(20),
        character: z.string().max(64).optional(),
        tier: z.string().max(8).optional(),
      },
      (args) => wrap(() => http(`/api/v1/gallery${qs(args)}`)),
    ),
    tool(
      "resource_search",
      "Compact inventory search across LoRAs, bases, checkpoints, VAE, controlnets, and support assets. Returns the backend envelope verbatim.",
      {
        q: z.string().max(160).default("").describe("Substring or canonical-name query; empty returns a broad sample"),
        kind: z.enum(RESOURCE_KINDS).default("lora").describe("Resource kind"),
        family: z.enum(RESOURCE_FAMILIES).optional().describe("Architecture family filter (sdxl, anima, sd15, any)"),
        role: z.enum(RESOURCE_ROLES).optional().describe("LoRA/base role filter (character, style, concept, pose, detail)"),
        box: z.enum(PROPOSAL_BOXES).default("auto").describe("Box availability; auto means either boxa or boxb"),
        limit: z.number().int().min(1).max(20).default(8).describe("Max results, 1..20"),
      },
      (args) => wrap(() => http(buildResourceSearchPath(args as Record<string, unknown>))),
    ),
    tool(
      "render_plan",
      "Build a compact, read-only render plan from a brief + 1..3 contexts. Returns candidates, suggested picks, and selectable base ids. Never submits a job.",
      RENDER_PLAN_SHAPE,
      (args) =>
        wrap(() =>
          http(
            "/api/v1/render-plans",
            {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify(buildRenderPlanBody(args as Record<string, unknown>)),
            },
            60_000,
          ),
        ),
    ),
    tool(
      "diagnostics",
      "Read-only diagnostic probe of a render box. Omit probe to list capabilities.",
      {
        target: z.enum(["boxa", "boxb"]).optional(),
        probe: z
          .enum(["transport", "overview", "network", "comfyui", "listener", "gpu", "snapshot"])
          .optional(),
      },
      (args) =>
        wrap(() => {
          if (!args.target || !args.probe) return http("/api/diagnostics/capabilities");
          return http(
            "/api/diagnostics",
            {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ target: args.target, probe: args.probe }),
            },
            60_000,
          );
        }),
    ),
    tool(
      "assist",
      "Fill slots and compose a validated prompt (IR + dropped_tags). Mandatory first step for generation. Do not write Danbooru tags from memory.",
      { text: z.string().min(1).max(2000) },
      (args) =>
        wrap(() =>
          http(
            "/api/assist",
            {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ text: args.text }),
            },
            200_000, // the backend's CLI helper may take up to 3 minutes
          ),
        ),
    ),
    tool(
      "propose_render",
      "Draft a render-job PROPOSAL for the user to confirm. Never submits anything.",
      PROPOSE_SHAPE,
      async () =>
        ok(
          "Proposal recorded. The user sees a confirmation card and decides whether to submit. " +
            "Do not call this again for the same job, and do not claim the job ran.",
        ),
    ),
  ],
});

// ---------------------------------------------------------------------------
// Lockdown: the agent sees exactly the workbench tools and nothing else.

const canUseTool: CanUseTool = async (toolName, input) =>
  toolName.startsWith("mcp__loramake__")
    ? { behavior: "allow", updatedInput: input }
    : { behavior: "deny", message: "This assistant only uses the workbench tools." };

const SYSTEM_PROMPT = `You are the chat assistant for BoxaComfy Studio, a local image-generation \
workbench whose FastAPI backend (127.0.0.1:8873) queues render jobs onto two ComfyUI boxes, \
"boxa" and "boxb". You talk to the backend only through your tools.

Capabilities:
- health / jobs_list / job_get / jobs_search: queue state and job history.
- gallery_search: the curated SFW gallery catalog.
- resource_search: compact inventory search (LoRA, base, checkpoints, VAE, controlnets, etc.). \
Use it only for explicit "what do we have" or catalog questions, not as a mandatory step in \
every generation.
- render_plan: build a read-only, compact render plan for a brief + 1..3 contexts. It returns \
candidates, suggested picks, and selectable base ids. It never submits a job.
- diagnostics: read-only probes of the render boxes; call without arguments to list capabilities.
- assist: fill + compose + validate. Call it first for any generation request. It returns \
{ir, prompt, prompt_domain, dropped_tags, rewrites, loras_suggested}. Do not write Danbooru \
tags from memory.
- propose_render: drafts a render-job proposal. This is the only signal that creates a \
confirmation card; the user clicks it to submit.

Generation fast path:
1. Call assist with the user's description. Use the returned prompt (and ir). Do not invent a \
tag wall.
2. Call render_plan once with that prompt; pass ir when assist returned one. If the tier is \
not specified, default to "draft". Supply 1..3 short contexts (each <=160 chars) and an \
optional per_role (1..3).
3. Inspect the returned loras_by_role and selectable_bases. Pick high-confidence canonical_name \
values from the plan and a registered base id. No LoRA is a valid choice; do not use \
low-confidence candidates.
4. Call propose_render exactly once with the composed prompt. Put a short rationale in note.

Hard rules:
- You cannot run, submit, or queue jobs. propose_render only shows the user a confirmation card; \
the user decides. Never claim a job was submitted, queued, or ran because you proposed it.
- At most one propose_render call per reply.
- Tier semantics: draft = fast Anima-turbo batch, runs on boxa only (auto resolves to boxa); \
best = quality; artistic = styled pipeline; sequence = identity-anchored and requires ref, an \
existing image path taken from a previous job's outputs (see job_get).
- res is WxH like 896x1152 (draft also accepts presets portrait/square/landscape/tall); n is 1..16.
- provider "civitai" is paid cloud generation: tier must be "best", SFW only, n at most 12, \
res at most 2048 per side and divisible by 16, and lora/loras/style_lora/base/ref are not \
supported there. You can never authorize the spend: the user confirms it on the card, and \
until they do nothing is charged. Use civitai only when the user asks for it.
- Do not call health or diagnostics as part of the normal generation fast path.
- Answer in the user's language. Keep replies short; lead with the answer.`;

function agentOptions(resumeSessionId: string | undefined): Options {
  return {
    cwd: APP_CHAT_DIR,
    resume: resumeSessionId,
    model: process.env.CHAT_MODEL || "sonnet",
    systemPrompt: SYSTEM_PROMPT,
    mcpServers: { loramake },
    strictMcpConfig: true,
    // No CLAUDE.md, skills, or hooks leak into the agent: its whole identity
    // is the system prompt above.
    settingSources: [],
    allowedTools: TOOL_NAMES.map((name) => `mcp__loramake__${name}`),
    disallowedTools: [
      "Bash",
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "WebFetch",
      "WebSearch",
      "Task",
      "TodoWrite",
      "NotebookEdit",
      "Skill",
    ],
    canUseTool,
    // Never park a turn on an approval prompt: anything not pre-approved by
    // allowedTools/canUseTool is denied instead of waiting forever.
    permissionMode: "dontAsk",
    maxTurns: 30,
  };
}

// "web_search: solid-state batteries" reads better than "web_search". Tool
// inputs are free-form JSON; pull the first human-meaningful field and
// truncate it.
function toolCallOf(name: string, input: unknown): ToolCall {
  const called = truncate(name.replace(/^mcp__loramake__/, ""), 70);
  const args = input as Record<string, unknown> | null | undefined;
  for (const key of ["q", "kind", "text", "prompt", "job_id", "probe", "query"]) {
    const value = args?.[key];
    if (typeof value === "string" && value) return { name: called, hint: truncate(value, 70) };
  }
  return { name: called, hint: "" };
}

// One user message in, the agent's replies out. Runs one query() per turn,
// resuming the SDK session behind the conversation. Never rejects: every
// failure path ends in a message to the thread.
export async function runTurn(
  thread: BotThread,
  conversationId: string,
  text: string,
  hooks: TurnHooks = {},
): Promise<void> {
  const note = hooks.activity ?? (() => {});
  try {
    const session = await ownedSession(conversationId);
    if (!session) {
      await thread.post(SESSION_ENDED);
      return;
    }
    if (hooks.title && session.title === DEFAULT_SESSION_TITLE) {
      retitle(conversationId, truncate(hooks.title, TITLE_MAX));
    }
    appendMessage(conversationId, {
      id: randomUUID(),
      role: "user",
      parts: [{ type: "text", text }],
    });

    const tools: ToolCall[] = [];
    const openCalls = new Map<string, ToolCall>();
    // propose_render inputs seen this turn, keyed by tool_use id; one becomes
    // the turn's proposal only when its tool_result comes back clean.
    const pendingProposals = new Map<string, unknown>();
    let proposal: Proposal | undefined;
    let replied = false;
    let finished = false;

    const post = async (body: string) => {
      await thread.post(body);
      appendMessage(conversationId, {
        id: randomUUID(),
        role: "assistant",
        parts: [{ type: "text", text: body }],
      });
    };

    for await (const message of query({
      prompt: text,
      options: agentOptions(session.sdk_session_id),
    })) {
      switch (message.type) {
        case "system":
          // Each resume materializes under a fresh SDK session id; store the
          // current one so the next turn resumes from here. A turn that dies
          // before init simply starts a new SDK session next time.
          if (message.subtype === "init") setSdkSessionId(conversationId, message.session_id);
          break;
        case "assistant": {
          if (message.parent_tool_use_id) break; // subagent frames: not ours
          for (const block of message.message.content) {
            if (block.type === "text" && block.text.trim()) {
              replied = true;
              note({ kind: "writing", label: "writing the reply" });
              await post(block.text);
            } else if (block.type === "tool_use") {
              const call = toolCallOf(block.name, block.input);
              tools.push(call);
              openCalls.set(block.id, call);
              if (block.name.endsWith("propose_render")) pendingProposals.set(block.id, block.input);
              note({ kind: "tool", label: toolLabel(call) });
              console.log(`[agent] ${conversationId} tool: ${call.name}`);
            } else if (block.type === "thinking") {
              note({ kind: "thinking", label: "thinking" });
            }
          }
          break;
        }
        case "user": {
          // Tool results come back as user messages.
          const content = message.message.content;
          if (!Array.isArray(content)) break;
          for (const block of content) {
            if (typeof block !== "object" || block === null || !("type" in block)) continue;
            if (block.type !== "tool_result") continue;
            const result = block as { tool_use_id: string; is_error?: boolean };
            const call = openCalls.get(result.tool_use_id);
            openCalls.delete(result.tool_use_id);
            if (result.is_error && call) call.error = true;
            const pending = pendingProposals.get(result.tool_use_id);
            pendingProposals.delete(result.tool_use_id);
            if (pending !== undefined && !result.is_error) {
              // Card data comes from the observed tool input, never from
              // agent prose; parse it once more so a malformed input can't
              // reach the page.
              const parsed = proposeInput.safeParse(pending);
              if (parsed.success) proposal = { id: randomUUID(), ...parsed.data };
            }
            const name = call?.name ?? "tool";
            note(
              result.is_error
                ? { kind: "tool_error", label: `${name} failed` }
                : { kind: "tool_done", label: `${name} done` },
            );
          }
          break;
        }
        case "result": {
          if (message.subtype === "success" && !message.is_error) {
            finished = true;
          } else if (message.subtype === "success") {
            // API-level error surfaced as the result text.
            await post(message.result || "The turn failed on the model side. Try again.");
          } else {
            await post(`The turn stopped early (${message.subtype}). Try again.`);
          }
          break;
        }
      }
    }

    // A cleanly ended turn closes with its trailing fences: the kept
    // tool-call trace (the live feed is progress-only and gone on reload),
    // then the proposal card. Both are appended to the store, so live and
    // replayed transcripts match.
    if (finished && replied) {
      if (tools.length > 0) await post(toolsFence(tools));
      if (proposal) await post(proposalFence(proposal));
    }
  } catch (err) {
    console.error(`[agent] turn failed for ${conversationId}:`, err);
    await thread.post("Hit a snag on my end. Send that again?").catch(() => {});
  }
}
