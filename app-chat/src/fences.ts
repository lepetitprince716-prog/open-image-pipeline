// Shared between the server (src/agent.ts, src/sessions.ts) and the browser
// bundle (web/app.tsx): the fenced-block shapes both sides must agree on.
// Keep this module dependency-free so esbuild can bundle it into the page.

// One tool call in a turn's kept trace. `hint` is a short extract of the
// tool's input (the search query, the job id); may be empty. The page renders
// hints as plain text only -- never markdown, never a link.
export type ToolCall = {
  name: string;
  hint: string;
  error?: boolean; // the tool result came back as an error
};

// One label rule for a tool call, shared by the live activity feed and the
// kept trace so the same call reads identically in both.
export function toolLabel(call: ToolCall): string {
  return call.hint ? `${call.name}: ${call.hint}` : call.name;
}

// The page rejects traces above this outright (see parseToolCalls in
// web/app.tsx), so toolsFence below must never emit more.
export const TOOLS_MAX = 500;

// The trace travels as a fenced block the page parses back, posted at the end
// of each turn and stored verbatim in the session store, so live and replayed
// transcripts match byte for byte.
export function toolsFence(tools: ToolCall[]): string {
  const capped =
    tools.length > TOOLS_MAX
      ? [
          ...tools.slice(0, TOOLS_MAX - 1),
          { name: `${tools.length - TOOLS_MAX + 1} more tool calls`, hint: "" },
        ]
      : tools;
  return "```tools\n" + JSON.stringify(capped) + "\n```";
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

// The first message of a conversation becomes the session title; the sidebar
// mirrors it optimistically.
export const TITLE_MAX = 60;

// A render-job proposal drafted by the agent. The bridge captures it from the
// observed propose_render tool_use input -- never from agent prose -- mints
// `id`, and posts it as a ```proposal fence; web/app.tsx validates the fence
// strictly and renders the confirm card. Submission happens only when the
// user clicks the card: the browser POSTs the brief to /api/jobs with `id` as
// the Idempotency-Key, so a re-click or a replayed card dedupes server-side.
//
// `provider` is part of the contract from day one so later channels (civitai,
// nai, video) widen the union instead of reshaping it. Paid-confirmation
// fields (confirm_paid_generation, nai.paid) are deliberately NOT here and
// must never be: paid consent can only come from the user's own click.
export const PROPOSAL_PROVIDERS = ["local", "civitai"] as const;
export const PROPOSAL_TIERS = ["draft", "best", "artistic", "sequence"] as const;
export const PROPOSAL_BOXES = ["auto", "z3", "zzz"] as const;

export type Proposal = {
  id: string; // minted by the bridge; the card's Idempotency-Key
  provider: (typeof PROPOSAL_PROVIDERS)[number];
  tier: (typeof PROPOSAL_TIERS)[number];
  prompt: string;
  box: (typeof PROPOSAL_BOXES)[number];
  n?: number;
  seed?: number;
  steps?: number;
  cfg?: number;
  res?: string;
  ref?: string; // identity-anchor image path (sequence tier requires one)
  lora?: string; // primary LoRA spec, FILE[:WEIGHT] (from lora_search's file field)
  loras?: string[]; // extra LoRA specs, max 4 server-side
  style_lora?: string; // artistic tier style LoRA spec
  base?: string; // registered base-model stem (from bases_list)
  note?: string; // agent's one-line rationale, display only, never submitted
};

export function proposalFence(proposal: Proposal): string {
  return "```proposal\n" + JSON.stringify(proposal) + "\n```";
}

// The brief actually POSTed to /api/jobs: the proposal minus the card-only
// fields. Server-side brief.validate remains the real gate.
export function proposalToBrief(proposal: Proposal): Record<string, unknown> {
  const { id: _id, note: _note, ...brief } = proposal;
  return brief;
}
