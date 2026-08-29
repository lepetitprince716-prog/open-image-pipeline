// Browser chat client. The sidebar lists local sessions and picking one
// replays its transcript from the store (/api/history). `useChat` POSTs the
// active conversation to /api/chat and reads the AI SDK message stream off
// the response; the server holds that response open for the whole agent
// turn. /api/activity is a second, message-free stream: the tool calls and
// progress notes the bridge observes while it works.

import { memo, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "react-markdown";
import { useChat, type UIMessage } from "@chat-adapter/web/react";
// Shared with the server: the same fence shapes, so live, replayed, and
// rendered output can't drift.
import {
  proposalToBrief,
  toolLabel,
  truncate,
  PROPOSAL_BOXES,
  PROPOSAL_PROVIDERS,
  PROPOSAL_TIERS,
  TITLE_MAX,
  TOOLS_MAX,
  type Proposal,
  type ToolCall,
} from "../src/fences";

type Activity = { kind: string; label: string };
type SessionSummary = { id: string; title: string; status: string; created_at: string };

// The bridge closes a proposing turn with a ```proposal fence. Only a message
// that is exactly that fence renders as a card, and the payload is rebuilt
// field by field from the validated JSON -- extra keys in a crafted fence
// never reach the submit body, and the prompt renders as plain text only.
const PROPOSAL_MESSAGE = /^```proposal\n(.*)\n```$/s;
const ID_SHAPE = /^[A-Za-z0-9-]{8,64}$/;

function optionalInt(value: unknown, min: number, max: number): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > max) {
    throw new Error("bad int");
  }
  return value;
}

function optionalString(value: unknown, max: number): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length === 0 || value.length > max) {
    throw new Error("bad string");
  }
  return value;
}

function parseProposal(text: string): Proposal | null {
  const match = PROPOSAL_MESSAGE.exec(text.trim());
  if (!match) return null;
  try {
    const raw = JSON.parse(match[1]) as Record<string, unknown>;
    if (typeof raw.id !== "string" || !ID_SHAPE.test(raw.id)) return null;
    if (!(PROPOSAL_PROVIDERS as readonly string[]).includes(raw.provider as string)) return null;
    if (!(PROPOSAL_TIERS as readonly string[]).includes(raw.tier as string)) return null;
    if (!(PROPOSAL_BOXES as readonly string[]).includes(raw.box as string)) return null;
    if (typeof raw.prompt !== "string" || !raw.prompt || raw.prompt.length > 4000) return null;
    const cfg = raw.cfg;
    if (cfg !== undefined && (typeof cfg !== "number" || !Number.isFinite(cfg))) return null;
    let loras: string[] | undefined;
    if (raw.loras !== undefined) {
      if (!Array.isArray(raw.loras) || raw.loras.length > 4) return null;
      loras = raw.loras.map((spec) => {
        if (typeof spec !== "string" || !spec || spec.length > 200) throw new Error("bad lora spec");
        return spec;
      });
    }
    // Rebuild from known fields only -- never spread the parsed object.
    return {
      id: raw.id,
      provider: raw.provider as Proposal["provider"],
      tier: raw.tier as Proposal["tier"],
      prompt: raw.prompt,
      box: raw.box as Proposal["box"],
      n: optionalInt(raw.n, 1, 16),
      seed: optionalInt(raw.seed, 0, Number.MAX_SAFE_INTEGER),
      steps: optionalInt(raw.steps, 1, 60),
      cfg: cfg as number | undefined,
      res: optionalString(raw.res, 20),
      ref: optionalString(raw.ref, 500),
      lora: optionalString(raw.lora, 200),
      loras,
      style_lora: optionalString(raw.style_lora, 200),
      base: optionalString(raw.base, 100),
      note: optionalString(raw.note, 200),
    };
  } catch {
    return null;
  }
}

// The kept tool-call trace travels the same way: a message that is exactly a
// ```tools fence, posted by the bridge at the end of each turn. Entries
// render as plain text only -- never markdown, never a link.
const TOOLS_MESSAGE = /^```tools\n(.*)\n```$/s;
const FIELD_MAX = 200;

function parseToolCalls(text: string): ToolCall[] | null {
  const match = TOOLS_MESSAGE.exec(text.trim());
  if (!match) return null;
  try {
    const value = JSON.parse(match[1]) as unknown;
    if (!Array.isArray(value) || value.length === 0 || value.length > TOOLS_MAX) return null;
    const valid = value.every((entry: Partial<ToolCall>) => {
      return (
        typeof entry === "object" &&
        entry !== null &&
        typeof entry.name === "string" &&
        entry.name.length > 0 &&
        entry.name.length <= FIELD_MAX &&
        typeof entry.hint === "string" &&
        entry.hint.length <= FIELD_MAX &&
        (entry.error === undefined || typeof entry.error === "boolean")
      );
    });
    return valid ? (value as ToolCall[]) : null;
  } catch {}
  return null;
}

function ToolTrace({ tools }: { tools: ToolCall[] }) {
  const failures = tools.filter((tool) => tool.error).length;
  return (
    <details className="tool-trace">
      <summary>
        {tools.length} tool call{tools.length === 1 ? "" : "s"}
        {failures > 0 ? ` (${failures} failed)` : ""}
      </summary>
      <ul className="activity">
        {tools.map((tool, index) => (
          <li key={index} className={tool.error ? "tool_error" : "tool"}>
            {toolLabel(tool)}
          </li>
        ))}
      </ul>
    </details>
  );
}

const TERMINAL_STATUSES = ["done", "failed", "error", "cancelled"];

// The fixed result block a finished job renders on its card: outcome, engine
// seconds, and the produced images (served through the /files proxy).
type JobOutcome = { files: string[]; seconds?: number; errorDetail?: string };

type SubmitState =
  | { phase: "idle" }
  | { phase: "submitting" }
  | { phase: "submitted"; jobId: string; status: string; outcome?: JobOutcome }
  | { phase: "dismissed" }
  | { phase: "error"; detail: string };

type JobRecord = {
  status?: string;
  error?: string | null;
  result?: {
    images?: string[];
    new_files?: string[];
    timings?: { engine_s?: number };
  } | null;
};

function outcomeOf(job: JobRecord): JobOutcome {
  return {
    files: job.result?.images ?? job.result?.new_files ?? [],
    seconds: job.result?.timings?.engine_s,
    errorDetail: job.error ?? undefined,
  };
}

type CostEntry = { amount: number; currency: string };

type Estimate =
  | { phase: "loading" }
  | { phase: "ready"; cost: CostEntry[] }
  | { phase: "unavailable" };

// The confirmation card: the one place a render job can actually be
// submitted. The browser POSTs the brief itself; the agent never does.
function ProposalCard({ proposal }: { proposal: Proposal }) {
  const [state, setState] = useState<SubmitState>({ phase: "idle" });
  // Paid providers charge on submit. The agent's contract has no paid field
  // at all -- this checkbox is the only place confirm_paid_generation can be
  // set, and the server refuses external briefs without it.
  const paidProvider = proposal.provider !== "local";
  const [paidConfirmed, setPaidConfirmed] = useState(false);
  const [estimate, setEstimate] = useState<Estimate>({ phase: "loading" });

  // Paid providers: fetch the free whatif cost preview up front, so the spend
  // is known before the confirm checkbox is ever ticked. Advisory only -- the
  // backend re-runs its own whatif on submit.
  useEffect(() => {
    if (!paidProvider) return;
    let cancelled = false;
    void fetch("/api/providers/whatif", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        provider: proposal.provider,
        prompt: proposal.prompt,
        n: proposal.n,
        seed: proposal.seed,
        steps: proposal.steps,
        cfg: proposal.cfg,
        res: proposal.res,
      }),
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((body: { cost?: CostEntry[] } | null) => {
        if (cancelled) return;
        if (body?.cost?.length) setEstimate({ phase: "ready", cost: body.cost });
        else setEstimate({ phase: "unavailable" });
      })
      .catch(() => {
        if (!cancelled) setEstimate({ phase: "unavailable" });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paidProvider, proposal.id]);

  // Follow the submitted job until it reaches a terminal status, then keep
  // the full outcome (files, timing, error) for the result block.
  useEffect(() => {
    if (state.phase !== "submitted" || TERMINAL_STATUSES.includes(state.status)) return;
    const timer = setInterval(() => {
      void fetch(`/api/jobs/${encodeURIComponent(state.jobId)}`)
        .then((response) => (response.ok ? response.json() : null))
        .then((job: JobRecord | null) => {
          if (!job?.status || job.status === state.status) return;
          setState({
            phase: "submitted",
            jobId: state.jobId,
            status: job.status,
            outcome: TERMINAL_STATUSES.includes(job.status) ? outcomeOf(job) : undefined,
          });
        })
        .catch(() => {});
    }, 5000);
    return () => clearInterval(timer);
  }, [state]);

  const submit = async () => {
    setState({ phase: "submitting" });
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          // The proposal id doubles as the idempotency key: a double click or
          // a replayed card resolves to the same job server-side.
          "idempotency-key": proposal.id,
        },
        body: JSON.stringify({
          ...proposalToBrief(proposal),
          ...(paidProvider && paidConfirmed ? { confirm_paid_generation: true } : {}),
        }),
      });
      const body = (await response.json().catch(() => ({}))) as {
        id?: string;
        status?: string;
        detail?: unknown;
      };
      if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : `submit failed (${response.status})`;
        setState({ phase: "error", detail });
        return;
      }
      setState({ phase: "submitted", jobId: body.id ?? "?", status: body.status ?? "queued" });
    } catch (err) {
      setState({ phase: "error", detail: String(err) });
    }
  };

  const params = [
    `${proposal.provider}/${proposal.tier}`,
    `box ${proposal.box}`,
    proposal.n !== undefined && `n=${proposal.n}`,
    proposal.seed !== undefined && `seed=${proposal.seed}`,
    proposal.steps !== undefined && `steps=${proposal.steps}`,
    proposal.cfg !== undefined && `cfg=${proposal.cfg}`,
    proposal.res && `res=${proposal.res}`,
    proposal.ref && `ref=${truncate(proposal.ref, 40)}`,
    proposal.base && `base=${truncate(proposal.base, 40)}`,
    proposal.lora && `lora=${truncate(proposal.lora, 50)}`,
    ...(proposal.loras ?? []).map((spec) => `+lora=${truncate(spec, 50)}`),
    proposal.style_lora && `style_lora=${truncate(proposal.style_lora, 50)}`,
  ]
    .filter(Boolean)
    .join(" · ");

  if (state.phase === "dismissed") {
    return <aside className="proposal-card dismissed">Proposal dismissed.</aside>;
  }
  return (
    <aside className="proposal-card">
      <p className="brief-title">Render proposal</p>
      <p className="brief-meta">{params}</p>
      <p className="proposal-prompt">{proposal.prompt}</p>
      {proposal.note && <p className="proposal-note">{proposal.note}</p>}
      {state.phase === "error" && <p className="error">{state.detail}</p>}
      {state.phase === "submitted" ? (
        <>
          <p className="proposal-status">
            job {state.jobId} · {state.status}
          </p>
          {state.outcome &&
            (state.status === "done" ? (
              <div className="proposal-result">
                <p className="proposal-status">
                  {state.outcome.files.length} image{state.outcome.files.length === 1 ? "" : "s"}
                  {state.outcome.seconds !== undefined && ` · ${Math.round(state.outcome.seconds)}s engine`}
                </p>
                <div className="result-thumbs">
                  {state.outcome.files.map((path) => {
                    const src = `/files?p=${encodeURIComponent(path)}`;
                    return (
                      <a key={path} href={src} target="_blank" rel="noreferrer">
                        <img src={src} alt="render output" />
                      </a>
                    );
                  })}
                </div>
              </div>
            ) : (
              <p className="error">{state.outcome.errorDetail ?? `job ended: ${state.status}`}</p>
            ))}
        </>
      ) : (
        <>
          {paidProvider && (
            <p className="brief-meta">
              {estimate.phase === "loading" && "estimating cost..."}
              {estimate.phase === "ready" &&
                `estimated cost: ${estimate.cost.map((entry) => `${entry.amount} ${entry.currency}`).join(" + ")} Buzz`}
              {estimate.phase === "unavailable" && "cost estimate unavailable"}
            </p>
          )}
          {paidProvider && (
            <label className="paid-confirm">
              <input
                type="checkbox"
                checked={paidConfirmed}
                onChange={(event) => setPaidConfirmed(event.target.checked)}
              />
              Confirm paid generation on {proposal.provider} (spends Buzz)
            </label>
          )}
          <div className="proposal-actions">
            <button
              onClick={() => void submit()}
              disabled={state.phase === "submitting" || (paidProvider && !paidConfirmed)}
            >
              {state.phase === "submitting" ? "Submitting..." : "Submit render"}
            </button>
            <button className="ghost" onClick={() => setState({ phase: "dismissed" })}>
              Dismiss
            </button>
          </div>
        </>
      )}
    </aside>
  );
}

// Cheap pre-test so the streaming bubble doesn't run trim + regex on every delta.
const FENCED = /^\s*```/;

// memo: don't re-parse every bubble's markdown on each streaming delta.
const Bubble = memo(function Bubble({ role, text }: { role: string; text: string }) {
  if (role === "assistant" && FENCED.test(text)) {
    const proposal = parseProposal(text);
    if (proposal) return <ProposalCard proposal={proposal} />;
    const tools = parseToolCalls(text);
    if (tools) return <ToolTrace tools={tools} />;
  }
  return (
    <article className={role}>
      <Markdown>{text}</Markdown>
    </article>
  );
});

// Live progress for this conversation, isolated from the transcript so feed
// events never re-render the messages.
function ActivityFeed({ conversationId, busy }: { conversationId: string; busy: boolean }) {
  const [items, setItems] = useState<(Activity & { key: number })[]>([]);
  const nextKey = useRef(0);

  useEffect(() => {
    const source = new EventSource(`/api/activity?conversation=${encodeURIComponent(conversationId)}`);
    source.onmessage = (event) => {
      const item = JSON.parse(event.data) as Activity;
      setItems((current) => [...current, { ...item, key: nextKey.current++ }].slice(-6));
    };
    return () => source.close();
  }, [conversationId]);

  useEffect(() => {
    if (busy) setItems([]);
  }, [busy]);

  if (!busy) return null;
  return (
    <div className="working">
      <p className="pulse">working</p>
      <ul className="activity">
        {items.map((item) => (
          <li key={item.key} className={item.kind}>
            {item.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

// One open conversation. Keyed by session ID in App, so switching sessions
// remounts with a fresh useChat whose conversation ID is the session ID and
// whose initial messages are the replayed transcript.
function Conversation({
  sessionId,
  history,
  onFirstMessage,
}: {
  sessionId: string;
  history: UIMessage[];
  onFirstMessage: (title: string) => void;
}) {
  const { messages, sendMessage, stop, status, error } = useChat({
    threadId: sessionId,
    messages: history,
  });
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  // The whole turn is one response: "streaming" covers the span between the
  // send and the reply, so it doubles as the progress indicator.
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    endRef.current?.scrollIntoView();
  }, [messages, status]);

  return (
    <>
      <section className="transcript">
        {messages.length === 0 && (
          <p className="hint">Try: is the backend healthy? or: draft me a render of a silver-haired girl</p>
        )}
        {messages.map((message) =>
          message.parts.map(
            (part, index) =>
              part.type === "text" &&
              part.text && (
                <Bubble key={`${message.id}-${index}`} role={message.role} text={part.text} />
              ),
          ),
        )}
        <ActivityFeed conversationId={sessionId} busy={busy} />
        {status === "error" && (
          <p className="error">{error?.message ?? "Something broke."} Check the server log.</p>
        )}
        <div ref={endRef} />
      </section>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          const text = input.trim();
          if (!text || busy) return;
          setInput("");
          // The first message of a fresh conversation doubles as its title.
          const first = messages.length === 0;
          void sendMessage({ text, metadata: first ? { title: text } : undefined });
          if (first) onFirstMessage(text);
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="ask the workbench, or describe a render"
          autoFocus
        />
        {busy ? (
          <button type="button" onClick={() => void stop()}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </>
  );
}

function App() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [history, setHistory] = useState<UIMessage[] | null>(null);
  const [pending, setPending] = useState(false);
  const [fault, setFault] = useState<string | null>(null);

  useEffect(() => {
    void fetch("/api/sessions")
      .then((response) => response.json())
      .then((fetched: SessionSummary[]) =>
        // Merge instead of replace: a chat started while this GET was in
        // flight is already in the list and must not be clobbered.
        setSessions((current) => {
          const seen = new Set((current ?? []).map((session) => session.id));
          return [...(current ?? []), ...fetched.filter((session) => !seen.has(session.id))];
        }),
      )
      .catch(() => setSessions((current) => current ?? []));
  }, []);

  const newChat = async () => {
    if (pending) return;
    setPending(true);
    try {
      const response = await fetch("/api/sessions", { method: "POST" });
      if (!response.ok) throw new Error(`sessions ${response.status}`);
      const session = (await response.json()) as SessionSummary;
      setSessions((current) => [session, ...(current ?? [])]);
      setHistory([]);
      setActive(session.id);
      setFault(null);
    } catch (err) {
      console.error("new chat failed", err);
      setFault("Couldn't create a session. Check the server log.");
    } finally {
      setPending(false);
    }
  };

  const openSession = async (id: string) => {
    if (pending || id === active) return;
    setPending(true);
    try {
      const response = await fetch(`/api/history?conversation=${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error(`history ${response.status}`);
      setHistory((await response.json()) as UIMessage[]);
      setActive(id);
      setFault(null);
    } catch (err) {
      console.error("open session failed", err);
      setFault("Couldn't open that conversation. Check the server log.");
    } finally {
      setPending(false);
    }
  };

  const retitle = (id: string, title: string) => {
    // Optimistic mirror of the title the bridge persists (same truncation).
    const short = truncate(title, TITLE_MAX);
    setSessions((current) =>
      (current ?? []).map((session) => (session.id === id ? { ...session, title: short } : session)),
    );
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>Z3Comfy Chat</h1>
          <p>Chat SDK × Claude Agent SDK</p>
        </header>
        <button className="new-chat" onClick={() => void newChat()} disabled={pending}>
          New chat
        </button>
        <nav>
          {sessions === null && <p className="hint">loading sessions...</p>}
          {sessions?.map((session) => (
            <button
              key={session.id}
              className={session.id === active ? "session active" : "session"}
              onClick={() => void openSession(session.id)}
              disabled={pending}
            >
              <span className="title">{session.title}</span>
              <span className="when">{new Date(session.created_at).toLocaleDateString()}</span>
            </button>
          ))}
        </nav>
        {fault && <p className="error">{fault}</p>}
        <p className="sidenote">
          The assistant can inspect the workbench freely, but a render job only runs when you
          confirm its proposal card. Conversations live in app-chat/data/sessions.json.
        </p>
      </aside>

      <main>
        {active && history ? (
          <Conversation
            key={active}
            sessionId={active}
            history={history}
            onFirstMessage={(title) => retitle(active, title)}
          />
        ) : (
          <section className="empty">
            <h2>Ask the workbench anything.</h2>
            <p>Queue state, gallery, diagnostics, prompt drafting -- renders only run on your confirm.</p>
            <button onClick={() => void newChat()} disabled={pending}>
              Start a chat
            </button>
          </section>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
