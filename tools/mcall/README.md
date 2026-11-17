# mcall — unified headless model-call dispatcher

One entry point for calling external models through their native CLI binaries
(codex / grok / hermes), with flags tuned for fast non-interactive single calls.
Replaces ad-hoc `codex exec ... < /dev/null` incantations.

## Why

- Each backend CLI has different syntax, env quirks, and startup pitfalls
  (codex hangs on inherited stdin; grok needs proxy env; hermes needs a clean
  `PYTHONPATH`). mcall bakes all of that in.
- `codex exec` with the full user config loads MCP servers and instruction
  files a single model call never uses. mcall defaults to
  `--ignore-user-config --ephemeral`, which is measurably faster and cheaper.
- Backend binaries are invoked directly (the vendored codex rust binary, the
  downloaded grok rust binary), skipping node/shell shims.

## Usage

```bash
mcall <alias> "PROMPT" [options] [-- extra backend args]
mcall <alias> --prompt-file task.md
mcall list
```

| Alias    | Backend | Provider    | Model               |
|----------|---------|-------------|---------------------|
| luna     | codex   | —           | gpt-5.6-luna        |
| terra    | codex   | —           | gpt-5.6-terra       |
| sol      | codex   | —           | gpt-5.6-sol         |
| spark    | codex   | —           | gpt-5.3-codex-spark |
| codex    | codex   | —           | requires `-m` (`-pro` variants exist, e.g. gpt-5.6-luna-pro) |
| grok     | grok    | —           | backend default (grok-4.5 as of 2026-07) |
| deepseek | codewhale | (first-party API) | deepseek-v4-pro default; `-m deepseek-v4-flash` |
| kimi     | hermes  | huggingface | moonshotai/Kimi-K2.6 |
| glm      | hermes  | huggingface | zai-org/GLM-5.2     |
| hf       | hermes  | huggingface | requires `-m` (Qwen3.6, gpt-oss, MiniMax-M3, DeepSeek-V4, ...) |
| copilot  | hermes  | copilot     | requires `-m` (gpt-5.4, claude-sonnet-4.6, gemini-3.1-pro, ...) |
| hermes   | hermes  | (profile default) | novelart profile default (deepseek-v4-pro) |

deepseek goes through codewhale's native `codewhale-tui` binary with the
first-party DeepSeek API key (`codewhale auth status`) — by far the fastest
call in the fleet.

The hermes-backed aliases run under the `novelart` profile by default (it
holds the full credential set; override with `--hermes-profile` or
`MCALL_HERMES_PROFILE`) and route through hermes' pooled credentials
(`hermes auth list`); the model catalog per provider is cached in
`~/.hermes/provider_models_cache.json`.

Options: `-f/--prompt-file`, `-m/--model` (override), `-P/--provider`
(hermes: override the credential provider), `--hermes-profile P` (hermes:
profile, default novelart), `-o/--output FILE` (codex: write final message to
file), `--timeout SECS` (default 900), `--sandbox MODE` (codex, default
read-only), `--full-config` (codex: keep `~/.codex/config.toml`, still drops
MCP servers). Everything after `--` is passed to the backend CLI verbatim.

Exit codes: backend's own code; 124 on timeout; 2 on usage errors.

## Measured (2026-07-18, "Reply with exactly: ok")

| Path | Wall time | Tokens |
|------|-----------|--------|
| `codex exec` spark, full config (baseline) | 15.8 s | 15.7k |
| `mcall spark` (lean default) | 10.4 s | 11.2k |
| `mcall luna` | 10.2 s | 8.4k |
| `mcall grok` (grok-4.5) | 14.4 s | n/a |
| `mcall deepseek` (codewhale native, first-party API) | 3.3 s | n/a |
| deepseek-v4-pro via hermes (for comparison) | 119 s | n/a |
| `mcall hermes` (novelart profile) | 103 s | n/a |

hermes one-shots (`-z`) spin up the full hermes agent, so expect ~2 min even
for trivial prompts (lean flags like `--ignore-user-config` were within noise:
102 s vs 119 s). Use the hermes-backed aliases when you specifically want
those models, not for latency.

Startup overhead of the launchers themselves (for reference): grok rust binary
9 ms, codex rust binary 23 ms, codex node shim 45 ms, hermes python venv
112 ms. The dominant cost is always backend session bootstrap + model latency,
which is why the lean codex config is the main win.

## Behavior notes

- stdin is always `/dev/null` — `codex exec` reads stdin when piped and hangs
  forever under background callers.
- The codex lean default drops `model_instructions_file`; use `--full-config`
  when the call should see the user's codex instructions.
- spark rejects the `reasoning.summary` API parameter; mcall injects
  `-c model_reasoning_summary=none` for it automatically.
- grok gets the same proxy env as the `grok-proxy` wrapper
  (`GROK_PROXY_URL` / `GROK_NO_PROXY` override the defaults).
- Binary resolution order: `MCALL_{CODEX,GROK,HERMES}_BIN` env override →
  known install location → wrapper fallback.

## Build / install

```bash
cargo build --release --manifest-path tools/mcall/Cargo.toml
install -m 755 tools/mcall/target/release/mcall ~/.local/bin/mcall
```

Zero dependencies; builds in a few seconds.

## Known issues (2026-07-18)

- The hermes *default* profile's xai-oauth token is expired (`xAI token
  refresh failed`); mcall sidesteps this by defaulting to the `novelart`
  profile, whose credential set is complete and whose default model is
  deepseek-v4-pro (verified working).
- The huggingface free pool returned HTTP 402 (monthly included credits
  depleted) — account-level, reproduced under the novelart profile too;
  kimi/glm/hf work again when the quota resets or the account upgrades.
- The copilot provider route is wired but was not live-tested.
- The former `grok-build` / `grok-composer-2.5-fast` model ids no longer exist
  in grok CLI 0.2.101; the logged-in account exposes only `grok-4.5`
  (`grok-build-0.1` lives on in opencode's xai provider, not in the grok CLI).
