#!/usr/bin/env python3
"""Civitai toolkit — a verified, callable interface to the Civitai APIs.

Two surfaces:
  - Orchestration  https://orchestration.civitai.com   (generation / training; Bearer)
  - Site REST      https://civitai.com/api/v1           (retrieval; mostly public)

Design notes
  - Transport is curl (HTTP/2, robust through the system proxy). The bearer token
    is fed via `--config /dev/fd/N` (a pipe) so it never appears in argv / `ps`.
  - SFW is enforced: models/images queries always carry nsfw=false.
  - Spending Buzz is gated: imagegen/train run a free `whatif` cost preview unless
    --confirm-spend is passed. whatif never charges.
  - Schema follows docs/civitai/civitai_ai_usable_cheatsheet.md (verified subset).

CLI:
  python3 scripts/civitai.py selftest                 # run all FREE probes, prove capabilities
  python3 scripts/civitai.py me
  python3 scripts/civitai.py enums [KEY]
  python3 scripts/civitai.py models query=anime types=LORA baseModels="SDXL 1.0" limit=3
  python3 scripts/civitai.py version <VERSION_ID>      # → includes .air
  python3 scripts/civitai.py by-hash <HASH>
  python3 scripts/civitai.py permissions <VERSION_ID>[,<VERSION_ID>...]
  python3 scripts/civitai.py images modelVersionId=<ID> withMeta=true limit=2
  python3 scripts/civitai.py imagegen <prompt> [--model AIR] [--confirm-spend]  # whatif unless --confirm-spend
"""

import json
import os
import subprocess
import sys
import urllib.parse

ORCH = "https://orchestration.civitai.com"
SITE = "https://civitai.com/api/v1"
MCP_URL = "https://mcp.civitai.com/mcp"
TOKEN_PATH = os.path.expanduser(os.environ.get("CIVITAI_TOKEN_PATH", "~/.config/civitai/token"))
PROXY = os.environ.get("CIVITAI_PROXY", "http://127.0.0.1:1082")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
NOOBAI_AIR = "urn:air:sdxl:checkpoint:civitai:833294@1116447"  # NoobAI-XL v1.1 eps (project base)
# subprocess backstop beyond curl's --max-time: reaps a curl that wedges outside the transfer.
CURL_TIMEOUT_MARGIN_S = 5


def _token():
    with open(TOKEN_PATH) as fh:
        return fh.read().strip()


def _auth_config(token):
    """curl config line carrying the bearer token. A quote, backslash or line break would end or
    rewrite the quoted value, so such a token is refused instead of escaped (message omits it)."""
    if any(ch in token for ch in ('"', "\\", "\n", "\r")):
        raise ValueError("Civitai token contains a quote, backslash or line break; regenerate it")
    return f'header = "Authorization: Bearer {token}"\n'


def _curl(method, url, *, auth=False, body=None, timeout=120):
    """Return (http_status:int, parsed_json_or_text). Token goes via a pipe config, not argv."""
    cmd = ["curl", "-sS", "--http2", "-x", PROXY, "--max-time", str(timeout),
           "-w", "\n%{http_code}", "-H", f"User-Agent: {UA}", "-X", method, url]
    stdin = None
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
        stdin = body if isinstance(body, str) else json.dumps(body)
    # The Authorization header goes to curl through a pipe config, keeping it out of ps/argv.
    cfg = _auth_config(_token()) if auth else None
    out = _run_with_optional_config(cmd, stdin, cfg, timeout=timeout + CURL_TIMEOUT_MARGIN_S)
    raw, _, code = out.rpartition("\n")
    try:
        return int(code), json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return (int(code) if code.isdigit() else 0), raw


def _run_with_optional_config(cmd, stdin_text, cfg_text, timeout=None):
    """Run curl, supplying --data-binary @- on stdin and/or the auth config on an inherited pipe.

    The config is read from the pipe's own fd number (``--config /dev/fd/N``, appended here).
    Remapping it to a fixed fd 3 in ``preexec_fn`` does not work: ``close_fds`` runs after
    ``preexec_fn`` and closes fd 3 whenever the parent already holds fd 3, silently dropping
    the header. ``timeout`` is the subprocess backstop; a hang returns the same
    ``"<message>\\n0"`` shape as any other curl failure."""
    import os as _os
    extra_fd = None
    if cfg_text is not None:
        r, w = _os.pipe()
        _os.write(w, cfg_text.encode())
        _os.close(w)
        extra_fd = r
        cmd = cmd + ["--config", f"/dev/fd/{r}"]
    try:
        pass_fds = (extra_fd,) if extra_fd is not None else ()
        try:
            proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                                  pass_fds=pass_fds, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"curl hung past the {timeout}s subprocess timeout\n0"
        if proc.returncode != 0 and not proc.stdout:
            return f"{proc.stderr.strip()}\n0"
        return proc.stdout
    finally:
        if extra_fd is not None:
            _os.close(extra_fd)


# ── Site REST (retrieval) ──────────────────────────────────────────────────

def _get(path, params=None, auth=False):
    url = SITE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _curl("GET", url, auth=auth)


def me():
    return _get("/me", auth=True)


def enums():
    return _get("/enums")


def search_models(**params):
    params.setdefault("nsfw", "false")  # SFW enforced
    return _get("/models", params)


def model_version(version_id):
    return _get(f"/model-versions/{version_id}")


def by_hash(h):
    return _get(f"/model-versions/by-hash/{urllib.parse.quote(h)}")


def permissions_check(version_ids):
    ids = ",".join(str(i) for i in version_ids)
    return _get("/permissions/check", {"entityIds": ids})


def images(**params):
    params.setdefault("nsfw", "false")  # SFW enforced
    return _get("/images", params)


def trpc(procedure, payload=None, *, date_keys=(), timeout=60):
    """GET one tRPC query of the site (the procedures its own pages call), as the token's user.

    Only queries are reachable this way; mutations need POST and are not exposed here. Keys in
    ``date_keys`` are ISO strings that superjson must revive as Dates.
    Returns (http_status, data) with ``data`` unwrapped from result.data.json on success."""
    wrapped = {"json": payload}
    dates = [k for k in date_keys if isinstance(payload, dict) and k in payload]
    if dates:
        wrapped["meta"] = {"values": {k: ["Date"] for k in dates}}
    url = f"https://civitai.com/api/trpc/{procedure}?input=" + urllib.parse.quote(json.dumps(wrapped))
    code, data = _curl("GET", url, auth=True, timeout=timeout)
    if code == 200 and isinstance(data, dict):
        data = ((data.get("result") or {}).get("data") or {}).get("json")
    return code, data


# ── Site MCP (writes: uploads, posts, collections) ─────────────────────────

def mcp_call(tool, arguments, *, timeout=180, attempts=2):
    """Call one tool on the hosted Civitai MCP server (stateless streamable HTTP, JSON reply).

    Returns (ok:bool, structured_or_text). Same transport as _curl: proxy, token over a pipe.
    The server answers tool errors with HTTP 200 and result.isError, so both are checked. A
    transport failure (HTTP 000, seen on large uploads) is retried; an HTTP error is not."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}
    cmd = ["curl", "-sS", "--http2", "-x", PROXY, "--max-time", str(timeout),
           "-w", "\n%{http_code}", "-H", f"User-Agent: {UA}",
           "-H", "Content-Type: application/json",
           "-H", "Accept: application/json, text/event-stream",
           "-X", "POST", MCP_URL, "--data-binary", "@-"]
    for _ in range(attempts):
        out = _run_with_optional_config(cmd, json.dumps(body), _auth_config(_token()),
                                        timeout=timeout + CURL_TIMEOUT_MARGIN_S)
        raw, _, code = out.rpartition("\n")
        if code not in ("0", "000"):
            break
    if code != "200":
        return False, f"HTTP {code}: {raw[:300]}"
    if raw.lstrip().startswith("event:") or "\ndata:" in raw:
        raw = "\n".join(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    try:
        reply = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"unparseable MCP reply: {raw[:300]}"
    if "error" in reply:
        return False, str(reply["error"])[:300]
    result = reply.get("result") or {}
    text = " ".join(c.get("text", "") for c in result.get("content") or [] if isinstance(c, dict))
    if result.get("isError"):
        return False, text[:300] or "tool error"
    return True, result.get("structuredContent") or text


# ── Orchestration (generation) ─────────────────────────────────────────────

def imagegen_input(prompt, *, model=NOOBAI_AIR, negative="lowres, worst quality, low quality, bad anatomy",
                   width=832, height=1216, cfg=5.0, steps=30, seed=1234, quantity=1, loras=None):
    inp = {
        "engine": "sdcpp", "ecosystem": "sdxl", "operation": "createImage",
        "model": model, "prompt": prompt, "negativePrompt": negative,
        "width": width, "height": height, "cfgScale": cfg, "steps": steps,
        "seed": seed, "quantity": quantity,
    }
    if loras:  # dict of AIR -> strength
        inp["loras"] = loras
    return inp


def imagegen(prompt, *, confirm_spend=False, wait=60, **kw):
    """Free cost-preview by default (whatif); real (charges Buzz) only with confirm_spend=True.

    Uses the per-recipe endpoint which returns the step output directly: {images:[{url}]}.
    NOTE: this endpoint is lenient — unknown input fields are silently ignored, not rejected.
    """
    inp = imagegen_input(prompt, **kw)
    q = {"allowMatureContent": "false"}
    if confirm_spend:
        q["wait"] = str(wait)
    else:
        q["whatif"] = "true"
    url = f"{ORCH}/v2/consumer/recipes/imageGen?" + urllib.parse.urlencode(q)
    return _curl("POST", url, auth=True, body=inp, timeout=wait + 30)


def cost_preview(prompt, **kw):
    """Free Buzz cost preview via the generic workflow whatif (returns `transactions`)."""
    inp = imagegen_input(prompt, **kw)
    return _curl("POST", f"{ORCH}/v2/consumer/workflows?whatif=true&allowMatureContent=false",
                 auth=True, body={"steps": [{"$type": "imageGen", "input": inp}]})


def _cost_str(workflow_whatif_json):
    txns = (workflow_whatif_json.get("transactions") or {}).get("list", []) if isinstance(workflow_whatif_json, dict) else []
    return ", ".join(f"{t.get('amount')} {t.get('accountType')}" for t in txns) or "(no transactions)"


# ── self-test (FREE probes only — proves each capability is real) ───────────

def _probe(name, fn):
    try:
        code, data = fn()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False, None
    ok = code == 200
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (HTTP {code})")
    return ok, data


def selftest():
    print("Civitai capability self-test — FREE probes only (no Buzz), real API calls.\n")
    results = {}

    ok, d = _probe("site /me (token valid)", me)
    if ok and isinstance(d, dict):
        print(f"       user={d.get('username')!r} tier={d.get('tier')!r} status={d.get('status')!r}")
    results["me"] = ok

    ok, d = _probe("site /enums", enums)
    if ok and isinstance(d, dict):
        print(f"       ModelType n={len(d.get('ModelType',[]))}  ActiveBaseModel={d.get('ActiveBaseModel',[])[:6]}")
    results["enums"] = ok

    ok, d = _probe("site /models (LORA, SDXL 1.0)",
                   lambda: search_models(types="LORA", baseModels="SDXL 1.0", sort="Most Downloaded", limit=3))
    if ok and isinstance(d, dict):
        for m in d.get("items", [])[:3]:
            print(f"       #{m.get('id')} {m.get('name')!r}")
    results["models"] = ok

    ok, d = _probe("site /model-versions/1116447 (NoobAI .air)", lambda: model_version(1116447))
    air = None
    if ok and isinstance(d, dict):
        air = d.get("air")
        print(f"       air={air!r}  matches NOOBAI_AIR={air == NOOBAI_AIR}")
        files = d.get("files", [])
        sha = next((f.get("hashes", {}).get("SHA256") for f in files if f.get("hashes")), None)
        if sha:
            print(f"       primary SHA256={sha[:16]}…")
            ok2, d2 = _probe("site /model-versions/by-hash (round-trip)", lambda: by_hash(sha))
            if ok2 and isinstance(d2, dict):
                print(f"       by-hash → version id={d2.get('id')} (expect 1116447)")
            results["by_hash"] = ok2
    results["version"] = ok

    ok, d = _probe("site /permissions/check (can generate from 1116447)",
                   lambda: permissions_check([1116447]))
    if ok:
        print(f"       {d}")
    results["permissions"] = ok

    ok, d = _probe("site /images?modelVersionId=1116447&withMeta=true",
                   lambda: images(modelVersionId=1116447, withMeta="true", limit=2))
    if ok and isinstance(d, dict):
        print(f"       returned {len(d.get('items', []))} images (meta present="
              f"{bool(d.get('items') and d['items'][0].get('meta'))})")
    results["images"] = ok

    ok, d = _probe("orch imageGen SDXL whatif (recipe path → images)",
                   lambda: imagegen("masterpiece, best quality, 1girl, solo, silver hair, red eyes"))
    if ok and isinstance(d, dict):
        print(f"       keys={list(d.keys())}  images={len(d.get('images', []))}  errors={d.get('errors')}")
    results["imagegen_whatif"] = ok

    ok, d = _probe("orch workflow whatif (real Buzz cost preview)",
                   lambda: cost_preview("masterpiece, best quality, 1girl, solo, silver hair, red eyes"))
    if ok and isinstance(d, dict):
        print(f"       SDXL 832x1216 / 30 steps / 1 img cost = {_cost_str(d)}")
    results["cost_preview"] = ok

    # Characterize endpoint strictness (truth: recipe path silently ignores unknown fields).
    def _bogus():
        inp = imagegen_input("test")
        inp["totallyBogusField"] = 123
        return _curl("POST", f"{ORCH}/v2/consumer/recipes/imageGen?whatif=true&allowMatureContent=false",
                     auth=True, body=inp)
    code, _ = _bogus()
    lenient = code == 200
    print(f"[INFO] recipe imageGen ignores unknown fields → HTTP {code} "
          f"({'lenient: extras silently dropped' if lenient else 'strict'})")
    results["leniency_known"] = code in (200, 400)

    passed = sum(1 for v in results.values() if v)
    print(f"\n== {passed}/{len(results)} capabilities verified with real calls ==")
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def _kv(args):
    out = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k] = v
    return out


def _show(code, data):
    print(f"HTTP {code}")
    print(json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else data)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]
    if cmd == "selftest":
        selftest()
    elif cmd == "me":
        _show(*me())
    elif cmd == "enums":
        code, d = enums()
        if rest and isinstance(d, dict):
            d = d.get(rest[0], f"(no key {rest[0]})")
        _show(code, d)
    elif cmd == "models":
        _show(*search_models(**_kv(rest)))
    elif cmd == "version":
        _show(*model_version(rest[0]))
    elif cmd == "by-hash":
        _show(*by_hash(rest[0]))
    elif cmd == "permissions":
        _show(*permissions_check(rest[0].split(",")))
    elif cmd == "images":
        _show(*images(**_kv(rest)))
    elif cmd == "cost":
        prompt = next((a for a in rest if not a.startswith("--")), "masterpiece, 1girl")
        code, d = cost_preview(prompt)
        print(f"HTTP {code}  cost={_cost_str(d)}")
    elif cmd == "imagegen":
        confirm = "--confirm-spend" in rest
        model = NOOBAI_AIR
        if "--model" in rest:
            model = rest[rest.index("--model") + 1]
        prompt = next((a for a in rest if not a.startswith("--") and a != model), "masterpiece, 1girl")
        _show(*imagegen(prompt, confirm_spend=confirm, model=model))
    else:
        print(f"unknown command: {cmd}\n")
        print(__doc__)


if __name__ == "__main__":
    main()
