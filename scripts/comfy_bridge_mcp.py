#!/usr/bin/env python3
"""ComfyUI selection bridge: stdio MCP server ``loramake-comfy``.

Lets Claude read what the user has box-selected on the ComfyUI canvas, point back at
nodes, and patch/edit the graph, without describing nodes in prose. Counterpart of
``tools/comfy_bridge/extension/js/index.js`` (browser side). The two sides never talk
directly: both read and write small JSON files in ComfyUI's ``/api/userdata`` store under
``loramake/``, so there is no Mac-side daemon, no port, and no WebSocket.

Protocol (files under userdata ``loramake/``):

    selection.json      written by the page whenever the selection changes
    cmd/<id>.json       {id, ts, op, args, session} written here, polled by the page
    res/<id>.json       {id, ok, result | error, ts, tab} written by the page

Tools: ``comfy_selection``, ``comfy_map``, ``comfy_find``, ``comfy_focus``,
``comfy_patch``, ``comfy_edit``, ``comfy_net_status`` (seven; do not grow the list).

API graph exchange: ``comfy_map(api=True)`` exports the canvas's API-format
graph and writes it under ``outputs/_analysis/comfy_bridge/<ts>_api.json`` for
``scripts/comfy_node_profile.py --graph``; ``comfy_edit``'s ``load_api`` op does the
reverse, loading an API graph (inline ``graph`` or a ``path`` under
``outputs/_analysis/``) onto the canvas via the page's ``loadApiJson``.

Guardrails: never queues a prompt; ``patch``/``edit`` results carry the graph snapshot
taken before the change, which is written to ``outputs/_analysis/comfy_bridge/`` and can
be replayed with ``comfy_edit(ops=[{"op": "restore", "snapshot": "<path>"}])``.

The ComfyUI host always comes from ``loramake.comfy_endpoints.resolve_comfy_host`` (the
managed loopback ``127.0.0.1:18188`` in a Claude session); no host literal lives here.

Dev CLI (no MCP SDK needed):  python3 scripts/comfy_bridge_mcp.py call <tool> ['{json args}']
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake.comfy_endpoints import resolve_comfy_host  # noqa: E402

NS = "loramake"
USERDATA = "/api/userdata"
HTTP_TIMEOUT_S = 20.0
TRANSPORT_RETRIES = 1  # exactly one retry: the loopback forward flaps briefly and self-heals
RETRY_BACKOFF_S = 1.0
COMMAND_TIMEOUT_S = 15.0
MUTATION_TIMEOUT_S = 40.0
RESULT_POLL_S = 0.5
FIRST_POLL_DELAY_S = 0.4  # the page polls cmd/ every 300 ms; a GET before that is a wasted tunnel hop
STALE_FILE_S = 600.0
STALE_SWEEP_LIMIT = 20
MAX_RESPONSE_BYTES = 64 << 20  # a full graph snapshot can be a few MB; cap well above that


def _outputs_root() -> Path:
    """Repo outputs tree by default; LORAMAKE_OUTPUTS moves it (off-LAN sessions keep the
    cloned tree read-only and write snapshots/downloads into their own scratch root)."""
    raw = os.environ.get("LORAMAKE_OUTPUTS", "")
    root = Path(raw).expanduser() if raw else REPO / "outputs"
    return root if root.is_absolute() else REPO / root


SNAPSHOT_DIR = _outputs_root() / "_analysis" / "comfy_bridge"
PAGE_OFFLINE_HINT = (
    "no result before the timeout: the ComfyUI page running loramake-bridge is not open, "
    "the bridge is not loaded there, or the command is still running"
)
EDIT_OPS = ("add_node", "remove_node", "connect", "disconnect", "restore", "load_api")

# Patchable seams so tests run instantly with no server.
_urlopen = urllib.request.urlopen
_sleep = time.sleep
_now = time.time


def _defer(fn) -> None:
    """Run cleanup off the caller's path: a tunnel hop costs ~1 s and the caller has its answer."""
    threading.Thread(target=fn, daemon=True).start()


class BridgeError(RuntimeError):
    """Any failure the caller should see verbatim."""


def comfy_host(box: str | None = None) -> str:
    box = box or os.environ.get("LORAMAKE_COMFY_BOX", "boxa")
    explicit = os.environ.get("COMFY_BRIDGE_HOST") or os.environ.get("BOXA_HOST")
    return resolve_comfy_host(box, explicit=explicit)


class BridgeClient:
    """Thin HTTP client for the userdata file protocol."""

    def __init__(self, host: str, session: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.session = session or uuid.uuid4().hex[:8]
        self._seq = 0

    # -- transport -------------------------------------------------------

    def _file_url(self, rel: str) -> str:
        return f"{self.host}{USERDATA}/{urllib.parse.quote(f'{NS}/{rel}', safe='')}"

    def _dir_url(self, rel: str) -> str:
        return f"{self.host}{USERDATA}?dir={urllib.parse.quote(f'{NS}/{rel}', safe='')}"

    def _request(self, method: str, url: str, body: dict | None = None) -> tuple[int, bytes]:
        data = None
        headers = {}
        if url.lower().startswith("https://"):
            # Tunnel URL: Cloudflare's integrity check 403s (error 1010) bare python agents.
            from loramake.comfy import _CF_UA

            headers["User-Agent"] = _CF_UA
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        # Every call here is idempotent (a file GET/DELETE, or a POST that rewrites the same
        # file), so one retry after a transport flap of the loopback forward is safe.
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                with _urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                    return resp.status, resp.read(MAX_RESPONSE_BYTES)
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()
            except (urllib.error.URLError, OSError) as exc:
                if attempt >= TRANSPORT_RETRIES:
                    raise BridgeError(f"{method} {url}: {exc}") from exc
                _sleep(RETRY_BACKOFF_S)
        raise AssertionError("unreachable")

    def get_json(self, rel: str) -> dict | list | None:
        code, raw = self._request("GET", self._file_url(rel))
        if code == 404:
            return None
        if code != 200:
            raise BridgeError(f"GET {rel} -> HTTP {code}: {raw[:200]!r}")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError(f"GET {rel}: not JSON ({exc})") from exc

    def put_json(self, rel: str, obj: dict) -> None:
        code, raw = self._request("POST", self._file_url(rel), obj)
        if code != 200:
            raise BridgeError(f"POST {rel} -> HTTP {code}: {raw[:200]!r}")

    def delete(self, rel: str) -> None:
        code, raw = self._request("DELETE", self._file_url(rel))
        if code not in (200, 204, 404):
            raise BridgeError(f"DELETE {rel} -> HTTP {code}: {raw[:200]!r}")

    def list_dir(self, rel: str) -> list[str]:
        code, raw = self._request("GET", self._dir_url(rel))
        if code == 404:
            return []
        if code != 200:
            raise BridgeError(f"LIST {rel} -> HTTP {code}: {raw[:200]!r}")
        names = json.loads(raw.decode("utf-8"))
        return [n for n in names if isinstance(n, str)]

    # -- command protocol ------------------------------------------------

    def next_id(self) -> str:
        self._seq += 1
        return f"{self.session}-{self._seq:04d}"

    def send(self, op: str, args: dict | None = None, timeout: float = COMMAND_TIMEOUT_S) -> dict:
        cid = self.next_id()
        self.put_json(f"cmd/{cid}.json", {
            "id": cid, "ts": _now(), "op": op, "args": args or {}, "session": self.session,
        })
        deadline = _now() + timeout
        _sleep(FIRST_POLL_DELAY_S)
        while True:
            res = self.get_json(f"res/{cid}.json")
            if isinstance(res, dict):
                _defer(lambda: self._quiet_delete(f"res/{cid}.json"))
                if not res.get("ok"):
                    raise BridgeError(f"{op}: {res.get('error', 'unknown error from page')}")
                result = res.get("result")
                return result if isinstance(result, dict) else {"result": result}
            if _now() >= deadline:
                self._quiet_delete(f"cmd/{cid}.json")
                raise BridgeError(f"{op}: {PAGE_OFFLINE_HINT}")
            _sleep(RESULT_POLL_S)

    def _quiet_delete(self, rel: str) -> None:
        try:
            self.delete(rel)
        except BridgeError:
            pass

    def sweep_stale(self) -> int:
        """Drop cmd/res files older than STALE_FILE_S (left by dead sessions)."""
        removed = 0
        for sub in ("cmd", "res"):
            for name in self.list_dir(sub)[:STALE_SWEEP_LIMIT]:
                doc = self.get_json(f"{sub}/{name}")
                ts = doc.get("ts") if isinstance(doc, dict) else None
                if isinstance(ts, (int, float)) and _now() - ts > STALE_FILE_S:
                    self._quiet_delete(f"{sub}/{name}")
                    removed += 1
        return removed


# -- snapshots -----------------------------------------------------------------


def save_snapshot(graph: dict, tag: str, directory: Path | None = None) -> Path:
    directory = directory or SNAPSHOT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)[:40] or "snap"
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_tag}.json"
    path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    return path


def load_snapshot(ref: str, directory: Path | None = None) -> dict:
    """Resolve ``ref`` (file name or path) strictly inside the snapshot directory."""
    directory = directory or SNAPSHOT_DIR
    candidate = Path(ref)
    if not candidate.is_absolute():
        candidate = directory / candidate
    resolved = candidate.resolve()
    root = directory.resolve()
    if root != resolved and root not in resolved.parents:
        raise BridgeError(f"restore: snapshot must live under {root}, got {ref}")
    if not resolved.is_file():
        raise BridgeError(f"restore: snapshot not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _detach_snapshot(result: dict, tag: str) -> dict:
    snapshot = result.pop("snapshot", None)
    if isinstance(snapshot, dict):
        result["snapshot_path"] = str(save_snapshot(snapshot, tag))
    return result


def _resolve_analysis_path(ref: str) -> Path:
    """Resolve ``ref`` (repo-relative or absolute) strictly inside ``outputs/_analysis/``."""
    candidate = Path(ref)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    resolved = candidate.resolve()
    root = (_outputs_root() / "_analysis").resolve()
    if root != resolved and root not in resolved.parents:
        raise BridgeError(f"load_api: path must live under {root}, got {ref}")
    if not resolved.is_file():
        raise BridgeError(f"load_api: graph not found: {resolved}")
    return resolved


def _validate_api_graph(graph) -> dict:
    """API-format graph (the shape the prompt endpoint accepts): non-empty {nodeId: {class_type: str, inputs: dict}}."""
    if not isinstance(graph, dict) or not graph:
        raise BridgeError("load_api: graph must be a non-empty object")
    for nid, node in graph.items():
        if (not isinstance(node, dict)
                or not isinstance(node.get("class_type"), str)
                or not isinstance(node.get("inputs"), dict)):
            raise BridgeError(
                f"load_api: node {nid!r} needs a string class_type and object inputs")
    return graph


def _prepare_load_api(op: dict) -> dict:
    """load_api takes exactly one of ``graph`` / ``path``; the path never reaches the page."""
    has_graph = op.get("graph") is not None
    has_path = op.get("path") is not None
    if has_graph == has_path:
        raise BridgeError("edit: load_api needs exactly one of graph or path")
    out = {"op": "load_api"}
    if has_path:
        resolved = _resolve_analysis_path(str(op["path"]))
        try:
            out["graph"] = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BridgeError(f"load_api: {resolved} is not JSON ({exc})") from exc
        out["name"] = op.get("name") or resolved.stem
    else:
        out["graph"] = op["graph"]
        if op.get("name") is not None:
            out["name"] = op["name"]
    _validate_api_graph(out["graph"])
    return out


# -- tools ---------------------------------------------------------------------


def _id_list(values, what: str) -> list[int | str]:
    """Node ids are ints in classic litegraph and strings in newer front-ends; pass them through."""
    if not isinstance(values, (list, tuple)) or not values:
        raise BridgeError(f"{what}: expected a non-empty list of node ids")
    if not all(isinstance(v, (int, str)) and not isinstance(v, bool) for v in values):
        raise BridgeError(f"{what}: node ids must be integers or strings")
    return list(values)


def tool_selection(client: BridgeClient, fresh: bool = False) -> dict:
    if fresh:
        return client.send("selection")
    data = client.get_json("selection.json")
    if not isinstance(data, dict):
        raise BridgeError("selection.json not published yet: open ComfyUI with loramake-bridge loaded")
    data["age_s"] = round(_now() - float(data.get("ts", 0)), 1)
    return data


def tool_map(client: BridgeClient, q: str | None = None, api: bool = False) -> dict:
    if api:
        result = client.send("api_prompt")
        output = result.get("output")
        if not isinstance(output, dict):
            raise BridgeError("api_prompt: page returned no graph output")
        path = save_snapshot(output, "api")
        return {
            "api_path": str(path),
            "node_count": result.get("node_count"),
            "workflow_nodes": result.get("workflow_nodes"),
        }
    return client.send("map", {"q": q or None})


def tool_find(client: BridgeClient, q: str) -> dict:
    if not isinstance(q, str) or not q.strip():
        raise BridgeError("find: q must be a non-empty string")
    return client.send("map", {"q": q.strip()})


def tool_focus(client: BridgeClient, ids) -> dict:
    return client.send("focus", {"ids": _id_list(ids, "focus")})


def tool_patch(client: BridgeClient, ops, tag: str = "patch") -> dict:
    if not isinstance(ops, list) or not ops:
        raise BridgeError("patch: ops must be a non-empty list of {id, widget, value}")
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or not {"id", "widget", "value"} <= set(op):
            raise BridgeError(f"patch: op[{i}] needs id, widget and value")
    result = client.send("patch", {"ops": ops}, timeout=MUTATION_TIMEOUT_S)
    return _detach_snapshot(result, tag)


def tool_edit(client: BridgeClient, ops, tag: str = "edit") -> dict:
    if not isinstance(ops, list) or not ops:
        raise BridgeError(f"edit: ops must be a non-empty list; op in {'/'.join(EDIT_OPS)}")
    prepared = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in EDIT_OPS:
            raise BridgeError(f"edit: op[{i}] must set op to one of {'/'.join(EDIT_OPS)}")
        op = dict(op)
        if op["op"] == "restore":
            op["graph"] = load_snapshot(str(op.pop("snapshot", "")))
        elif op["op"] == "load_api":
            op = _prepare_load_api(op)
        prepared.append(op)
    result = client.send("edit", {"ops": prepared}, timeout=MUTATION_TIMEOUT_S)
    return _detach_snapshot(result, tag)


def tool_net_status(ping: bool = False) -> dict:
    from scripts import comfy_net_status

    return comfy_net_status.collect(ping=ping)


TOOLS = {
    "selection": tool_selection,
    "map": tool_map,
    "find": tool_find,
    "focus": tool_focus,
    "patch": tool_patch,
    "edit": tool_edit,
}


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


# -- entry points ----------------------------------------------------------------


INSTRUCTIONS = (
    "ComfyUI canvas bridge. comfy_selection returns the nodes the user has selected "
    "on the canvas (ids, types, titles, widget values, links): call it when the user "
    "says 'this node', 'the selected sampler', or points at something. comfy_map "
    "and comfy_find locate nodes by type or title; comfy_focus selects and centres "
    "them so the user can see what you mean. comfy_patch changes widget values and "
    "comfy_edit adds/removes nodes or links; both snapshot first and return what was "
    "actually applied. Never queue a render here: the user presses Queue themselves."
)


def new_client() -> BridgeClient:
    """Client for serving. The stale-file sweep runs off-thread: a slow loopback forward
    must not hold the server's startup handshake past the agent's startup timeout."""
    client = BridgeClient(comfy_host())

    def sweep() -> None:
        try:
            client.sweep_stale()
        except BridgeError:
            pass  # server unreachable at start; every tool reports that on its own

    _defer(sweep)
    return client


def register(mcp, client: BridgeClient) -> None:
    """Attach the seven canvas tools to an MCPServer (shared with scripts/loramake_mcp.py)."""

    @mcp.tool(name="comfy_selection", structured_output=False)
    def comfy_selection(fresh: bool = False) -> str:
        """Nodes currently selected on the ComfyUI canvas. fresh=True re-reads via the page."""
        return _dump(tool_selection(client, fresh=fresh))

    @mcp.tool(name="comfy_map", structured_output=False)
    def comfy_map(q: str | None = None, api: bool = False) -> str:
        """Whole-graph overview: node count, type histogram, id/type/title table. Optional filter q.
        api=True instead exports the API-format graph to a file and returns its
        path — feed it to scripts/comfy_node_profile.py --graph."""
        return _dump(tool_map(client, q, api=api))

    @mcp.tool(name="comfy_find", structured_output=False)
    def comfy_find(q: str) -> str:
        """Nodes whose type or title contains q (case-insensitive)."""
        return _dump(tool_find(client, q))

    @mcp.tool(name="comfy_focus", structured_output=False)
    def comfy_focus(ids: list[int | str]) -> str:
        """Select the given node ids (as returned by comfy_map/comfy_selection) and centre the view on them."""
        return _dump(tool_focus(client, ids))

    @mcp.tool(name="comfy_patch", structured_output=False)
    def comfy_patch(ops: list[dict], tag: str = "patch") -> str:
        """Set widget values: ops=[{id, widget, value}, ...]. Snapshots first; returns before/after."""
        return _dump(tool_patch(client, ops, tag))

    @mcp.tool(name="comfy_edit", structured_output=False)
    def comfy_edit(ops: list[dict], tag: str = "edit") -> str:
        """Structural edits: add_node{type,pos,title,widgets} remove_node{id} connect{from:[id,slot],to:[id,slot]} disconnect{to} restore{snapshot} load_api{graph|path,name}."""
        return _dump(tool_edit(client, ops, tag))

    @mcp.tool(name="comfy_net_status", structured_output=False)
    def comfy_net_status(ping: bool = False) -> str:
        """Health and latency of every ComfyUI channel (BOXA/BOXB loopback, bridge page). ping=True round-trips the page."""
        return _dump(tool_net_status(ping=ping))


def main() -> None:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("loramake-comfy", instructions=INSTRUCTIONS)
    register(mcp, new_client())
    mcp.run(transport="stdio")


def cli(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "call":
        name = argv[1]
        args = json.loads(argv[2]) if len(argv) > 2 else {}
        if name == "net_status":
            print(_dump(tool_net_status(**args)))
            return 0
        if name not in TOOLS:
            print(f"unknown tool {name}; one of {', '.join(TOOLS)}, net_status", file=sys.stderr)
            return 2
        try:
            print(_dump(TOOLS[name](BridgeClient(comfy_host()), **args)))
        except BridgeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
