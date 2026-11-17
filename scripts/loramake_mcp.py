#!/usr/bin/env python3
"""Unified stdio MCP server ``loramake``: every ComfyUI surface an agent needs, one process.

Tool groups (22 tools):

    generate  studio_generate / studio_job / studio_jobs / media_ingest
              scripts/loramake_image_mcp.py (Studio /api/jobs queue; off-LAN sessions ride
              the same queue with the channel scope, job-id media refs and webp previews)
              image_generate / image_edit (Qwen-Image line, natural language, any image file)
    canvas    comfy_selection / comfy_map / comfy_find / comfy_focus / comfy_patch / comfy_edit /
              comfy_net_status                    scripts/comfy_bridge_mcp.py (browser selection bridge)
    raw       comfy_status / comfy_models / comfy_nodes / comfy_upload / comfy_run / comfy_history /
              comfy_fetch / comfy_cancel / comfy_free   this file, thin over loramake/comfy.py

Raw tools talk to a box's ComfyUI HTTP API. The host always comes from
``loramake.comfy_endpoints.resolve_comfy_host`` (managed loopback under Claude, LAN elsewhere,
``BOXA_HOST``/``BOXB_HOST`` env overrides — a tunnel URL lights the same tools up off-LAN).
``comfy_run`` refuses a busy box unless ``force=True`` and frees VRAM afterwards: BOXA is a shared
gaming box. Run outputs land under ``$LORAMAKE_OUTPUTS/_analysis/comfy_runs/``
(default ``outputs/_analysis/comfy_runs/``).

Every agent points at the same launcher, ``scripts/loramake-mcp`` (uv, project venv, ``agent``
extra). Agents without MCP support (pi) use the CLI form, which needs no MCP client:

    scripts/loramake-mcp tools
    scripts/loramake-mcp call comfy_status
    scripts/loramake-mcp call comfy_run '{"path": "outputs/_analysis/comfy_bridge/x_api.json"}'
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import comfy_bridge_mcp as bridge  # noqa: E402
import loramake_image_mcp as image  # noqa: E402
from comfy_bridge_mcp import BridgeError, comfy_host  # noqa: E402

SERVER_NAME = "loramake"
HTTP_TIMEOUT_S = 20.0
RUN_TIMEOUT_S = 600.0
MAX_JSON_BYTES = 64 << 20  # a full /object_info is a few MB
MAX_FETCH_BYTES = 256 << 20
LIST_CAP = 200
OBJECT_INFO_TTL_S = 300.0
def _outputs_root() -> Path:
    """Repo outputs tree by default; LORAMAKE_OUTPUTS moves it (cloud sessions keep the
    cloned tree read-only and download into their own scratch root)."""
    raw = os.environ.get("LORAMAKE_OUTPUTS", "")
    root = Path(raw).expanduser() if raw else REPO / "outputs"
    return root if root.is_absolute() else REPO / root


RUNS_DIR = _outputs_root() / "_analysis" / "comfy_runs"
MODEL_KINDS = {
    "checkpoint": ("CheckpointLoaderSimple", "ckpt_name"),
    "lora": ("LoraLoader", "lora_name"),
    "vae": ("VAELoader", "vae_name"),
    "controlnet": ("ControlNetLoader", "control_net_name"),
    "upscale": ("UpscaleModelLoader", "model_name"),
    "clip_vision": ("CLIPVisionLoader", "clip_name"),
    "unet": ("UNETLoader", "unet_name"),
}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

_urlopen = urllib.request.urlopen  # patchable seam
_object_info_cache: dict[str, tuple[float, dict]] = {}

INSTRUCTIONS = (
    "loramake ComfyUI server for agents, three groups. "
    "GENERATE: studio_generate is the product path (Studio queue, tiers, LoRAs, region repaint, "
    "rescue); use it for any image the user wants to keep; before any fresh prompt, run "
    "scripts/deai_lint.py prompt (image-deslop skill) until it reports zero errors; "
    "studio_jobs queries job history; "
    "media_ingest uploads a local PNG/JPEG and returns a med_ id for source_media_id / ref_media_id. "
    "image_generate / image_edit are the "
    "Qwen-Image line: natural-language prompts, and edits of any local image file or earlier result "
    "(size and format handled for you). "
    "studio_generate(provider=\"civitai\") renders on Civitai's cloud (checkpoint alias or AIR plus "
    "LoRA AIRs) under a standing Buzz budget; add quote=true for a free price check. "
    "CANVAS: comfy_selection / comfy_map / comfy_find / comfy_focus / comfy_patch / comfy_edit read "
    "and change the graph the user has open in the ComfyUI browser page; never queue from there. "
    "RAW: comfy_status (queue + VRAM, call first), comfy_models and comfy_nodes (what the box can "
    "load and which node types exist), comfy_upload (image into input/), comfy_run (submit an API "
    "graph, wait, download outputs; refuses a busy box unless force=true; frees VRAM after), "
    "comfy_history / comfy_fetch (inspect and download past outputs), comfy_cancel, comfy_free. "
    "Run one smoke image before any batch. box is boxa (default) or boxb. "
    "Off-LAN (an https Studio API), the generate group rides the tunnel under the session "
    "channel; canvas/raw tools need a reachable box host (BOXA_HOST / COMFY_BRIDGE_HOST env, "
    "e.g. a tunnel URL) and report a transport error until one exists."
)


# -- transport -------------------------------------------------------------------


def _box_request(url: str):
    """Tunnel (https) URLs get a browser-family UA: Cloudflare's integrity check 403s
    (error 1010) bare python agents. LAN URLs keep the plain string form so patched
    ``_urlopen`` seams observe no change."""
    if url.lower().startswith("https://"):
        return urllib.request.Request(url, headers={"User-Agent": image._CF_UA})
    return url


def _get_json(host: str, path: str, timeout: float = HTTP_TIMEOUT_S) -> dict:
    url = f"{host.rstrip('/')}{path}"
    try:
        with _urlopen(_box_request(url), timeout=timeout) as resp:
            raw = resp.read(MAX_JSON_BYTES)
    except urllib.error.HTTPError as exc:
        raise BridgeError(f"GET {path} -> HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BridgeError(f"GET {url}: {exc}") from exc
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"GET {path}: not JSON ({exc})") from exc
    if not isinstance(body, dict):
        raise BridgeError(f"GET {path}: expected object, got {type(body).__name__}")
    return body


def _box(box: str | None) -> str:
    return box or bridge.os.environ.get("LORAMAKE_COMFY_BOX", "boxa")


def _safe(tag: str) -> str:
    return _SAFE.sub("_", tag).strip("._")[:40] or "run"


def _unique_path(path: Path) -> Path:
    """Return ``path``, or ``stem_2`` / ``stem_3``… if it already exists.

    Same convention as ``boxa_quality_gen._unique_dest``: the first occupant keeps the
    historical name; later claims get a numeric suffix instead of clobbering.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = path.with_name(f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _as_prompt_id(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _entry_prompt_id(entry: object) -> str | None:
    """Prompt id from one /queue item: classic tuple, dict, or numbered-dict tuple."""
    if isinstance(entry, str):
        return _as_prompt_id(entry)
    if isinstance(entry, dict):
        for key in ("prompt_id", "promptId", "1"):
            pid = _as_prompt_id(entry.get(key))
            if pid:
                return pid
        return None
    if isinstance(entry, (list, tuple)) and len(entry) > 1:
        return _as_prompt_id(entry[1])
    return None


def _queue_entries(raw: object) -> list:
    """Running/pending field as a list of items.

    ComfyUI's /queue is normally two lists of ``[number, prompt_id, prompt, extra, outs]``.
    Frontends and some builds also emit dict items (``{"prompt_id": ...}``), a numbered
    dict of those items (``{"0": [...], "1": {...}}``), or a single dict standing in for
    the list. Any of those means work is on the box.
    """
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, dict):
        if "prompt_id" in raw or "promptId" in raw:
            return [raw]
        return list(raw.values())
    if raw:
        return [raw]
    return []


def _queue_ids(queue: dict, key: str) -> list[str]:
    ids: list[str] = []
    for entry in _queue_entries(queue.get(key)):
        pid = _entry_prompt_id(entry)
        if pid:
            ids.append(pid)
    return ids


def _queue_has_work(queue: dict) -> bool:
    """True when running or pending contains any item, even one with no parseable id."""
    return bool(_queue_entries(queue.get("queue_running")) or _queue_entries(queue.get("queue_pending")))


# -- raw tools ---------------------------------------------------------------------


def tool_status(box: str | None = None) -> dict:
    box = _box(box)
    host = comfy_host(box)
    stats = _get_json(host, "/system_stats")
    queue = _get_json(host, "/queue")
    dev = (stats.get("devices") or [{}])[0]
    free, total = dev.get("vram_free"), dev.get("vram_total")
    running, pending = _queue_ids(queue, "queue_running"), _queue_ids(queue, "queue_pending")
    return {
        "box": box,
        "host": host,
        "comfyui_version": (stats.get("system") or {}).get("comfyui_version"),
        "gpu": dev.get("name"),
        "vram_free_pct": round(100.0 * free / total, 1) if free is not None and total else None,
        "queue_running": running,
        "queue_pending": pending,
        "idle": not _queue_has_work(queue),
    }


def tool_models(kind: str, q: str | None = None, box: str | None = None) -> dict:
    if kind not in MODEL_KINDS:
        raise BridgeError(f"models: kind must be one of {', '.join(MODEL_KINDS)}")
    from loramake.comfy import object_info_enum

    node_type, field = MODEL_KINDS[kind]
    names = sorted(object_info_enum(node_type, field, comfy_host(_box(box))))
    if q:
        needle = q.lower()
        names = [n for n in names if needle in n.lower()]
    return {"kind": kind, "loader": f"{node_type}.{field}", "total": len(names), "names": names[:LIST_CAP]}


def _object_info(host: str) -> dict:
    hit = _object_info_cache.get(host)
    if hit and time.monotonic() - hit[0] < OBJECT_INFO_TTL_S:
        return hit[1]
    info = _get_json(host, "/object_info", timeout=60.0)
    _object_info_cache[host] = (time.monotonic(), info)
    return info


def _trim_spec(spec: dict) -> dict:
    """Input spec with combo option lists cut short: the full lists are comfy_models' job."""
    out: dict = {}
    for group in ("required", "optional"):
        fields = (spec.get("input") or {}).get(group) or {}
        for name, decl in fields.items():
            head = decl[0] if isinstance(decl, list) and decl else decl
            meta = decl[1] if isinstance(decl, list) and len(decl) > 1 and isinstance(decl[1], dict) else {}
            # Combo fields come as [[names], meta] (classic) or ["COMBO", {"options": [names]}] (new).
            options = head if isinstance(head, list) else meta.get("options") if head == "COMBO" else None
            if options is not None:
                out[name] = {"type": "COMBO", "options": list(options)[:10], "option_count": len(options)}
            else:
                out[name] = {"type": head, **{k: meta[k] for k in ("default", "min", "max") if k in meta}}
            if group == "optional":
                out[name]["optional"] = True
    return out


def tool_nodes(q: str | None = None, node_type: str | None = None, box: str | None = None) -> dict:
    host = comfy_host(_box(box))
    if node_type:
        info = _get_json(host, f"/object_info/{urllib.parse.quote(node_type)}")
        spec = info.get(node_type)
        if not isinstance(spec, dict):
            raise BridgeError(f"nodes: unknown node type {node_type!r}")
        return {
            "node_type": node_type,
            "category": spec.get("category"),
            "inputs": _trim_spec(spec),
            "outputs": list(spec.get("output_name") or spec.get("output") or []),
        }
    names = sorted(_object_info(host))
    if q:
        needle = q.lower()
        names = [n for n in names if needle in n.lower()]
    return {"total": len(names), "node_types": names[:LIST_CAP]}


def tool_upload(path: str, box: str | None = None) -> dict:
    from loramake.comfy import upload_image

    local = Path(path).expanduser()
    if not local.is_file():
        raise BridgeError(f"upload: file not found: {path}")
    with contextlib.redirect_stdout(sys.stderr):
        name = upload_image(str(local), comfy_host(_box(box)))
    if not name:
        raise BridgeError(f"upload: ComfyUI did not accept {local.name}")
    return {"name": name, "bytes": local.stat().st_size}


def _looks_like_media(raw: bytes) -> bool:
    """PNG, GIF, WebP/RIFF or MP4 magic: an HTML error page from /view must not be saved as output."""
    from loramake.comfy import looks_like_png

    return looks_like_png(raw) or raw[:4] in (b"GIF8", b"RIFF") or raw[4:8] == b"ftyp"


# Nodes that mutate the cached model in place; later renders on that base stay altered until
# ComfyUI's node cache is reset, which a plain unload does not do.
IN_PLACE_PATCH_NODES = frozenset({"AnimaIPAdapterApply", "ModelComputeDtype"})


def _patches_in_place(graph: dict) -> bool:
    return any(isinstance(node, dict) and node.get("class_type") in IN_PLACE_PATCH_NODES
               for node in graph.values())


def _free_if_idle(host: str, box: str, *, enabled: bool, clear_cache: bool = False) -> bool | None:
    """POST /free only when this box's queue is empty.

    A timeout or ``force=true`` path can leave (or arrive behind) someone else's
    prompt. Unreadable queue is a skip, same discipline as ``vram_reaper``.
    """
    if not enabled:
        return None
    try:
        if not tool_status(box).get("idle"):
            return False
    except Exception:
        return False
    from loramake.comfy import free_vram

    return free_vram(host, clear_cache=clear_cache)


def tool_run(
    graph: dict | None = None,
    path: str | None = None,
    box: str | None = None,
    timeout: float = RUN_TIMEOUT_S,
    label: str = "run",
    force: bool = False,
    free: bool = True,
) -> dict:
    from loramake.comfy import submit_and_fetch

    prepared = bridge._prepare_load_api({"graph": graph, "path": path, "name": label})
    box = _box(box)
    host = comfy_host(box)
    status = tool_status(box)
    if not status["idle"] and not force:
        raise BridgeError(
            f"run: {box} is busy (running={status['queue_running']} pending={status['queue_pending']}); "
            "wait, or pass force=true to queue behind it"
        )
    log = io.StringIO()
    started = time.monotonic()
    # submit_and_fetch narrates on stdout, which is the MCP wire on stdio: keep it in a buffer.
    # Release VRAM after our own run, but never from under a job that queued meanwhile.
    try:
        with contextlib.redirect_stdout(log):
            fetched = submit_and_fetch(
                prepared["graph"], label=label, host=host, timeout=timeout,
                output_keys=("images", "gifs"), validate=_looks_like_media, first_only=False,
            )
    finally:
        freed = _free_if_idle(host, box, enabled=free, clear_cache=_patches_in_place(prepared["graph"]))
    tail = log.getvalue().strip().splitlines()[-8:]
    if not fetched:
        raise BridgeError("run: no output (" + " | ".join(tail) + ")")
    out_dir = _unique_path(RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{_safe(label)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, (raw, item) in enumerate(fetched):
        # Index prefix: two nodes can emit the same filename in one run.
        dest = _unique_path(out_dir / f"{index:02d}_{_safe(item.get('filename', 'output.png'))}")
        dest.write_bytes(raw)
        saved.append(str(dest))
    return {
        "box": box,
        "outputs": saved,
        "elapsed_s": round(time.monotonic() - started, 1),
        "vram_freed": freed,
        "log": tail,
    }


def _compact_history(pid: str, entry: dict) -> dict:
    from loramake.comfy_errors import _execution_error_detail

    status = entry.get("status") or {}
    outputs = []
    for node_id, node in (entry.get("outputs") or {}).items():
        for key in ("images", "gifs"):
            for item in node.get(key, []):
                outputs.append({
                    "node": node_id, "filename": item.get("filename"),
                    "subfolder": item.get("subfolder", ""), "type": item.get("type", "output"),
                })
    out = {"prompt_id": pid, "status": status.get("status_str"), "completed": status.get("completed"),
           "outputs": outputs}
    if status.get("status_str") == "error":
        out["error"] = _execution_error_detail(entry)
    return out


def tool_history(prompt_id: str | None = None, box: str | None = None, limit: int = 5) -> dict:
    host = comfy_host(_box(box))
    if prompt_id:
        body = _get_json(host, f"/history/{urllib.parse.quote(prompt_id)}")
        if prompt_id not in body:
            raise BridgeError(f"history: prompt {prompt_id} not found")
        return _compact_history(prompt_id, body[prompt_id])
    body = _get_json(host, f"/history?max_items={max(1, min(int(limit), 50))}")
    return {"items": [_compact_history(pid, entry) for pid, entry in reversed(list(body.items()))]}


def tool_fetch(filename: str, subfolder: str = "", type: str = "output", box: str | None = None) -> dict:
    host = comfy_host(_box(box))
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": type})
    url = f"{host.rstrip('/')}/view?{query}"
    try:
        with _urlopen(_box_request(url), timeout=120.0) as resp:
            raw = resp.read(MAX_FETCH_BYTES)
    except urllib.error.HTTPError as exc:
        raise BridgeError(f"fetch: HTTP {exc.code} for {filename}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BridgeError(f"fetch: {exc}") from exc
    if not raw:
        raise BridgeError(f"fetch: empty body for {filename}")
    dest_dir = RUNS_DIR / "fetched"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(dest_dir / _safe(Path(filename).name))
    dest.write_bytes(raw)
    return {"path": str(dest), "bytes": len(raw)}


def tool_cancel(prompt_id: str, box: str | None = None) -> dict:
    from loramake.comfy import QueueStateError, cancel_prompt

    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise BridgeError("cancel: prompt_id must be a non-empty string")
    try:
        return cancel_prompt(comfy_host(_box(box)), prompt_id.strip())
    except QueueStateError as exc:
        raise BridgeError(f"cancel: {exc}") from exc


def tool_free(box: str | None = None, clear_cache: bool = False) -> dict:
    from loramake.comfy import free_vram

    box = _box(box)
    return {"box": box, "ok": free_vram(comfy_host(box), clear_cache=bool(clear_cache))}


RAW_TOOLS = {
    "comfy_status": tool_status,
    "comfy_models": tool_models,
    "comfy_nodes": tool_nodes,
    "comfy_upload": tool_upload,
    "comfy_run": tool_run,
    "comfy_history": tool_history,
    "comfy_fetch": tool_fetch,
    "comfy_cancel": tool_cancel,
    "comfy_free": tool_free,
}


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def register_raw(mcp) -> None:
    @mcp.tool(name="comfy_status", structured_output=False)
    def comfy_status(box: str | None = None) -> str:
        """Box health: ComfyUI version, GPU, VRAM free %, running/pending prompt ids, idle flag. Call before comfy_run."""
        return _dump(tool_status(box))

    @mcp.tool(name="comfy_models", structured_output=False)
    def comfy_models(kind: str, q: str | None = None, box: str | None = None) -> str:
        """Loadable model files by kind (checkpoint|lora|vae|controlnet|upscale|clip_vision|unet), optional substring q. Returns the exact names loader nodes accept. For LoRA intent search use scripts/boxa_model_index.py lookup."""
        return _dump(tool_models(kind, q, box))

    @mcp.tool(name="comfy_nodes", structured_output=False)
    def comfy_nodes(q: str | None = None, node_type: str | None = None, box: str | None = None) -> str:
        """Node types installed on the box (substring q), or with node_type the input/output spec of one node for composing API graphs."""
        return _dump(tool_nodes(q, node_type, box))

    @mcp.tool(name="comfy_upload", structured_output=False)
    def comfy_upload(path: str, box: str | None = None) -> str:
        """Upload a local image into the box's input/ folder; returns the name a LoadImage node takes."""
        return _dump(tool_upload(path, box))

    @mcp.tool(name="comfy_run", structured_output=False)
    def comfy_run(
        graph: dict | None = None,
        path: str | None = None,
        box: str | None = None,
        timeout: float = RUN_TIMEOUT_S,
        label: str = "run",
        force: bool = False,
        free: bool = True,
    ) -> str:
        """Submit an API-format graph (inline graph, or path under outputs/_analysis/), wait, download every output to outputs/_analysis/comfy_runs/<ts>_<label>/. Refuses a busy box unless force=true. Frees VRAM after unless free=false. Smoke one image before a batch."""
        return _dump(tool_run(graph, path, box, timeout, label, force, free))

    @mcp.tool(name="comfy_history", structured_output=False)
    def comfy_history(prompt_id: str | None = None, box: str | None = None, limit: int = 5) -> str:
        """Past executions: one prompt_id in full (status, outputs, node error detail), or the last `limit` items."""
        return _dump(tool_history(prompt_id, box, limit))

    @mcp.tool(name="comfy_fetch", structured_output=False)
    def comfy_fetch(filename: str, subfolder: str = "", type: str = "output", box: str | None = None) -> str:
        """Download one output/input/temp file by name (as listed by comfy_history) to outputs/_analysis/comfy_runs/fetched/."""
        return _dump(tool_fetch(filename, subfolder, type, box))

    @mcp.tool(name="comfy_cancel", structured_output=False)
    def comfy_cancel(prompt_id: str, box: str | None = None) -> str:
        """Cancel one prompt: deletes it if pending, interrupts only if it is the sole running item."""
        return _dump(tool_cancel(prompt_id, box))

    @mcp.tool(name="comfy_free", structured_output=False)
    def comfy_free(box: str | None = None, clear_cache: bool = False) -> str:
        """POST /free: unload models and release VRAM (house rule after ad-hoc runs on the shared BOXA). Set clear_cache=true after IP-Adapter or any node that patches the model in place: it also resets the node cache so the next plain render is clean."""
        return _dump(tool_free(box, clear_cache))


def register_all(mcp, client: bridge.BridgeClient | None = None) -> None:
    """``client`` defaults to the serving client (background stale sweep); the CLI passes a
    plain one so listing tools never touches the network."""
    image.register(mcp)
    bridge.register(mcp, client or bridge.new_client())
    register_raw(mcp)


# -- entry points --------------------------------------------------------------------


class _Collector:
    """Stand-in MCPServer for the CLI: records tool callables instead of serving them."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *, name: str, structured_output: bool = False):
        def decorator(fn):
            self.tools[name] = fn
            return fn
        return decorator


def main() -> None:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    if image._cloud_mode():
        # Off-LAN (cloud) session: the full surface stays registered. The generate tools
        # ride the Studio tunnel under the session's channel scope; the canvas/raw box
        # tools resolve their hosts from env (BOXA_HOST / COMFY_BRIDGE_HOST /
        # LORAMAKE_COMFY_BOX) and fail fast until a tunnel reaches the box. A plain
        # bridge client — the serving client's startup sweep would only poke an
        # unreachable host.
        register_all(mcp, bridge.BridgeClient(bridge.comfy_host()))
    else:
        register_all(mcp)
    mcp.run(transport="stdio")


def _print_blocks(result) -> None:
    if isinstance(result, list):
        for block in result:
            print(block if isinstance(block, str) else getattr(block, "path", repr(block)))
    else:
        print(result)


def cli(argv: list[str]) -> int:
    if argv and argv[0] == "tools":
        collector = _Collector()
        register_all(collector, bridge.BridgeClient(comfy_host()))
        for name, fn in collector.tools.items():
            print(f"{name:18s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    if len(argv) >= 2 and argv[0] == "call":
        collector = _Collector()
        register_all(collector, bridge.BridgeClient(comfy_host()))
        name = argv[1]
        if name not in collector.tools:
            print(f"unknown tool {name}; run `tools` for the list", file=sys.stderr)
            return 2
        try:
            args = json.loads(argv[2]) if len(argv) > 2 else {}
            if not isinstance(args, dict):
                raise BridgeError("args must be a JSON object")
            _print_blocks(collector.tools[name](**args))
        except (BridgeError, json.JSONDecodeError, TypeError, KeyError) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 1
        return 0
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
