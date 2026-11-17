#!/usr/bin/env python3
"""Intent -> engine router for the loramake image-generation system.

There are ~5 real generation intents, each served by a different engine with a different CLI. This
router maps the intent + a few uniform options (prompt, n, seed, lora, box) onto the correct engine's
actual flags, and injects the per-box quirks that callers keep re-deriving. It does NOT re-expose every
engine option: anything engine-specific goes after `--` and is appended verbatim. Engines are invoked
by subprocess and never modified.

Decision tree (run `gen.py --list` to print it):
  best       one high-quality image            -> boxa_quality_gen.py --family illustrious --max
  artistic   anti-slop artistic single-char    -> boxa_quality_gen.py --max + art_prompt (filter + contact sheet)
  variants   explore many candidates           -> boxa_gen.py --n N
  draft      turbo draft in seconds            -> boxa_anima_fast.py (promote via `gen.py promote --source <draft>`)
  anima_structured_pilot  strict source + registered identity/pose -> anima_structured_pilot.py (BOXB only)
  sdxl_structured_pilot   strict source + registered identity/pose/direction -> sdxl_structured_pilot.py (BOXB only)
  promote    draft -> finished Anima img2img    -> boxa_quality_gen.py --family anima --max --init-image <source>
  regional   multi-subject / interaction       -> region_prompt.py --mode couple  (+ --refine for detail)
  region     localized repaint of a masked area -> region_fix.py --image <source> (mask spec after `--`)
  reproduce  clone an existing image           -> image_reproduce.py <source>

Uniform options (mapped to each engine's real flag): --prompt, --n, --seed, --lora (repeatable).
  - The checkpoint flag differs per engine; --box handles it, you don't pass a ckpt.
  - `--prompt` for `regional` becomes the global base/scene prompt; pass the per-region prompts after `--`.
  - `reproduce` takes `--source IMG` instead of `--prompt`; `promote` takes both source and prompt/meta.

Box quirks (the centralized knowledge — verified 2026-06-28):
  --box auto (default): resolve at invocation time by a quick BOXA scan first — if BOXA looks
                       gaming/busy (low VRAM or non-empty queue, or unreachable), or another
                       session holds BOXA in the GPU ledger, route to BOXB immediately; otherwise
                       use BOXA (user policy 2026-08-16). BOXA-only intents
                       (gacha2.0, and `draft --turbo` without `--unet`, whose 2.9B baked UNET only
                       BOXA has) pin boxa without probing.
  --box boxa           : BOXA_HOST=192.0.2.10:8188, each engine's own default checkpoint.
  --box boxb          : BOXA_HOST=192.0.2.20:8388, checkpoint forced to iLustMix.safetensors; and for
                       `regional --refine`, also BOXA_FACE_DETECTOR=bbox/face_yolov8m.pt + --detailers face
                       (BOXB has no eye detector). `draft` and `promote` run on either box (user
                       decisions 2026-09-30; boxa_quality_gen resolves the Anima UNET name per box);
                       `draft --turbo` without `--unet` stays on BOXA. For `draft -- --unet <card>`
                       the card in anima_models.json decides: its `boxes` (a BOXA-only card pins boxa),
                       its `turbo` pairing for `--turbo`, and the loader name per box (the router
                       renames `--unet` from `unet_by_box`). `region` on an Anima
                       source is not forced to iLustMix; region_fix resolves the Anima UNET itself.
  Explicit BOXA_HOST overrides only the selected box's endpoint (for a proven SSH/Cloudflare/local adapter);
  --box still controls model and detector quirks.
  Under managed Claude, an unset BOXA_HOST resolves to loopback SSH adapters
  (BOXA 127.0.0.1:18188 / BOXB 127.0.0.1:18388); normal callers keep the LAN defaults.

Output:
  omit --run         print the exact, copy-pasteable command (env prefix + argv); touches nothing.
  --run              wait until the box is idle (VRAM + empty queue — never fires onto a busy/gaming
                     GPU), then execute it via subprocess with the right env merged. --no-wait skips
                     the idle probe. A busy box is waited on up to 10 minutes; a box that does not
                     answer 4 probes in a row (~1 min) exits 75 at once, so batch wrappers can stop.
  --pick-best        (with --run) generate N, then QC-score the batch via qc_score.py and print the
                     single best-of-N path — so a caller (Hermes) gets one vetted image, no eyeballing.
                     Pass --subjects N to also flag dropped/duplicated subjects. Non-destructive.

Examples:
  gen.py best --prompt "1girl, silver hair, gothic dress, masterpiece"
  gen.py variants --prompt "1girl, red dress" --n 6 --lora "some_style:0.8"
  gen.py best --prompt "1girl, silver hair, gothic dress" --n 4 --subjects 1 --pick-best --run
  gen.py artistic --prompt "1girl, solo, long silver hair, red eyes" --n 6 --run
  gen.py artistic --prompt "1girl, solo, ..." --style-mode artist --artist "by ke-ta:1.0" --aspect tall --n 6 --run
  gen.py artistic --prompt "1girl, solo, ..." --style-lora "Illustrious\\style\\HueSpark1llust.safetensors" --ref face.png --n 6 --run
  gen.py regional --box boxb --refine --prompt "2girls, cafe, side by side, looking at viewer" \
      -- --split lr --region "1girl, silver hair, black dress" --region "1girl, red hair, red uniform" \
         --interact "holding hands"
  gen.py reproduce --source ref.png --n 4 --run
  gen.py promote --source outputs/anima/by_model/claude/anima_fast/draft.png --n 1 --run
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import anima_models  # sibling: Anima model cards (boxes, loader name per box, turbo pairing)
import art_directions  # sibling: art direction cards (coherent slot bundles for --direction)
import art_prompt  # sibling: artistic-intent prompt builder (anti-slop body + negative + aspect/sampler)
import model_quirks  # sibling: per-base quirks cards (resolves a card's base to its engine ckpt)
from config import DIRECT_COMFY_HOSTS, OUTPUTS, REPO_OUTPUTS, REPO_ROOT, resolve_comfy_host  # sibling shim to loramake.config
from loramake import image_providers  # paid cloud adapters; credentials never enter argv/logs
from loramake import model_resources as _mr  # shared LoRA catalog: alias/family/availability
from loramake import net_routes  # route cache: preview-only fetch off the LAN
from loramake import route_plan  # route-plan seam (leaf): resource compiler + machine-readable plan
from loramake import vram_lease  # VRAM lease seam (leaf): keep models warm between renders
from loramake.forge_endpoints import resolve_forge_host  # forge intents: FORGE_HOST, never BOXA_HOST
from loramake.prompt_compose import apply_optional_ir, emit_ir_warnings  # Phase C --ir lint/warn
from boxa_comfy import (POST_FX_DEFAULT, _access_argv, apply_post_fx,  # shared driver helpers
                      atomic_write_text, free_payload)

ROOT = Path(__file__).resolve().parent
HOSTS = dict(DIRECT_COMFY_HOSTS)
ANIMA_FAMILY_NAMES = ("anima", "flow")    # foot_fix.FAMILY_ALIASES entries that resolve to anima
BOXB_CKPT = "iLustMix.safetensors"          # BOXB's house Illustrious base (BOXA's is the engine default)
BOXB_FACE_DETECTOR = "bbox/face_yolov8m.pt"  # BOXB's only face detector (no yolov9c / eye detector)
PROBE_VRAM = 0.40       # min free-VRAM ratio to treat a box as idle (matches image_reproduce)
PROBE_MAX_WAIT = 600.0  # seconds to wait for a busy/gaming box before giving up
PROBE_INTERVAL = 15.0   # seconds between idle re-checks
PROBE_UNREACHABLE_STREAK = 4  # consecutive no-answer probes (~1 min at PROBE_INTERVAL) = route down
EXIT_BOX_UNREACHABLE = 75     # sysexits EX_TEMPFAIL: the box did not answer; batch wrappers stop on it
PLAN_BOX_PROBE_TIMEOUT_S = 2.0  # print/route planning: fail fast instead of blocking on a dead box
RUN_BOX_PROBE_TIMEOUT_S = 8.0   # real launches keep the historical reachability budget
PROBE_ONESHOT_TIMEOUT_S = 5.0   # --probe: one readiness probe, never a wait loop
LEDGER_TIMEOUT_S = 5.0          # GPU ledger read; capped further by the caller's probe budget
LEDGER_BOOK = Path.home() / "Documents" / "fleet" / "ledger" / "book.py"  # LORAMAKE_LEDGER_BOOK overrides
BOX_CACHE_TTL_S = 30.0          # cross-process reachability cache for planning (a stale true is re-checked)
BOX_CACHE_PATH = ROOT.parent / "app" / "data" / "box_reachability_v1.json"
ROUTE_PLAN_SCHEMA = "loramake.route-plan/v1"
ROUTE_CATALOG_SCHEMA = "loramake.route-catalog/v1"
OUTPUTS_ROOT = OUTPUTS  # every engine writes under here; --pick-best diffs this tree
DEFAULT_ARTISTIC_CONTACT_SHEET = OUTPUTS_ROOT / "_gallery" / "artistic_contact_sheet.png"

# artistic pose-lock: the verified openpose recipe (pose-library LIBRARY.md, 2026-07-01). Skeletons are
# pre-made, so strength/end run at 1.0 and --res MUST match the skeleton canvas (a mismatch stretches
# the control map into duplicate faces). Index read with stdlib only (pose_library would pull in PIL).
POSE_INDEX = REPO_OUTPUTS / "poses/library/index.json"
CHARACTERS_ROOT = REPO_ROOT / "characters"
POSE_RES_TALL = "832x1472"     # 9:16 skeletons (768x1344 canvas)
POSE_RES_SQUARE = "1024x1024"  # 1:1 skeleton variants (floor / lying poses)
ART_SLOT_KEYS = ("medium", "camera", "composition", "lighting", "pose", "color_mood", "expression")
ART_LINE = "illustrious"  # the artistic intent renders on the Illustrious quality stack
PAIR_SLOTS = ("lighting", "camera", "composition", "color_mood", "expression")


def _contact_sheet_path(value: str | None) -> tuple[Path, bool]:
    """Return the contact-sheet target and whether it was explicitly supplied.

    Explicit paths are expanded and resolved so machine-readable results never carry a path that a
    downstream packet consumer could accidentally resolve relative to its own result directory.
    The default remains the historical fixed absolute path and keeps its overwrite behaviour.
    """
    if value is None:
        return DEFAULT_ARTISTIC_CONTACT_SHEET, False
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False), True


def _host_ip(url: str) -> str:
    return url.split("//", 1)[-1].split(":", 1)[0]


def _effective_host(box: str) -> str:
    """Return an explicit connection-adapter endpoint or the selected box's canonical LAN URL.

    A box-specific BOXB_HOST outranks the generic BOXA_HOST for --box boxb, as on the pilot line."""
    box_specific = (os.environ.get("BOXB_HOST") or "").strip() if box == "boxb" else ""
    return resolve_comfy_host(box, explicit=box_specific or os.environ.get("BOXA_HOST"))


def _curl_json(url: str, timeout: float = RUN_BOX_PROBE_TIMEOUT_S) -> dict | None:
    """GET a ComfyUI JSON endpoint (LAN curl --noproxy by default; CF Access mode via env)."""
    try:
        # timeout= backstops a wedged curl process; --max-time only caps the transfer
        out = subprocess.run(["curl", "-s", *_access_argv(_host_ip(url)),
                              "--max-time", f"{max(1.0, timeout):g}", url],
                             capture_output=True, text=True,
                             timeout=max(1.0, timeout) + 5).stdout
    except subprocess.TimeoutExpired:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def _free_vram(host: str) -> bool:
    """POST /free to release the box's VRAM (standing op after every gen — the boxes are shared).

    The body comes from ``free_payload`` so the router and the driver cannot drift on which of
    the two flags they send; see that function for why ``free_memory`` is off by default.
    """
    proc = subprocess.run(["curl", "-s", *_access_argv(_host_ip(host)), "--max-time", "10",
                           "-X", "POST", "-H", "Content-Type: application/json",
                           "-d", json.dumps(free_payload()), f"{host}/free"],
                          capture_output=True, text=True)
    return proc.returncode == 0


# The lease and its reaper are anchored at REPO_ROOT, not at this script's own directory, so
# the NSFW mirror shares them instead of keeping a second copy. Paths are nsfw-local; a GPU is
# not. BOXA is one physical box: two lease files would mean two reapers, two locks and neither
# aware of the other, so one could free the card while the other was mid-burst.
def _lease_dir() -> Path:
    """Where the VRAM leases live. Same app/data convention as the box-reachability cache."""
    override = os.environ.get("LORAMAKE_VRAM_LEASE_DIR", "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / "app" / "data"


def _start_reaper(box: str) -> bool:
    """Detach a reaper for this box. Idempotent: the reaper's own lock turns away the second one,
    so the caller never has to know whether one is already running.

    start_new_session detaches it from this process group, or a Ctrl-C aimed at the router would
    take the reaper with it and leave the box leased with nobody watching.
    """
    try:
        subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "vram_reaper.py"), "--box", box],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def _release_or_lease(host: str, box: str, *, free_now: bool) -> dict:
    """Hand the box back now, or stamp the lease and let the reaper do it once things go quiet.

    Returns the ``free`` block of the result JSON. ``free_ok`` stays None in lease mode, which is
    the field's existing initial value and what app-web already declares (`free_ok?: boolean`).
    """
    if free_now or not vram_lease.enabled():
        ok = bool(_free_vram(host))
        print(f"  [free] {box} VRAM " + ("released" if ok else
                                         f"release FAILED — POST {host}/free manually"))
        return {"mode": "now", "released": ok, "free_ok": ok}
    idle_s = vram_lease.idle_seconds()
    path = vram_lease.lease_path(_lease_dir(), box)
    try:
        vram_lease.touch(path, box=box, host=host, write_text=atomic_write_text)
    except OSError as exc:
        # A lease we cannot write is a box nobody will release. Fall back rather than leak it.
        ok = bool(_free_vram(host))
        print(f"  [free] {box} lease unavailable ({exc}); released now" if ok else
              f"  [free] {box} lease unavailable and release FAILED — POST {host}/free manually")
        return {"mode": "now", "released": ok, "free_ok": ok, "note": f"lease unavailable: {exc}"}
    started = _start_reaper(box)
    print(f"  [free] {box} VRAM leased — releases after {idle_s:g}s idle"
          + ("" if started else " (reaper did not start; run scripts/vram_reaper.py --box "
                               f"{box}, or POST {host}/free)"))
    return {"mode": "lease", "idle_s": idle_s, "reaper": "started" if started else "not_started",
            "released": False}


def _is_shared_gaming_host(host: str) -> bool:
    """True for BOXA — the box that may be running a game with an empty ComfyUI queue."""
    parsed = urlparse(host if "://" in host else f"http://{host}")
    if parsed.port in (8188, 18188):
        return True
    boxa = urlparse(resolve_comfy_host("boxa", explicit=os.environ.get("BOXA_HOST")))
    return (parsed.hostname or "").lower() == (boxa.hostname or "").lower() and parsed.port == boxa.port


# When resolve_box's auto scan just declared BOXA idle (VRAM healthy + queue empty), the
# immediate wait_for_idle re-probe of the same box is redundant; this timestamp lets the
# --run flow skip it while the observation is seconds old. Explicit --box never sets it.
_AUTO_BOXA_IDLE_AT: float | None = None
_AUTO_PROBE_REUSE_S = 3.0


class IdleVerdict(tuple):
    """``(ok, reason)`` from probe_idle / wait_for_idle; ``.unreachable`` marks a box that did not answer.

    It unpacks and compares as the historical 2-tuple, so ``ok, reason = wait_for_idle(...)``
    callers keep working; callers that must tell "down" from "busy" read ``.unreachable``.
    """

    unreachable: bool

    def __new__(cls, ok: bool, reason: str, *, unreachable: bool = False) -> "IdleVerdict":
        if ok and unreachable:
            raise ValueError("an idle verdict cannot be unreachable")
        verdict = super().__new__(cls, (bool(ok), reason))
        verdict.unreachable = unreachable
        return verdict


def is_unreachable(verdict: tuple[bool, str]) -> bool:
    """True when a probe_idle / wait_for_idle verdict says the box did not answer (not merely busy)."""
    return isinstance(verdict, IdleVerdict) and verdict.unreachable


def probe_idle(host: str, timeout: float = RUN_BOX_PROBE_TIMEOUT_S) -> IdleVerdict:
    """True when ComfyUI is reachable and its queue is empty.

    BOXA also requires VRAM headroom so a game holding the GPU is not treated as idle.
    BOXB only checks the queue — resident models from the previous job are expected.
    A box that does not answer comes back with ``unreachable=True``.
    """
    stats = _curl_json(f"{host}/system_stats", timeout=timeout)
    if not stats:
        return IdleVerdict(False, "system_stats unreachable", unreachable=True)
    dev = (stats.get("devices") or [{}])[0]
    total, free = dev.get("vram_total") or 0, dev.get("vram_free") or 0
    ratio = (free / total) if total else 0.0
    queue = _curl_json(f"{host}/queue", timeout=timeout)
    if queue is None:
        return IdleVerdict(False, "queue unreachable", unreachable=True)
    running, pending = len(queue.get("queue_running") or []), len(queue.get("queue_pending") or [])
    if running or pending:
        return IdleVerdict(False, f"queue busy (running={running}, pending={pending})")
    if _is_shared_gaming_host(host) and ratio < PROBE_VRAM:
        return IdleVerdict(False, f"VRAM only {ratio:.0%} free ({free // 1048576}MB/{total // 1048576}MB)")
    if _is_shared_gaming_host(host):
        return IdleVerdict(True, f"idle (VRAM {ratio:.0%} free)")
    return IdleVerdict(True, f"idle (queue empty; VRAM {ratio:.0%} free)")


def wait_for_idle(host: str, max_wait: float = PROBE_MAX_WAIT, interval: float = PROBE_INTERVAL,
                  unreachable_streak: int = PROBE_UNREACHABLE_STREAK) -> IdleVerdict:
    """Poll probe_idle until the box is free or max_wait elapses (probes exactly once if max_wait <= 0).

    A box that answers and is busy is waited on up to ``max_wait``. A box that does not answer
    ``unreachable_streak`` probes in a row is a dropped route, not a busy GPU: return at once with
    ``unreachable=True`` so a batch caller stops instead of waiting out max_wait per scene.
    """
    start = time.monotonic()
    misses = 0
    while True:
        verdict = probe_idle(host)
        ok, reason = verdict
        if ok:
            return IdleVerdict(True, reason)
        misses = misses + 1 if is_unreachable(verdict) else 0
        if misses >= unreachable_streak:
            return IdleVerdict(False, f"unreachable: {reason} ({misses} probes in a row)", unreachable=True)
        if time.monotonic() - start >= max_wait:
            return IdleVerdict(False, reason, unreachable=misses > 0)
        print(f"  {host} not idle: {reason} — waiting {int(interval)}s ...", flush=True)
        time.sleep(interval)


def _box_cache_enabled() -> bool:
    return os.environ.get("LORAMAKE_BOX_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _box_cache_path() -> Path:
    override = os.environ.get("LORAMAKE_BOX_CACHE_PATH", "").strip()
    return Path(override).expanduser() if override else BOX_CACHE_PATH


def _box_cache_lookup(boxa_host: str, boxb_host: str) -> tuple[str, str] | None:
    """Last auto-box decision for this exact host pair, when fresh enough.

    ``None`` = no usable cache entry.  Only print/route planning consumes it; a real
    ``--run`` always re-scans BOXA so a cached "BOXA idle" can never make a GPU job wait on
    a BOXA the user just started gaming on.
    """
    if not _box_cache_enabled():
        return None
    try:
        data = json.loads(_box_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != "loramake.box-route/v1":
        return None
    if data.get("boxa_host") != boxa_host or data.get("boxb_host") != boxb_host:
        return None
    box, reason, observed_at = data.get("box"), data.get("reason"), data.get("observed_at")
    if box not in HOSTS or not isinstance(reason, str) or not isinstance(observed_at, (int, float)):
        return None
    try:
        ttl = float(os.environ.get("LORAMAKE_BOX_CACHE_TTL", str(BOX_CACHE_TTL_S)))
    except ValueError:
        ttl = BOX_CACHE_TTL_S
    age = time.time() - observed_at
    return (box, reason) if 0.0 <= age <= max(0.0, ttl) else None


def _box_cache_write(boxa_host: str, boxb_host: str, box: str, reason: str) -> None:
    """Best-effort atomic cache write; never changes route resolution on failure."""
    if not _box_cache_enabled():
        return
    path = _box_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        payload = {"schema": "loramake.box-route/v1", "boxa_host": boxa_host, "boxb_host": boxb_host,
                   "box": box, "reason": reason, "observed_at": time.time()}
        Path(tmp).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except OSError:
        pass


# --------------------------------------------------------------------------- resource compiler

def _catalog_freshness_problems(*, index_path=None) -> list[str]:
    """Offline LoRA-catalog gate; skipped for mock index_path (tests) and opt-out env."""
    if index_path is not None and Path(index_path) != _mr.INDEX_PATH:
        return []
    allow = os.environ.get("LORAMAKE_ALLOW_STALE_CATALOG", "").lower() in ("1", "true", "yes")
    return _mr.catalog_readiness_problems(allow_stale=allow)


def _compile_resources(intent: str,
                       requested_box: str | None = None,
                       base: str | None = None,
                       loras: list[str] | None = None,
                       style_loras: list[str] | None = None,
                       pose_ref: str | None = None,
                       index_path=None) -> _mr.ResourcePlan:
    """Compile resource constraints and eligible boxes for an intent/brief.

    Facade over ``loramake.route_plan.compile_resources``: the router-owned inputs (engine-table
    entry, quirks-card lookup, box host table, offline catalog gate) are bound here at call time.
    """
    return _compile_resources_for_spec(ENGINES.get(intent, {}), intent, requested_box, base, loras,
                                       style_loras, pose_ref, index_path)


def _compile_resources_for_spec(spec: dict, intent: str, requested_box: str | None, base: str | None,
                                loras: list[str] | None, style_loras: list[str] | None,
                                pose_ref: str | None, index_path=None) -> _mr.ResourcePlan:
    """``_compile_resources`` for an explicit engine-table entry (see ``_source_spec``)."""
    return route_plan.compile_resources(
        spec,
        intent,
        requested_box,
        base,
        loras,
        style_loras,
        pose_ref,
        index_path,
        card_lookup=model_quirks.card,
        allowed_boxes=set(HOSTS),
        catalog_problems=_catalog_freshness_problems(index_path=index_path),
    )


def _list_from_value(value: object) -> list[str]:
    """Coerce validated brief/arg LoRA values without raising on untrusted input."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [value for value in value if isinstance(value, str)]


def _gather_loras(primary: object, extras: object) -> list[str]:
    """Collect the primary ``lora`` plus every ``loras[]`` entry."""
    out = []
    out.extend(_list_from_value(primary))
    out.extend(_list_from_value(extras))
    return out


def _gather_style_loras(primary: object, extras: object) -> list[str]:
    """Collect the primary ``style_lora`` plus every ``style_loras[]`` entry."""
    return _gather_loras(primary, extras)


def _direction_style_loras(direction: object) -> list[str]:
    """Return style LoRAs declared on an artistic direction card, if any."""
    if not isinstance(direction, str) or not direction:
        return []
    card = art_directions.get(direction)
    if not card:
        return []
    return [str(s) for s in card.get("style_loras") or []]


def _is_resource_tier(intent: object) -> bool:
    """Return False for malformed intents and tiers that keep their own resource domains."""
    if not isinstance(intent, str) or not intent:
        return False
    if intent not in ENGINES:
        return False
    spec = ENGINES[intent]
    if spec.get("structured_pilot") or spec.get("forge"):
        return False
    if intent in ("forge", "forge_repro", "comfy_repro"):
        return False
    return True


def _provider_local(brief: dict) -> bool:
    return brief.get("provider") in (None, "", "local")


def _gather_brief_resources(brief: dict) -> tuple[list[str], list[str]]:
    """Collect (loras, style_loras) for a brief, including direction card defaults."""
    loras = _gather_loras(brief.get("lora"), brief.get("loras"))
    style_loras = _gather_style_loras(brief.get("style_lora"), brief.get("style_loras"))
    if not style_loras:
        style_loras = _direction_style_loras(brief.get("direction"))
    return loras, style_loras


def _passthrough_value(extra: list[str], flag: str) -> str | None:
    """Last value of ``flag`` in the engine passthrough (``--flag V`` or ``--flag=V``)."""
    value = None
    for i, tok in enumerate(extra):
        if tok == flag and i + 1 < len(extra):
            value = extra[i + 1]
        elif tok.startswith(flag + "="):
            value = tok.split("=", 1)[1]
    return value


def _source_spec(intent: str, args: argparse.Namespace, extra: list[str]) -> dict:
    """ENGINES[intent], with resource_family switched to anima when a family_from_source intent
    repaints an Anima image. Decided as foot_fix.source_family decides it in the engine:
    passthrough --family, then the sidecar's family (unless the passthrough has --no-meta), then
    anima when the sidecar names an Anima UNET."""
    spec = ENGINES[intent]
    if intent == "draft":
        why = _draft_boxa_only_why(extra)
        if why:
            return _boxa_only_spec(spec, why)
    source = getattr(args, "source", None)
    if not spec.get("family_from_source") or not source:
        return spec
    family = _passthrough_value(extra, "--family")
    if family is None and "--no-meta" not in extra:
        try:
            meta = json.loads(Path(str(source) + ".meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if isinstance(meta, dict):
            family = meta.get("family") or ("anima" if _mr.anima_family(meta.get("unet")) else None)
    if str(family or "").strip().lower() in ANIMA_FAMILY_NAMES:
        return {**spec, "resource_family": "anima"}
    return spec


def _compile_resources_for_args(args: argparse.Namespace, extra: list[str]) -> _mr.ResourcePlan:
    """Pre-box resource compilation for a CLI invocation, family resolved from the source."""
    return _compile_resources_for_spec(
        _source_spec(args.intent, args, extra),
        args.intent,
        args.box,
        getattr(args, "base", None),
        _effective_loras_for_compile(args, args.intent),
        _effective_style_loras_for_compile(args),
        getattr(args, "pose_ref", None),
    )


def _effective_loras_for_compile(args: argparse.Namespace, intent: str) -> list[str]:
    """Effective character LoRAs for pre-box resource compilation.

    For ``promote`` this includes the draft sidecar's ``content_loras`` when the
    caller did not override with explicit ``--lora`` flags.
    """
    loras = list(getattr(args, "lora", None) or [])
    if intent == "promote" and not loras:
        sidecar = Path(str(args.source) + ".meta.json")
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                for entry in meta.get("content_loras") or []:
                    if not isinstance(entry, dict):
                        continue
                    fname = entry.get("file")
                    strength = entry.get("strength")
                    if isinstance(fname, str) and fname:
                        if isinstance(strength, (int, float, str)) and strength != "":
                            loras.append(f"{fname}:{strength}")
                        else:
                            loras.append(fname)
            except (json.JSONDecodeError, OSError):
                pass
    return loras


def _effective_style_loras_for_compile(args: argparse.Namespace) -> list[str]:
    """Effective style LoRAs for pre-box resource compilation.

    Explicit ``--style-lora`` wins; otherwise an artistic direction card's
    ``style_loras`` are inherited.
    """
    style_loras = list(getattr(args, "style_lora", None) or [])
    if style_loras:
        return style_loras
    direction = getattr(args, "direction", None)
    if direction:
        card = art_directions.get(direction)
        if card:
            return [str(s) for s in card.get("style_loras") or []]
    return []


def _resolved_for_box(traces: list[dict], box: str) -> list[str]:
    """Pick the already-compiled per-box resolved name from a list of traces."""
    out = []
    for trace in traces:
        resolved = trace.get(f"resolved_{box}") or trace.get("resolved") or trace["requested"]
        out.append(resolved)
    return out


# draft runs on either box, except these engine flags, which load a checkpoint the BOXB route
# cannot serve (anima_models.json: turbo_profile is registered on BOXA only; the legacy-turbo base is
# on BOXB under another loader name, and boxa_anima_fast does not rename it per box). An explicit
# --unet replaces the --turbo pin with its card's verdict (_draft_boxa_only_why).
BOXA_ONLY_DRAFT_FLAGS = {
    "--turbo": "draft --turbo loads the 2.9B baked UNET, which only BOXA has",
    "--legacy-turbo": ("draft --legacy-turbo loads anima-preview3-base by its BOXA loader name, "
                       "which boxa_anima_fast does not rename per box"),
}


def _draft_unet_card(unet: str | None) -> dict | None:
    """The anima_models card a draft ``--unet`` value names, or None.

    Strict, as anima_models.unet_for_box matches: the value must contain the card key, so a bare
    stem such as ``ultrabake`` never picks a sibling card."""
    card = anima_models.card(unet) if unet else None
    if card and card["name"].lower() in unet.lower():
        return card
    return None


def _draft_boxa_only_why(extra: list[str]) -> str | None:
    """Why a draft passthrough can run on BOXA only, or None when BOXB can serve it too.

    --legacy-turbo, and --turbo without --unet, keep the BOXA_ONLY_DRAFT_FLAGS pins. An explicit
    --unet defers to its card: BOXB must be in the card's ``boxes``, and with --turbo the turbo
    LoRA (the card's pairing, or --turbo-lora) must not be catalogued as missing from BOXB or as
    present there under another name, because boxa_anima_fast loads that LoRA by name without
    renaming it per box. An uncatalogued LoRA passes, as everywhere else in the router; the live
    preflight is the authority. A --turbo whose --unet names no card stays on BOXA: nothing records
    what BOXB has for it."""
    if "--legacy-turbo" in extra:
        return BOXA_ONLY_DRAFT_FLAGS["--legacy-turbo"]
    turbo = "--turbo" in extra
    unet = _passthrough_value(extra, "--unet")
    if unet is None:
        return BOXA_ONLY_DRAFT_FLAGS["--turbo"] if turbo else None
    card = _draft_unet_card(unet)
    if card is None:
        if turbo:
            return (f"draft --turbo with --unet {unet!r}: no anima_models card names that UNET, "
                    "so its turbo pairing cannot be checked on BOXB")
        return None
    boxes = card.get("boxes") or []
    if "boxb" not in boxes:
        return f"anima_models card {card['name']!r} lists boxes {boxes}"
    if not turbo or card.get("baked_turbo"):
        return None
    lora = _passthrough_value(extra, "--turbo-lora") or (card.get("turbo") or {}).get("lora")
    if not lora:
        return f"draft --turbo: anima_models card {card['name']!r} has no turbo pairing"
    trace, eligible = _mr.compile_lora_trace(lora, "anima")
    if "boxb" not in eligible or trace.get("resolved_boxb") not in (None, lora):
        return f"draft --turbo on {card['name']!r}: turbo LoRA {lora!r} is not on BOXB under that name"
    return None


def _draft_passthrough_for_box(extra: list[str], box: str) -> list[str]:
    """A copy of a draft passthrough with each ``--unet`` value renamed to ``box``'s loader name.

    anima_models.unet_for_box does the renaming from the card's ``unet_by_box``; a value that
    names no card is kept as written."""
    out: list[str] = []
    rename_next = False
    for tok in extra:
        if rename_next:
            out.append(anima_models.unet_for_box(tok, box))
            rename_next = False
        elif tok == "--unet":
            out.append(tok)
            rename_next = True
        elif tok.startswith("--unet="):
            out.append("--unet=" + anima_models.unet_for_box(tok.split("=", 1)[1], box))
        else:
            out.append(tok)
    return out


def _boxa_only_spec(spec: dict, why: str) -> dict:
    """An engine row pinned to BOXA for one invocation, with the reason printed on refusal."""
    return {**spec, "boxa_only": True, "boxa_only_why": why}


def _brief_spec(intent: str, brief: dict) -> dict:
    """ENGINES[intent] as a job brief sees it: a Studio turbo draft (`turbo: true`) is BOXA-only."""
    spec = ENGINES.get(intent, {})
    if intent == "draft" and brief.get("turbo") is True:
        return _boxa_only_spec(spec, BOXA_ONLY_DRAFT_FLAGS["--turbo"])
    return spec


def resource_problems_for_brief(brief: dict, *, index_path=None) -> list[str]:
    """Submit-time resource problems for a job brief (empty = pass).  Does not probe."""
    intent = brief.get("tier") or brief.get("intent")
    if not _provider_local(brief) or not _is_resource_tier(intent):
        return []
    requested = brief.get("box") or "auto"
    if not isinstance(requested, str):
        return []
    base = brief.get("base")
    loras, style_loras = _gather_brief_resources(brief)
    pose_ref = brief.get("pose_ref")
    plan = _compile_resources_for_spec(_brief_spec(intent, brief), intent, requested, base, loras,
                                       style_loras, pose_ref, index_path)
    return plan.problems


def resolve_box_for_brief(brief: dict, *, use_cache: bool = False,
                          probe_timeout: float = RUN_BOX_PROBE_TIMEOUT_S,
                          index_path=None) -> tuple[str, str]:
    """Resolve a job brief to a concrete box, validating resource constraints."""
    intent = brief.get("tier") or brief.get("intent")
    if not _provider_local(brief) or not _is_resource_tier(intent):
        return resolve_box(brief.get("box") or "auto", intent,
                           use_cache=use_cache, probe_timeout=probe_timeout)
    requested = brief.get("box") or "auto"
    base = brief.get("base")
    loras, style_loras = _gather_brief_resources(brief)
    pose_ref = brief.get("pose_ref")
    plan = _compile_resources_for_spec(_brief_spec(intent, brief), intent, requested, base, loras,
                                       style_loras, pose_ref, index_path)
    if plan.problems:
        raise ValueError("; ".join(plan.problems))
    if requested != "auto" and requested in plan.eligible_boxes:
        return requested, "resource-constrained explicit --box"
    return resolve_box(
        requested, intent,
        use_cache=use_cache,
        probe_timeout=probe_timeout,
        eligible=plan.eligible_boxes,
    )


def _curl_json_budget(url: str, timeout: float) -> dict | None:
    """Timeout-aware curl while preserving the historical one-arg monkeypatch surface."""
    if timeout == RUN_BOX_PROBE_TIMEOUT_S:
        return _curl_json(url)
    return _curl_json(url, timeout=timeout)


def boxa_ledger_holder(timeout: float = LEDGER_TIMEOUT_S) -> str | None:
    """Say why the GPU ledger keeps this run off BOXA, or None when it does not.

    Another session's in_progress entry keeps it off; the caller's own entry (owner_cli equals
    CLAUDE_CODE_SESSION_ID) does not count.  A book that exists but cannot be read keeps it off
    too: a failed read says nothing about who holds BOXA, and reading it as free lands a batch on
    a held card.  Only a missing book reads as no holder, because gen.py also runs on machines
    without the fleet repo.
    """
    book = os.environ.get("LORAMAKE_LEDGER_BOOK", "").strip() or str(LEDGER_BOOK)
    if not Path(book).is_file():
        return None
    session = os.environ.get("CLAUDE_CODE_SESSION_ID")
    try:
        proc = subprocess.run(["python3", "-I", book, "--box", "boxa", "show", "--json"],
                              capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        if proc.returncode != 0:
            return f"BOXA ledger unreadable (book.py exited {proc.returncode})"
        rows = json.loads(proc.stdout)
        if not isinstance(rows, list):
            raise ValueError("not a list of rows")
        holder = next((row for row in rows
                       if row["status"] == "in_progress"
                       and not (session and row.get("owner_cli") == session)), None)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"BOXA ledger unreadable ({type(exc).__name__})"
    except (ValueError, TypeError, KeyError):
        return "BOXA ledger unreadable (book.py printed no row list)"
    if holder is None:
        return None
    return f"BOXA held by ledger entry {holder.get('id')} ({holder.get('owner')}: {holder.get('title')})"


def probe_boxa_in_use(host: str, timeout: float = RUN_BOX_PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Quick BOXA gaming/busy scan for auto routing.

    True = use BOXB.  VRAM is checked first (games hold VRAM even with an empty ComfyUI
    queue); only a VRAM-healthy BOXA pays the second /queue request.  Unreachable BOXA also
    routes away rather than assuming it is free.  A BOXA that passes both is still in use
    when another session holds it in the GPU ledger, or when the ledger cannot be read: a
    training job can leave VRAM free and the ComfyUI queue empty, and the gpu-guard no longer
    pauses it for ComfyUI work.
    """
    stats = _curl_json_budget(f"{host}/system_stats", timeout)
    if not stats:
        return True, "BOXA system_stats unreachable"
    dev = (stats.get("devices") or [{}])[0]
    total, free = dev.get("vram_total") or 0, dev.get("vram_free") or 0
    ratio = (free / total) if total else 0.0
    if ratio < PROBE_VRAM:
        return True, f"BOXA VRAM only {ratio:.0%} free (gaming/busy possible)"
    queue = _curl_json_budget(f"{host}/queue", timeout)
    if queue is None:
        return True, "BOXA queue unreachable"
    running, pending = len(queue.get("queue_running") or []), len(queue.get("queue_pending") or [])
    if running or pending:
        return True, f"BOXA queue busy (running={running}, pending={pending})"
    held = boxa_ledger_holder(timeout=min(LEDGER_TIMEOUT_S, timeout))
    if held:
        return True, held
    return False, f"BOXA idle (VRAM {ratio:.0%} free; queue empty)"


def resolve_box(requested: str = "auto", intent: str | None = None, *,
                use_cache: bool = False,
                probe_timeout: float = RUN_BOX_PROBE_TIMEOUT_S,
                eligible: set[str] | frozenset[str] | None = None) -> tuple[str, str]:
    """Resolve --box auto; explicit boxa/boxb passes through untouched.

    Policy (user decision 2026-08-16): auto first quick-scans BOXA.  If BOXA looks like it
    is being used for gaming / is busy, route to BOXB immediately; otherwise use BOXA.
    This replaces the unconditional BOXB-first policy of 2026-07-18.

    ``use_cache`` is for planning commands only: it reuses the last auto decision across
    processes so new conversations do not re-pay a dead-box timeout for every
    `gen.py --print`.  ``--run`` always calls with ``use_cache=False`` and re-scans BOXA.

    When ``eligible`` is given, resolution is restricted to that box set.
    """
    global _AUTO_BOXA_IDLE_AT
    # Any path that does not actually probe BOXA must not leave a stale idle timestamp
    # from a previous call, otherwise the --run flow may skip wait_for_idle wrongly.
    _AUTO_BOXA_IDLE_AT = None
    if eligible is None:
        if requested in HOSTS:
            return requested, "explicit --box"
        if intent and ENGINES.get(intent, {}).get("boxa_only"):
            return "boxa", f"'{intent}' is BOXA-only"
        # BOXA_HOST is one opaque adapter endpoint and carries no box identity. Before
        # --box auto existed, omitting --box meant boxa; preserve that contract instead of
        # scanning canonical BOXA and then applying model quirks to an adapter that may
        # actually terminate elsewhere. Callers targeting a BOXB adapter must say --box boxb.
        if str(os.environ.get("BOXA_HOST") or "").strip():
            return "boxa", "explicit BOXA_HOST with implicit box — preserving BOXA compatibility"
        boxa_host, boxb_host = HOSTS["boxa"], HOSTS["boxb"]
        if use_cache:
            cached = _box_cache_lookup(boxa_host, boxb_host)
            if cached is not None:
                return cached
        boxa_in_use, boxa_reason = probe_boxa_in_use(boxa_host, timeout=probe_timeout)
        box = "boxb" if boxa_in_use else "boxa"
        reason = f"{boxa_reason} -> {box}"
        _AUTO_BOXA_IDLE_AT = time.monotonic() if not boxa_in_use else None
        if use_cache:
            _box_cache_write(boxa_host, boxb_host, box, reason)
        return box, reason

    # Constrained path: resource compilation has produced an allowed box set.
    if requested != "auto" and requested in eligible:
        return requested, "explicit --box"
    if requested != "auto":
        if requested in {"boxa", "boxb"}:
            raise ValueError(f"box {requested!r} is incompatible with the resource constraints")
        raise ValueError(f"unknown box {requested!r}")
    if not eligible:
        raise ValueError("no eligible box satisfies the resource constraints")
    # explicit BOXA_HOST + auto + boxb-only resources is a conflict: the caller must
    # explicitly say --box boxb for an opaque adapter, otherwise BOXA quirks get
    # silently applied to an adapter that may terminate elsewhere.
    boxa_host_env = str(os.environ.get("BOXA_HOST") or "").strip()
    if boxa_host_env:
        if "boxa" in eligible:
            return "boxa", "explicit BOXA_HOST with implicit box — preserving BOXA compatibility"
        raise ValueError(
            "explicit BOXA_HOST with auto and boxb-only resource constraints: "
            "pass --box boxb if you intend to use the opaque adapter"
        )
    if len(eligible) == 1:
        box = next(iter(eligible))
        return box, f"resource constraints pin {box}"
    # eligible == {"boxa", "boxb"}: preserve the historical BOXA-first scan policy.
    boxa_host, boxb_host = HOSTS["boxa"], HOSTS["boxb"]
    if use_cache:
        cached = _box_cache_lookup(boxa_host, boxb_host)
        if cached is not None and cached[0] in eligible:
            return cached
    boxa_in_use, boxa_reason = probe_boxa_in_use(boxa_host, timeout=probe_timeout)
    box = "boxb" if boxa_in_use else "boxa"
    reason = f"{boxa_reason} -> {box}"
    _AUTO_BOXA_IDLE_AT = time.monotonic() if not boxa_in_use else None
    if use_cache:
        _box_cache_write(boxa_host, boxb_host, box, reason)
    return box, reason

# intent -> engine spec. flag names are the engine's ACTUAL argparse flags (verified), so the router
# can map the uniform options onto each. `ckpt` is the flag that carries a checkpoint override on that
# engine (differs: --base on the quality/fast lines, --ckpt on region/reproduce). None = not applicable.
ENGINES: dict[str, dict] = {
    "best": {"script": "boxa_quality_gen.py", "fixed": ["--family", "illustrious", "--max"],
             "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
             "resource_family": "sdxl",
             "gist": "one high-quality image (FreeU/PAG + upscale + face/eye detailer)"},
    "sequence": {"script": "boxa_quality_gen.py", "fixed": ["--family", "illustrious", "--max", "--ip", "face"],
                 "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
                 "resource_family": "sdxl", "ref_aware": True,
                 "gist": "same character across varied scenes (--ref face-lock via IPAdapter + wildcard "
                         "scenes; auto CCIP consistency report). Put __background__/__lighting__ etc. in --prompt"},
    "artistic": {"script": "boxa_quality_gen.py", "fixed": ["--family", "illustrious", "--max"],
                 "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
                 "resource_family": "sdxl", "artistic": True, "ref_optional": True,
                 "gist": "anti-slop artistic single-char (art_prompt composition/medium/lighting/pose "
                         "scaffold; on --run filters defects/identity then surfaces survivors + a contact "
                         "sheet for a by-eye pick). --style-mode lora|artist, --style-lora, --aspect, "
                         "--detail; optional --ref to also lock identity"},
    "variants": {"script": "boxa_gen.py", "fixed": [],
                 "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
                 "resource_family": "sdxl",
                 "gist": "explore many candidates (seed/denoise sweep, semantic LoRA)"},
    "draft": {"script": "boxa_anima_fast.py", "fixed": [],
              "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": None,
              "resource_family": "anima",
              "gist": "turbo Anima draft (~seconds); promote via `gen.py promote`. "
                      "`-- --turbo` alone (2.9B baked) runs on BOXA only; `-- --unet <card> [--turbo]` "
                      "follows the anima_models card (boxes, turbo pairing, loader name per box)"},
    "gacha2.0": {"script": "t2i_gacha.py", "fixed": [],
                 "prompt": None, "n": "--count", "seed": "--seed", "lora": None, "ckpt": None,
                 "resource_family": "anima", "boxa_only": True,
                 "prompt_note": "prompts come from catalogs/t2i; preview via `-- --dry-run`",
                 "gist": "random catalog gacha 2.0: chaos-sampled catalogs/t2i prompts on the Anima "
                         "fast line, fail-closed wd14+qc+aesthetic gate; no --prompt (prompts come "
                         "from the catalog; preview via `-- --dry-run`); knobs after `--`. BOXA only"},
    "anima_structured_pilot": {
        "script": "anima_structured_pilot.py", "fixed": [],
        "prompt": None, "n": None, "seed": None, "lora": None, "ckpt": None,
        "boxb_only": True, "structured_pilot": True,
        "gist": "isolated Anima Preview3 identity+pose Pilot; strict JSON, exact source text, BOXB only",
    },
    "sdxl_structured_pilot": {
        "script": "sdxl_structured_pilot.py", "fixed": [],
        "prompt": None, "n": None, "seed": None, "lora": None, "ckpt": None,
        "boxb_only": True, "structured_pilot": True,
        "gist": "isolated SDXL iLustMix identity+pose+direction Pilot; strict JSON, exact source text, BOXB only",
    },
    "promote": {"script": "boxa_quality_gen.py",
                "fixed": ["--family", "anima", "--max", "--denoise", "0.45"],
                "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
                "resource_family": "anima", "source": True, "source_flag": "--init-image",
                "gist": "promote a draft to a finished Anima image (img2img --max, stays in-family). "
                        "Reads the draft's .meta.json for prompt/res/LoRAs; --prompt overrides. "
                        "Base defaults to the registry's quality_default card (Dreamland ANM); "
                        "--base <anima quirks stem> re-bases the promote. Passthrough after `--` can "
                        "override fixed flags or add Anima control, e.g. `-- --denoise 0.4` or "
                        "`-- --control lineart --control-image ref.png`"},
    "region": {"script": "region_fix.py", "fixed": [],
               "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
               "resource_family": "sdxl", "family_from_source": True,
               "source": True, "source_flag": "--image",
               "gist": "localized repaint of a masked area (mask compiled from boxes/detector picks/"
                       "keep-outs); an Anima source (sidecar family/unet) repaints with Anima. "
                       "Selection goes after `--`: `-- --spec-json '<json>'` or "
                       "`-- --bbox X,Y,W,H --mode refine|replace|polish`"},
    "regional": {"script": "region_prompt.py", "fixed": ["--mode", "couple"],
                 "prompt": "--base", "n": None, "seed": "--seed", "lora": "--lora", "ckpt": "--ckpt",
                 "resource_family": "sdxl", "refine_aware": True,
                 "gist": "multi-subject / interaction (couple<=2; area for 3+). --refine adds fine detail"},
    "rescue": {"script": "rescue_gen.py", "fixed": [],
               "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": "--base",
               "resource_family": "sdxl", "source": True, "source_flag": "--source", "boxb_only": True,
               "gist": "structure rescue: pose-locked re-render of a broken image (openpose extracted "
                       "from the source drops extra limbs/fused objects/broken proportions; sidecar "
                       "prompt+LoRAs reused; pass --prompt to also fix an OOD-token literalisation). "
                       "One shot, fixed recipe. BOXB only (pose lock unverified on BOXA)"},
    "revise": {"script": "revise.py", "fixed": [],
               "prompt": None, "n": None, "seed": None, "lora": None, "ckpt": None,
               "source": True, "source_flag": "--parent-image", "revision": True,
               "gist": "bounded change to an existing image: the parent's own recipe with a "
                       "declared minimal delta written over it and every other dimension "
                       "frozen. --source is the picture being revised; --revision-json carries "
                       "the freeze contract (changed/frozen dimensions, the delta, the "
                       "intervention rung). scope=region repaints only inside an uploaded mask "
                       "and leaves the rest of the frame untouched by construction"},
    "reproduce": {"script": "image_reproduce.py", "fixed": [],
                  "prompt": None, "n": "--candidates", "seed": "--seed", "lora": "--lora", "ckpt": "--ckpt",
                  "resource_family": "sdxl", "source": True,
                  "gist": "invert a no-metadata image (multi-metric loop: CCIP+graycorr+tagF1, "
                          "tag-diff prompt refinement; metadata-bearing sources are redirected "
                          "to forge_repro/comfy_repro)"},
    # Forge line (A1111 kernel on BOXA :7862 via the 17862 loopback tunnel — docs/forge_line.md).
    # The engine is PLAN-by-default, so --run travels in `fixed`; `ckpt: None` because Forge
    # checkpoints are aki-junctioned, not model_quirks-registered (pass `-- --ckpt NAME`).
    "forge": {"script": "forge_gen.py", "fixed": ["--run"],
              "prompt": "--prompt", "n": "--n", "seed": "--seed", "lora": "--lora", "ckpt": None,
              "boxa_only": True, "forge": True,
              "gist": "Forge txt2img (A1111 kernel, BOXA only) — prompt/LoRA/hires/ADetailer; "
                      "the faithful line for SDE-sampler recipes"},
    "forge_repro": {"script": "forge_gen.py", "fixed": ["--run"],
                    "prompt": None, "n": "--n", "seed": "--seed", "lora": None, "ckpt": None,
                    "boxa_only": True, "forge": True, "source": True, "source_flag": "--source",
                    "gist": "replay a Forge/A1111 image from its embedded parameters "
                            "(hash-matched assets + metadata round-trip + similarity report)"},
    "comfy_repro": {"script": "comfy_repro.py", "fixed": ["--run"],
                    "prompt": None, "n": "--n", "seed": "--seed", "lora": None, "ckpt": None,
                    "source": True, "source_flag": "--source",
                    "gist": "replay a ComfyUI image from its embedded API graph (same-kernel "
                            "exact repro, Anima included; full-graph enum preflight)"},
}


def print_list(as_json: bool = False) -> None:
    if as_json:
        routes = []
        for name, spec in ENGINES.items():
            routes.append({
                "intent": name,
                "engine": spec["script"],
                "family": spec.get("resource_family"),
                "gist": spec["gist"],
                "boxa_only": bool(spec.get("boxa_only")),
                "boxb_only": bool(spec.get("boxb_only")),
                "source_required": bool(spec.get("source")),
                "ref_required": bool(spec.get("ref_aware")),
                "ref_optional": bool(spec.get("ref_optional")),
                "prompt_flag": spec.get("prompt"),
                "n_flag": spec.get("n"),
                "seed_flag": spec.get("seed"),
                "lora_flag": spec.get("lora"),
                "ckpt_flag": spec.get("ckpt"),
            })
        print(json.dumps({"schema": ROUTE_CATALOG_SCHEMA,
                          "uniform_flags": ["--prompt", "--n", "--seed", "--lora"],
                          "boxes": sorted(HOSTS),
                          "routes": routes}, ensure_ascii=False, indent=2))
        return
    print("intent     engine                gist")
    print("-" * 92)
    for name, spec in ENGINES.items():
        print(f"{name:<10} {spec['script']:<21} {spec['gist']}")
    print("\nuniform: --prompt --n --seed --lora --box {auto,boxa,boxb}; engine-specific flags go after `--`.")
    print("output : omit --run to show the command; --run executes it.")
    print("plan   : --route / --route-json PATH print or persist a machine-readable route plan;")
    print("         --probe adds a one-shot readiness check (GET-only, never launches the engine).")
    print("routes : lines outside this router (krea2 / civitai / SillyTavern / pose-library)")
    print("         -> docs/GEN_ROUTES.md (the route map)")


def _source_intents_label() -> str:
    return " / ".join(f"'{name}'" for name, spec in ENGINES.items() if spec.get("source"))


def _resolve_base(args: argparse.Namespace, spec: dict) -> tuple[str | None, dict | None]:
    """Router-level --base: resolve a model_quirks card stem to the engine ckpt path.

    The card must explicitly register the chosen --box (cards carry `boxes: [...]`), so a
    stale or mistyped stem can never send an absent / wrong-box checkpoint to an engine.
    Intents without a ckpt flag (draft) keep their engine-line checkpoint and refuse --base
    outright; promote's sidecar --base is appended before the router's, so an explicit
    --base still wins (argparse last-wins). Returns (ckpt, card): the ckpt path to append after the
    passthrough (argparse last-wins) plus the resolved card so the caller can apply its
    riders (neg_add / no_freeu), or (None, None) when --base was not given."""
    if getattr(args, "base", None) is None:
        return None, None
    if not spec["ckpt"]:
        sys.exit(f"--base does not apply to '{args.intent}' (its checkpoint is fixed by the engine line).")
    card = model_quirks.card(args.base)
    if not card:
        stems = list(model_quirks._load().get("bases", {}))
        close = difflib.get_close_matches(args.base.lower(), [s.lower() for s in stems], n=3, cutoff=0.4)
        hint = f"; closest: {', '.join(s for s in stems if s.lower() in close)}" if close else ""
        sys.exit(f"--base: no quirks card matches {args.base!r} (see: model_quirks.py list{hint})")
    boxes = card.get("boxes")
    if not boxes:
        sys.exit(f"--base: card {card['base']!r} has no per-box registration yet (boxes field missing)")
    if args.box not in boxes:
        sys.exit(f"--base: {card['base']!r} is not registered on box {args.box} "
                 f"(registered: {', '.join(boxes)})")
    if spec.get("resource_family"):
        base_family = card.get("family") or card.get("ckpt")
        try:
            if not _mr.family_compatible(base_family, spec["resource_family"]):
                sys.exit(f"--base: {card['base']!r} is not a {spec['resource_family']}-family checkpoint")
        except _mr.UnknownFamilyError:
            sys.exit(f"--base: unknown family {spec['resource_family']!r}")
    ckpt = card.get("ckpt")
    if not ckpt:
        sys.exit(f"--base: card {card['base']!r} has no engine ckpt path")
    return ckpt, card


def _promote_defaults(args: argparse.Namespace) -> tuple[str | None, list[str], list[str]]:
    """Defaults for promote from <source>.meta.json; explicit prompt/LoRAs win."""
    sidecar = Path(str(args.source) + ".meta.json")
    prompt = args.prompt
    loras = list(args.lora or [])
    extra: list[str] = []
    meta: dict | None = None
    meta_problem = "missing"
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
                meta_problem = ""
            else:
                meta_problem = "not a JSON object"
        except (json.JSONDecodeError, OSError) as exc:
            meta_problem = f"{type(exc).__name__}: {exc}"
    if meta is None:
        if prompt is None:
            sys.exit(f"'promote' needs --prompt when sidecar metadata is unavailable "
                     f"({meta_problem}); expected {sidecar}.")
        return prompt, extra, loras

    if prompt is None:
        meta_prompt = meta.get("prompt")
        if not isinstance(meta_prompt, str) or not meta_prompt.strip():
            sys.exit(f"'promote' needs --prompt because expected sidecar {sidecar} has no prompt.")
        prompt = meta_prompt
        # the sidecar prompt is the FINAL positive that produced the draft (scaffold and
        # trigger words already baked in by boxa_anima_fast), so reuse it verbatim: suppress
        # the quality line's own trigger injection AND scaffold prefix — the fast-line
        # SCAFFOLD is the same string as the anima recipe pos_prefix, so it would double up
        extra += ["--no-trigger-inject", "--no-scaffold"]

    res = meta.get("res")
    if isinstance(res, list) and len(res) == 2:
        try:
            w, h = int(res[0]), int(res[1])
        except (TypeError, ValueError):
            pass
        else:
            if w > 0 and h > 0:
                extra += ["--res", f"{w}x{h}"]

    # Anima drafts (boxa_anima_fast.py) record which checkpoint/ModelSampling/sampler actually made
    # them; forward those so promote stays on the same checkpoint instead of silently falling back
    # to boxa_quality_gen.py's RECIPES["anima"] default. steps/cfg are deliberately NOT forwarded:
    # promote always runs a fixed --denoise 0.45 img2img pass, and ComfyUI's effective diffusion
    # steps is roughly steps * denoise — forwarding a draft's low step count (10-12) into an
    # already-low-denoise refine would run only ~5 effective steps, likely worse rather than better.
    unet = meta.get("unet")
    if isinstance(unet, str) and unet.strip():
        extra += ["--base", unet]
    shift = meta.get("shift")
    if isinstance(shift, (int, float)):
        extra += ["--model-sampling", "auraflow", "--shift", str(shift)]
    elif meta.get("model_sampling") == "none":
        # the draft deliberately ran without ModelSamplingAuraFlow (e.g. anima_aestheticV11's
        # author recipe); without this, promote would silently re-add the quality recipe's
        # default shift and refine under a different sampling curve than the draft
        extra += ["--model-sampling", "none"]
    sampler = meta.get("sampler")
    if isinstance(sampler, str) and sampler.strip():
        extra += ["--sampler", sampler]
    scheduler = meta.get("scheduler")
    if isinstance(scheduler, str) and scheduler.strip():
        extra += ["--scheduler", scheduler]

    if not loras:
        for entry in meta.get("content_loras") or []:
            if not isinstance(entry, dict):
                continue
            fname = entry.get("file")
            strength = entry.get("strength")
            if isinstance(fname, str) and fname and isinstance(strength, (int, float, str)):
                loras.append(f"{fname}:{strength}")
    return prompt, extra, loras


def _build_structured_pilot(intent: str, spec: dict, args: argparse.Namespace,
                            extra: list[str]) -> tuple[dict[str, str], list[str]]:
    """Build an isolated structured-Pilot command without exposing a generic engine passthrough."""
    # LORAMAKE_NSFW_MIRROR is set by nsfw/gen_nsfw.py; under it the structured-Pilot intents are
    # refused because their engines resolve against the repo root and are flagged to the SFW tree.
    if os.environ.get("LORAMAKE_NSFW_MIRROR") == "1":
        sys.exit(
            f"'{intent}' is not available in the NSFW mirror: the structured-Pilot engines resolve "
            "their registry against the repo root and are feature-flagged to the SFW tree. "
            "Run it from scripts/gen.py instead."
        )
    if args.box != "boxb":
        sys.exit(f"'{intent}' is BOXB-only; BOXA remains reserved for the user.")
    if not args.task_json:
        sys.exit(f"'{intent}' requires --task-json with a strict TaskSpec.")
    incompatible = []
    for name in ("prompt", "source", "ref", "base"):
        if getattr(args, name) is not None:
            incompatible.append(f"--{name}")
    if args.n is not None:
        incompatible.append("--n")
    if args.seed is not None:
        incompatible.append("--seed")
    if args.lora:
        incompatible.append("--lora")
    if args.refine:
        incompatible.append("--refine")
    if args.pick_best:
        incompatible.append("--pick-best")
    if args.ccip_keep:
        incompatible.append("--ccip-keep")
    if incompatible:
        sys.exit(f"{intent} rejects router prompt/model overrides: " + ", ".join(incompatible))

    arity = {"--control-mode": 1, "--run-id": 1}
    seen: set[str] = set()
    index = 0
    while index < len(extra):
        flag = extra[index]
        if flag not in arity:
            sys.exit(f"{intent} passthrough rejects {flag!r}; allowed: --control-mode, --run-id")
        if flag in seen:
            sys.exit(f"{intent} passthrough repeats {flag}")
        seen.add(flag)
        if index + arity[flag] >= len(extra):
            sys.exit(f"{intent} passthrough {flag} is missing its value")
        index += 1 + arity[flag]

    # A box-specific BOXB adapter outranks the repo-wide generic transport override. The child
    # independently rejects known BOXA and unknown generic endpoints before any request is sent.
    box_specific = (os.environ.get("BOXB_HOST") or "").strip()
    pilot_explicit = box_specific or os.environ.get("BOXA_HOST")
    resolved = resolve_comfy_host("boxb", explicit=pilot_explicit)
    env = {"BOXA_HOST": resolved}
    # The execution path inherits the caller's full environment, but the default
    # print path must also be self-contained when copied into a fresh shell.
    if box_specific:
        env["BOXB_HOST"] = box_specific
    argv = [sys.executable, str(ROOT / spec["script"]),
            "--task-json", args.task_json, "--run", *extra]
    return env, argv


def build(intent: str, args: argparse.Namespace, extra: list[str],
          resource_plan: _mr.ResourcePlan | None = None) -> tuple[dict[str, str], list[str]]:
    """Return (env_overrides, argv) for the chosen intent. Raises SystemExit on invalid combinations."""
    spec = _source_spec(intent, args, extra)
    # LORAMAKE_NSFW_MIRROR is set by nsfw/gen_nsfw.py; under it t2i_gacha.py is refused because
    # the script and its catalogs live in the SFW tree (nsfw/t2i_gacha_nsfw.py is the remake).
    if spec.get("script") == "t2i_gacha.py" and os.environ.get("LORAMAKE_NSFW_MIRROR") == "1":
        sys.exit(
            f"'{intent}' is not available in the NSFW mirror: t2i_gacha.py and its catalogs live "
            "in the SFW tree. Use nsfw/t2i_gacha_nsfw.py instead."
        )
    if args.box == "boxb" and spec.get("boxa_only"):
        why = spec.get("boxa_only_why") or ("the Forge deployment lives on BOXA only" if spec.get("forge")
                                          else "the Anima router lines are pinned to BOXA by routing policy")
        sys.exit(f"'{intent}' is BOXA-only ({why}); drop --box boxb.")
    if args.box == "boxa" and spec.get("boxb_only"):
        sys.exit(f"'{intent}' is BOXB-only; use --box boxb.")
    if spec.get("structured_pilot"):
        return _build_structured_pilot(intent, spec, args, extra)
    if getattr(args, "task_json", None) is not None:
        sys.exit("--task-json only applies to the structured Pilot intents.")
    # Checked before _build_artistic returns, so the flag cannot be silently ignored on the
    # one intent that takes its own build path.
    if not spec.get("revision") and getattr(args, "revision_json", None) is not None:
        sys.exit("--revision-json only applies to 'revise'.")
    source_intents = _source_intents_label()
    if spec.get("source") and not args.source:
        sys.exit(f"'{intent}' needs --source <image> (source intents: {source_intents}).")
    if not spec.get("source") and args.source:
        sys.exit(f"--source only applies to {source_intents}, not '{intent}'.")
    if spec.get("ref_aware") and not args.ref:
        sys.exit(f"'{intent}' needs --ref <face image> (the identity anchor for IPAdapter face-lock).")
    if not spec.get("ref_aware") and not spec.get("ref_optional") and args.ref:
        sys.exit(f"--ref only applies to 'sequence' / 'artistic', not '{intent}'.")
    if args.refine and not spec.get("refine_aware"):
        sys.exit(f"--refine only applies to 'regional', not '{intent}'.")

    if spec.get("artistic"):
        return _build_artistic(args, spec, extra, resource_plan=resource_plan)

    env: dict[str, str] = {"BOXA_HOST": _effective_host(args.box)}
    if spec.get("forge"):
        # The engine talks to Forge through FORGE_HOST; BOXA_HOST stays pointed at the box's
        # ComfyUI on purpose — the run tail's idle probe and post-run POST /free go there,
        # keeping the shared-GPU gaming guard and VRAM handover intact (docs/forge_line.md).
        env["FORGE_HOST"] = resolve_forge_host("boxa", explicit=os.environ.get("FORGE_HOST"))
    argv: list[str] = [sys.executable, str(ROOT / spec["script"]), *spec["fixed"]]
    prompt_value = args.prompt
    engine_args: list[str] = []
    lora_values = list(args.lora or [])
    if intent == "promote":
        prompt_value, engine_args, lora_values = _promote_defaults(args)
    family = spec.get("resource_family")
    style_frag = None
    if getattr(args, "style", None):
        # --style on the generic path: draft / promote (Anima) and best (Illustrious). The
        # fragment's prefix leads the prompt, its LoRAs join --lora, its base pin becomes
        # --base (validated by _resolve_base like a user --base), its neg_add rides the
        # quality engine's --neg-add. Explicit --base wins; explicit --lora is kept alongside.
        from loramake import style_registry as sr
        line = "anima" if family == "anima" else "illustrious" if family == "sdxl" else None
        if line is None or not spec["prompt"]:
            sys.exit(f"--style does not apply to '{intent}' (no aesthetic line for it).")
        style_id = args.style
        try:
            if style_id == "random":
                import combo
                style_id = combo.draw_aesthetic(args.seed if args.seed is not None else 1, line)
            style_frag = sr.resolve_aesthetic(style_id, line, density=getattr(args, "density", None), box=args.box)
        except (sr.StyleRegistryError, ValueError) as exc:
            sys.exit(f"[style] {exc}")
        print(f"[style] {style_id} -> {style_frag.style_id} on {line}: {style_frag.realization_id} "
              f"({style_frag.kind}, tier={style_frag.tier}; {style_frag.note})")
        if style_frag.pos_prefix:
            prompt_value = f"{style_frag.pos_prefix}, {prompt_value}" if prompt_value else style_frag.pos_prefix
        if style_frag.lora_specs and spec["lora"]:
            lora_values = [*style_frag.lora_specs, *lora_values]
        if style_frag.base_pin and spec["ckpt"] and getattr(args, "base", None) is None:
            args.base = style_frag.base_pin
    if family and lora_values:
        if resource_plan is not None and not style_frag:
            lora_values = _resolved_for_box(resource_plan.lora_traces, args.box)
        else:
            try:
                lora_values, _ = _mr.lora_trace_for_box(lora_values, args.box, family)
            except ValueError as exc:
                sys.exit(f"[resource] {exc}")

    if spec.get("source"):
        if spec.get("source_flag"):
            argv += [spec["source_flag"], args.source]
        else:
            argv.append(args.source)  # image_reproduce takes the source as a positional
    if spec.get("revision"):
        if not args.revision_json:
            sys.exit(f"'{intent}' requires --revision-json with the freeze contract.")
        argv += ["--revision-json", args.revision_json]
    if spec.get("ref_aware"):
        argv += ["--ip-image", args.ref]  # sequence: identity anchor -> IPAdapter face reference (mode fixed above)
    if prompt_value is not None and spec["prompt"]:
        argv += [spec["prompt"], prompt_value]
    elif prompt_value is not None and not spec["prompt"]:
        note = spec.get("prompt_note") or "use --source / region args after `--`"
        sys.exit(f"'{intent}' does not take --prompt ({note}).")
    argv += engine_args
    if args.n is not None and spec["n"]:
        argv += [spec["n"], str(args.n)]
    if args.seed is not None and spec["seed"]:
        argv += [spec["seed"], str(args.seed)]
    if lora_values and not spec["lora"]:
        sys.exit(f"'{intent}' does not take --lora (LoRAs come from the source metadata).")
    for lora in lora_values:
        argv += [spec["lora"], lora]

    if args.refine:  # regional only (guarded above)
        argv.append("--refine")
        if args.box == "boxb":  # BOXB refine quirks: face detector name + drop the absent eye detailer
            env["BOXA_FACE_DETECTOR"] = BOXB_FACE_DETECTOR
            if "--detailers" not in extra:
                argv += ["--detailers", "face"]

    resolved_base, base_card = _resolve_base(args, spec)
    # --base rider auto-apply (quality engine only — the other engines have no --neg-add /
    # --no-freeu): the quirks card's hard-won appendix travels with its checkpoint instead of
    # relying on the caller to re-derive it. The engine flag is single-valued (last wins), so
    # any caller `-- --neg-add ...` is merged with the card's into ONE occurrence.
    rider_engines = ("boxa_quality_gen.py", "boxa_gen.py")  # both take --neg-add; only the quality engine has --no-freeu
    if style_frag and style_frag.neg_add and spec["script"] in rider_engines and not base_card:
        base_card = {"neg_add": style_frag.neg_add}
    elif style_frag and style_frag.neg_add and base_card and spec["script"] in rider_engines:
        base_card = {**base_card, "neg_add": ", ".join(p for p in (style_frag.neg_add, base_card.get("neg_add", "")) if p)}
    if base_card and spec["script"] in rider_engines:
        neg_parts = [p for p in [base_card.get("neg_add", "")] if p]
        while "--neg-add" in extra:
            i = extra.index("--neg-add")
            if i + 1 >= len(extra):
                sys.exit(f"'{intent}' passthrough `--neg-add` is missing its value")
            neg_parts.append(extra[i + 1])
            del extra[i:i + 2]
        if neg_parts:
            argv += ["--neg-add", ", ".join(dict.fromkeys(neg_parts))]
        if (base_card.get("no_freeu") and spec["script"] == "boxa_quality_gen.py"
                and "--no-freeu" not in extra and "--no-freeu" not in argv):
            argv.append("--no-freeu")

    # engine-specific passthrough, verbatim except a draft --unet, renamed to this box's loader name
    argv += _draft_passthrough_for_box(extra, args.box) if intent == "draft" else extra
    # checkpoint: an explicit router --base (quirks-card validated, box-registered) wins and is
    # appended after the passthrough (argparse last-wins); otherwise BOXB keeps forcing its house
    # ckpt so a caller-supplied --base/--ckpt cannot send a BOXA-only checkpoint to BOXB.
    if resolved_base:
        argv += [spec["ckpt"], resolved_base]
    elif args.box == "boxb" and spec["ckpt"] and spec.get("resource_family") != "anima":
        argv += [spec["ckpt"], BOXB_CKPT]
    return env, argv


def _png_snapshot() -> set[str]:
    """Every PNG path under outputs/ — snapshotted before/after a run to find what it produced.
    Engine-agnostic (no per-engine output-dir knowledge); mirrors image_reproduce._new_outputs."""
    if not OUTPUTS_ROOT.exists():
        return set()
    return {str(p) for p in OUTPUTS_ROOT.rglob("*.png")}


def _collect_run_files(files_out: Path | None, before: set[str], *,
                       allow_snapshot_fallback: bool = True) -> list[str]:
    """The PNGs THIS run produced. Engine-reported manifest first (GEN_FILES_OUT protocol:
    boxa_comfy.save_image_with_meta appends one absolute path per saved file), so a concurrent
    process writing the same outputs tree cannot leak into this run's result. Snapshot-diff
    fallback for engines that do not implement the protocol (e.g. boxa_anima_fast)."""
    if files_out is not None and files_out.is_file():
        lines = (ln.strip() for ln in files_out.read_text(encoding="utf-8").splitlines())
        reported = sorted({ln for ln in lines
                           if ln and ln.lower().endswith(".png") and Path(ln).is_file()})
        if reported:
            return reported
    return sorted(_png_snapshot() - before) if allow_snapshot_fallback else []


def _auto_fx(files: list[str], preset: str, overrides: str = "") -> list[str]:
    """Auto-apply the post_fx finishing pass to every raw output of this run.

    Each render lands as a `<stem>_fx.png` sibling (the raw engine PNG is never touched),
    and the fx paths are returned so callers can surface them alongside the raws.
    Failures are best-effort — a finishing pass must never fail a successful generation.
    """
    fx_files: list[str] = []
    for path in files:
        if not path.lower().endswith(".png") or path.lower().endswith("_fx.png"):
            continue
        sibling = str(Path(path).with_name(f"{Path(path).stem}_fx.png"))
        if os.path.isfile(sibling):
            # An fx-aware engine (boxa_anima_fast) already rendered it — adopt the sibling,
            # don't re-render (preset is consistent via LORAMAKE_POST_FX_PRESET).
            if sibling not in files:
                fx_files.append(sibling)
            continue
        out = apply_post_fx(path, preset, overrides=overrides)
        if out:
            fx_files.append(out)
    return fx_files


def _prompt_ids_from_events(path: str | None) -> list[str]:
    """Unique prompt ids emitted by transport events, in submission order."""
    if not path or not Path(path).is_file():
        return []
    prompt_ids: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        prompt_id = event.get("prompt_id") if isinstance(event, dict) else None
        if isinstance(prompt_id, str) and prompt_id not in prompt_ids:
            prompt_ids.append(prompt_id)
    return prompt_ids


def _effective_config_hash(intent: str, box: str, argv: list[str]) -> str:
    """Stable digest of effective engine argv plus the router configuration inputs."""
    files = {}
    for path in (ROOT / "model_quirks.json", ROOT / "anima_models.json", ROOT / "art_directions.json"):
        if path.is_file():
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"intent": intent, "box": box, "argv": argv, "files": files}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _pick_best(candidates: list[str], subjects: int | None) -> str | None:
    """QC-score the candidates via qc_score.py --best; return the chosen path (None if scoring
    is unavailable, e.g. the imgutils venv is missing). Non-destructive: losers are left in place."""
    cmd = [sys.executable, str(ROOT / "qc_score.py"), "--best", *candidates]
    if subjects is not None:
        cmd += ["--subjects", str(subjects)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)  # surface per-image SKIP lines / all-flagged note
    chosen = proc.stdout.strip()
    return chosen.splitlines()[-1].strip() if (proc.returncode == 0 and chosen) else None


def _ccip_keep(ref: str, candidates: list[str]) -> list[str]:
    """Identity-filter: the candidates that HELD the anchor's identity (consistency_check.py --keep)."""
    cmd = [sys.executable, str(ROOT / "consistency_check.py"), "--ref", ref, "--keep", *candidates]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def _qc_keep(candidates: list[str], subjects: int | None) -> list[str]:
    """Defect pre-filter: candidates with no QC defect flag (qc_score.py --keep). Aesthetic percentile is
    deliberately NOT used to rank the artistic line — scalar aesthetic scorers don't separate 'artistic'
    from 'competent render' (verified: both dbaesthetic and aesthetic-predictor-v2.5 rank a flat baseline
    above the more artistic candidates)."""
    cmd = [sys.executable, str(ROOT / "qc_score.py"), "--keep", *candidates]
    if subjects is not None:
        cmd += ["--subjects", str(subjects)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


# imgutils venv (same default as qc_score.py / consistency_check.py; override via IMGUTILS_VENV_PY).
IMGUTILS_VENV_PY = os.environ.get(
    "IMGUTILS_VENV_PY", "/opt/comfyui-boxa/tools/visual_features/.venv/bin/python")
# wd14 SFW output gate (mirrors daily_gacha). 'sensitive' is allowed — normal clothed anime
# rates high sensitive; only 'explicit'/'questionable' over threshold are rejected.
SFW_EXPLICIT_MAX = 0.15
SFW_QUESTIONABLE_MAX = 0.25


def _sfw_keep(candidates: list[str]) -> list[str]:
    """Content gate for the SFW artistic line: keep candidates whose wd14 rating is SFW
    (explicit < 0.15 and questionable < 0.25). The artistic negative prompt alone does NOT
    guarantee a SFW *output* (a lewd-trigger LoRA or a revealing wildcard slips through), so this
    gates the rendered image. Runs the wd14 op through the imgutils venv (like qc_score /
    consistency_check). Fail-open when the venv/tagger is unavailable (warns) — the negative prompt
    stays the primary defense, and nuking a whole batch because the tagger is missing is worse."""
    if not os.path.exists(IMGUTILS_VENV_PY):
        sys.stderr.write(f"[sfw] imgutils venv not found ({IMGUTILS_VENV_PY}) — SFW gate skipped "
                         f"(set IMGUTILS_VENV_PY); negative prompt is the primary defense\n")
        return candidates
    if not candidates:
        return []
    proc = subprocess.run(
        [IMGUTILS_VENV_PY, str(ROOT / "imgutils_bridge.py"), "wd14", *candidates],
        capture_output=True, text=True,
    )
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if len(candidates) == 1:
        results = {candidates[0]: payload}
    else:
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
    kept = []
    for c in candidates:
        res = results.get(c) if isinstance(results, dict) else None
        if not isinstance(res, dict):
            res = {}
        if not res.get("ok"):
            sys.stderr.write(f"[sfw] wd14 unavailable for {c} — keeping (negative prompt is the primary SFW defense)\n")
            kept.append(c)
            continue
        rating = res.get("rating", {})
        if (rating.get("explicit", 0.0) < SFW_EXPLICIT_MAX
                and rating.get("questionable", 0.0) < SFW_QUESTIONABLE_MAX):
            kept.append(c)
        else:
            sys.stderr.write(f"[sfw] dropped {c} (explicit={rating.get('explicit', 0):.2f} "
                             f"questionable={rating.get('questionable', 0):.2f})\n")
    return kept


def _surface_artistic(candidates: list[str], args: argparse.Namespace) -> dict:
    """Artistic selection: defect-filter (QC flags) [∩ identity if --ref], then surface ALL survivors plus
    a contact-sheet montage for a by-eye / multimodal pick — no auto-crown by a (misleading) scalar score.
    Returns the machine-readable outcome consumed by --json-out."""
    dropped = {"qc": 0, "sfw": 0, "ccip": 0}
    pool = _qc_keep(candidates, args.subjects)
    dropped["qc"] = len(candidates) - len(pool)
    note = f"{len(pool)}/{len(candidates)} passed the defect filter"
    if not pool:
        pool, note = candidates, f"all {len(candidates)} QC-flagged — surfacing them all anyway"
    if getattr(args, "sfw", False):
        safe = _sfw_keep(pool)
        dropped["sfw"] = len(pool) - len(safe)
        note += f"; {len(safe)}/{len(pool)} passed the SFW gate"
        pool = safe
        if not pool:
            print(f"\n[artistic] {note}. Nothing surfaced — all candidates failed the SFW content gate "
                  f"(or wd14 unavailable). Tighten the outfit/scene, or check the imgutils venv.")
            return {"survivors": [], "dropped": dropped, "contact_sheet": None, "note": note}
    if args.ref:
        held = _ccip_keep(args.ref, pool)
        if held:
            dropped["ccip"] = len(pool) - len(held)
            pool, note = held, note + f"; {len(held)} held the anchor identity"
        else:
            note += "; none held identity — showing the defect-survivors"
    print(f"\n[artistic] {note}. Pick by eye — scalar aesthetic scores do NOT rank artistic:")
    for i, p in enumerate(pool):
        print(f"  [{i}] {p}")
    sheet, custom = _contact_sheet_path(getattr(args, "contact_sheet_out", None))
    temp_sheet: Path | None = None
    output = sheet
    try:
        if custom:
            # Keep the temporary output beside the target so the final exclusive create is on the
            # same filesystem.  This avoids replacing a target that appears after the preflight.
            sheet.parent.mkdir(parents=True, exist_ok=True)
            # Keep a PNG suffix: contact_sheet.py delegates format selection to PIL/Image.save.
            fd, temp = tempfile.mkstemp(prefix=f".{sheet.name}.", suffix=".tmp.png", dir=str(sheet.parent))
            os.close(fd)
            temp_sheet = Path(temp)
            output = temp_sheet
        proc = subprocess.run([sys.executable, str(ROOT / "contact_sheet.py"), "--out", str(output), *pool],
                              capture_output=True, text=True)
        sheet_ok = proc.returncode == 0 and proc.stdout.strip()
        if sheet_ok and custom:
            try:
                # Do not use replace/rename here: exclusive-create is the no-clobber seam for a
                # target that another process may have claimed while contact_sheet.py ran.
                with temp_sheet.open("rb") as src, sheet.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            except FileExistsError:
                sys.stderr.write(f"[contact-sheet] custom target became occupied; preserving it: {sheet}\n")
                sheet_ok = False
            except OSError as exc:
                sys.stderr.write(f"[contact-sheet] custom target write failed for {sheet}: {exc}\n")
                sheet_ok = False
        if sheet_ok:
            shown = str(sheet) if custom else proc.stdout.strip()
            print(f"\n[contact-sheet] {shown}")
            print("[pre-screen] optional: a delegated reviewer can grade this sheet per "
                  "docs/visual_review_protocol.md (rank, never crown)")
        elif proc.stderr:
            sys.stderr.write(proc.stderr)
    finally:
        if temp_sheet is not None:
            temp_sheet.unlink(missing_ok=True)
    return {"survivors": pool, "dropped": dropped,
            "contact_sheet": str(sheet) if sheet_ok else None, "note": note}


def _load_pose_entry(slug: str) -> dict:
    """The pose-library index entry for a slug (stdlib JSON read; exits with guidance on a miss)."""
    if not POSE_INDEX.is_file():
        sys.exit(f"[artistic] pose library index not found at {POSE_INDEX} — the outputs/ pose library "
                 f"is not on this machine; drop --pose-ref (or use a text pose / restore the library).")
    data = json.loads(POSE_INDEX.read_text(encoding="utf-8"))
    poses = data.get("poses", []) if isinstance(data, dict) else data
    for e in poses:
        if e.get("slug") == slug:
            return e
    sys.exit(f"[artistic] pose {slug!r} not in the library; available: "
             + ", ".join(sorted(e.get("slug", "?") for e in poses)))


def _load_character_pack(pack_id: str) -> dict:
    """Load a character pack (characters/<id>.pack.json); prompt_identity is required (schema v1.1).

    The pack owns the prompt-surface identity: color_lock (color_hijack injection) and
    neg_guards (character drift + body language). Direction cards must not carry these.
    """
    path = CHARACTERS_ROOT / f"{pack_id}.pack.json"
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in CHARACTERS_ROOT.glob("*.pack.json")))
        sys.exit(f"[artistic] character pack {pack_id!r} not found under characters/ "
                 f"(expected {path.name}); available: {available}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"[artistic] corrupt character pack {path}: {exc}")
    if "prompt_identity" not in data:
        sys.exit(f"[artistic] character pack {pack_id!r} has no prompt_identity "
                 f"(schema v1.1 required — see characters/character.pack.json migration)")
    return data


def _filter_pool_lines(lines: list[str], exclude: set[str]) -> list[str]:
    """Drop pool lines whose inline register metadata intersects the pack's exclude set.

    Lines carry ``text # motion:<class>, register:<tag>`` metadata; the engine strips it
    (wildcard.py). Section comments (leading '#') always survive.
    """
    kept: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            kept.append(line)
            continue
        meta = line.split(" # ", 1)[1] if " # " in line else ""
        tags = {
            part.split(":", 1)[1].strip()
            for part in meta.split(",")
            if part.strip().startswith("register:")
        }
        if tags & exclude:
            continue
        kept.append(line)
    return kept


def _build_filtered_wildcard_dir(pack: dict) -> Path:
    """Copy the wildcard pool to a temp dir with pose/expression filtered by the pack register."""
    source = REPO_ROOT / "wildcards"
    tmp = Path(tempfile.mkdtemp(prefix="loramake-register-"))
    for pool in source.glob("*.txt"):
        shutil.copy2(pool, tmp / pool.name)
    exclude = set((pack.get("prompt_identity") or {}).get("register", {}).get("exclude") or [])
    for name in ("pose", "expression"):
        path = tmp / f"{name}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(_filter_pool_lines(lines, exclude)) + "\n", encoding="utf-8")
    return tmp


def _pose_prompt_text(entry: dict) -> str:
    """The pose text a --pose-ref may inject into the prompt: `prompt_hint` ONLY. `desc` is
    human-facing (status notes, variant provenance, re-extract TODOs) and must never reach the
    prompt — injecting it once leaked '[PARTIAL — ...]' maintenance notes into gen metadata."""
    return (entry.get("prompt_hint") or "").strip()


def _pair_group(options: list) -> str:
    """Pairing suggestions -> a `{a|b}` wildcard group so the slot varies WITHIN the fitting set per
    seed (reuses the engine's existing syntax; zero engine change). Falls back to the first literal
    when an option itself contains wildcard syntax characters."""
    opts = [o for o in options if isinstance(o, str) and o.strip()]
    if not opts:
        return ""
    if len(opts) == 1 or any(ch in o for o in opts for ch in "{}|"):
        return opts[0]
    return "{" + "|".join(opts) + "}"


def _build_artistic(args: argparse.Namespace, spec: dict, extra: list[str],
                    resource_plan: _mr.ResourcePlan | None = None) -> tuple[dict[str, str], list[str]]:
    """Artistic intent: art_prompt assembles the anti-slop prompt body + negative + aspect + sampler; we
    route it to boxa_quality_gen (illustrious --max) to reuse the full quality stack (FreeU/PAG/detailers).
    Character LoRA(s) via --lora; style LoRA(s) via --style-lora (golden 0.6); optional --ref identity lock.

    Slot priority (left wins): explicit per-slot flag > pose pairing injection (--pose-ref, off with
    --no-pair) > --direction card slots > __wildcard__ default. A pose_ref injects the verified
    openpose ControlNet recipe and forces --res to the skeleton canvas."""
    if args.prompt is None:
        sys.exit("'artistic' needs --prompt (the subject/character tags, e.g. '1girl, solo, silver hair').")

    family = spec.get("resource_family")
    if family and args.lora:
        if resource_plan is not None:
            args.lora = _resolved_for_box(resource_plan.lora_traces, args.box)
        else:
            try:
                args.lora, _ = _mr.lora_trace_for_box(args.lora, args.box, family)
            except ValueError as exc:
                sys.exit(f"[resource] {exc}")
    if family and args.style_lora:
        if resource_plan is not None:
            args.style_lora = _resolved_for_box(resource_plan.style_lora_traces, args.box)
        else:
            try:
                args.style_lora, _ = _mr.lora_trace_for_box(args.style_lora, args.box, family)
            except ValueError as exc:
                sys.exit(f"[resource] {exc}")

    card = None
    if args.direction:
        card = art_directions.get(args.direction)
        if card is None:
            sys.exit(f"[artistic] unknown direction card {args.direction!r}; run `gen.py --directions`.")
        print(f"[direction] {card['slug']} · {card['name_cn']} — {card['intent']}")

    style_frag = None
    if getattr(args, "style", None):
        from loramake import style_registry as sr
        style_id = args.style
        try:
            if style_id == "random":
                import combo
                style_id = combo.draw_aesthetic(args.seed if args.seed is not None else 1, ART_LINE)
            style_frag = sr.resolve_aesthetic(style_id, ART_LINE, density=getattr(args, "density", None),
                                              box=args.box)
        except (sr.StyleRegistryError, ValueError) as exc:
            sys.exit(f"[style] {exc}")
        print(f"[style] {style_id} -> {style_frag.style_id} on {ART_LINE}: {style_frag.realization_id} "
              f"({style_frag.kind}, tier={style_frag.tier}; {style_frag.note})")
        if style_frag.pos_prefix:
            args.prompt = f"{style_frag.pos_prefix}, {args.prompt}"
    if getattr(args, "composition", None) == "random":
        import combo
        draw = combo.draw_composition(args.seed if args.seed is not None else 1, ART_LINE)
        args.composition = draw["text"]
        print(f"[composition] random -> {draw['text']} (shot={draw['shot']} angle={draw['angle']} "
              f"placement={draw['placement']})")

    pack = _load_character_pack(getattr(args, "character", None)) if getattr(args, "character", None) else None
    register_dir: Path | None = None
    combo_draw: dict | None = None
    if card and "color_hijack" in (card.get("flags") or []):
        if pack is None:
            sys.exit(f"[artistic] direction card {card['slug']!r} declares flags:color_hijack — pass "
                     f"--character <pack id> so the assembler can inject the pack's color_lock "
                     f"(no pack = no silent bare run)")
        color_lock = (pack.get("prompt_identity") or {}).get("color_lock") or []
        if color_lock:
            args.prompt = f"{args.prompt}, {', '.join(color_lock)}"
            print(f"[identity] {card['slug']} color_hijack -> color_lock injected "
                  f"(pack {pack['id']}): {', '.join(color_lock)}")
    if pack:
        exclude = (pack.get("prompt_identity") or {}).get("register", {}).get("exclude") or []
        if exclude:
            register_dir = _build_filtered_wildcard_dir(pack)
            print(f"[identity] register filter active (pack {pack['id']}, exclude={sorted(exclude)})")

    if getattr(args, "outfit", None) or getattr(args, "action", None) or getattr(args, "scene", False):
        # N4 combo engine: outfit x action (x scene) seeded sampling. Any combo flag
        # activates the whole draw; unspecified dimensions are random. Needs a pack for
        # the register filter (and the SFW line context).
        import combo
        if pack is None:
            sys.exit("[artistic] --outfit/--action/--scene need --character <pack id> "
                     "(register filter + SFW line context)")
        exclude = (pack.get("prompt_identity") or {}).get("register", {}).get("exclude") or []
        combo_seed = args.seed if args.seed is not None else 1
        combo_draw = combo.sample(
            combo_seed,
            outfit_id=None if args.outfit == "random" else args.outfit,
            action_id=None if args.action == "random" else args.action,
            scene=bool(args.scene), sfw_only=True, exclude_register=set(exclude))
        print(f"[combo] seed={combo_seed} outfit={combo_draw['outfit_id']} "
              f"action={combo_draw['action_id']} ({combo_draw['motion_class']})"
              + (f" bg={combo_draw['scene']['background']} "
                 f"tod={combo_draw['scene']['time_of_day']}" if combo_draw["scene"] else ""))
        if combo_draw["outfit_id"]:
            import outfit_cards as oc
            outfit_tags = ", ".join(oc.get(combo_draw["outfit_id"])["tags"])
            args.prompt = f"{args.prompt}, {outfit_tags}"
            print(f"[combo] outfit tags injected: {outfit_tags[:90]}…")
        if combo_draw["action_text"]:
            args.pose = combo_draw["action_text"]
            print(f"[combo] pose pinned: {combo_draw['action_text'][:90]}…")
        if combo_draw["scene"]:
            scene_text = f"{combo_draw['scene']['time_of_day']}, {combo_draw['scene']['background']}"
            args.prompt = f"{args.prompt}, {scene_text}"
            print(f"[combo] scene injected: {scene_text}")

    user_set = {k for k in ART_SLOT_KEYS if getattr(args, k) is not None}

    pose_slug = args.pose_ref  # N3: cards no longer pin a skeleton; only an explicit --pose-ref locks
    entry = _load_pose_entry(pose_slug) if pose_slug else None

    if entry and not args.no_pair:  # pairing: only into slots the user left open
        pairing = entry.get("pairing") or {}
        for key in PAIR_SLOTS:
            if key in user_set or not pairing.get(key):
                continue
            group = _pair_group(pairing[key])
            if group:
                setattr(args, key, group)
                print(f"[pair] {key} <- {group}   (pose {pose_slug})")
    if card:  # card slots fill whatever is still open
        for key in ART_SLOT_KEYS:
            if getattr(args, key) is None and key in card["slots"]:
                setattr(args, key, card["slots"][key])
    if entry and args.pose is None:
        hint = _pose_prompt_text(entry)
        if hint:
            args.pose = hint  # text prompt corroborates the skeleton instead of fighting it
        else:
            print(f"[pose] {pose_slug} has no prompt_hint — pose text slot left to its default "
                  f"(backfill: pose_library.py annotate {pose_slug} --prompt-hint '...')")
    if card:
        if card.get("sfw") and not args.sfw:
            args.sfw = True
        if card.get("detail") and not args.detail:
            args.detail = True
    aspect = args.aspect or (card or {}).get("aspect") or art_prompt.DEFAULT_ASPECT

    try:
        r = art_prompt.build(args.prompt, style_mode=args.style_mode, artist=args.artist,
                             medium=args.medium, camera=args.camera, composition=args.composition,
                             lighting=args.lighting, pose=args.pose, color_mood=args.color_mood,
                             expression=args.expression, aspect=aspect, detail=args.detail)
    except ValueError as exc:
        sys.exit(f"[artistic] {exc}")

    res = f"{r.width}x{r.height}"
    if entry:  # hard-won: --res must match the skeleton canvas or the control map stretches
        dims = entry.get("dims") or [768, 1344]
        res = POSE_RES_SQUARE if dims[0] == dims[1] else POSE_RES_TALL
        print(f"[pose] --res {res} forced to match the skeleton canvas ({dims[0]}x{dims[1]}); "
              f"the aspect flag is ignored for pose-locked gen")

    env = {"BOXA_HOST": _effective_host(args.box)}
    argv = [sys.executable, str(ROOT / spec["script"]), *spec["fixed"],
            "--prompt", r.positive, "--neg", r.negative, "--res", res,
            "--label", "artistic"]
    if r.sampler:
        argv += ["--sampler", r.sampler]
    if args.n is not None:
        argv += ["--n", str(args.n)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    for lora in args.lora or []:               # character LoRA(s), engine syntax FILE[:STR]
        argv += ["--lora", lora]
    # style LoRA(s): the user's set wins as a WHOLE; otherwise the card's (never mixed). Golden 0.6.
    if args.style_lora and resource_plan is not None:
        style_loras = _resolved_for_box(resource_plan.style_lora_traces, args.box)
    else:
        if args.style_lora:
            style_loras = list(args.style_lora)
        elif style_frag and style_frag.lora_specs:
            style_loras = list(style_frag.lora_specs)  # --style realization; never mixed with a card's
        else:
            style_loras = list((card or {}).get("style_loras") or [])
        if family and style_loras:
            try:
                style_loras, _ = _mr.lora_trace_for_box(style_loras, args.box, family)
            except ValueError as exc:
                sys.exit(f"[resource] {exc}")
    for sl in style_loras:
        argv += ["--lora", sl if ":" in sl else f"{sl}:0.6"]
    if register_dir is not None:
        argv += ["--wildcard-dir", str(register_dir)]
    # base: an explicit router --base (quirks-card validated, box-registered) wins outright;
    # otherwise boxb keeps forcing its house ckpt, and on boxa the card's primary base becomes
    # --base (resolved via its quirks card) unless the caller already passed one after `--`.
    resolved_base, base_card = _resolve_base(args, spec)
    chosen_base = resolved_base
    if chosen_base is None and args.box == "boxb":
        chosen_base = BOXB_CKPT  # flag itself is appended after passthrough, see below
        if card and card.get("base", {}).get("primary") not in (None, "iLustMix"):
            print(f"[direction] --box boxb forces {BOXB_CKPT}; card base {card['base']['primary']!r} skipped")
    elif chosen_base is None and "--base" in extra:
        i = extra.index("--base")
        chosen_base = extra[i + 1] if i + 1 < len(extra) else None
    elif chosen_base is None and style_frag and style_frag.base_pin:
        mq = model_quirks.card(style_frag.base_pin)
        if mq and args.box in (mq.get("boxes") or []):
            chosen_base = mq.get("ckpt") or style_frag.base_pin
            argv += ["--base", chosen_base]
            base_card = mq
            print(f"[style] base pinned by realization: {chosen_base}")
        else:
            print(f"[style] base_pin {style_frag.base_pin!r} not registered on {args.box}; engine default kept")
    elif chosen_base is None and card:
        primary = card.get("base", {}).get("primary")
        mq = model_quirks.card(primary) if primary else None
        if mq:
            chosen_base = mq.get("ckpt") or primary
            argv += ["--base", chosen_base]
    if entry and (chosen_base is None or "ilustmix" not in chosen_base.lower()):
        print("[warn] openpose pose-lock is only verified on iLustMix — this base will likely IGNORE "
              "the skeleton (BOXA's Illustrious does). Prefer --box boxb for pose-locked gen.")

    if entry:  # the verified pose-lock recipe (skeleton pre-made -> no preprocess, full strength)
        argv += ["--control", "openpose", "--control-image", str(OUTPUTS_ROOT.parent / entry["file"]),
                 "--no-preprocess", "--control-strength", "1.0", "--control-end", "1.0"]
        if "--detailers" not in extra:
            argv += ["--detailers", "face"]
        if args.box == "boxb":  # BOXB's only face detector (mirrors the regional --refine quirk)
            env["BOXA_FACE_DETECTOR"] = BOXB_FACE_DETECTOR

    # neg_add: merge the direction card's appendix, the --base quirks card's appendix, and any
    # user `-- --neg-add ...` into ONE flag — the engine's --neg-add is single-valued (last
    # wins); two occurrences would silently drop the card's.
    neg_parts = [p for p in [(card or {}).get("neg_add", ""),
                             (style_frag.neg_add if style_frag else ""),
                             (base_card or {}).get("neg_add", "")] if p]
    if (base_card or {}).get("no_freeu") and "--no-freeu" not in extra and "--no-freeu" not in argv:
        argv.append("--no-freeu")
    if pack:  # plan N1 layered negatives: pack neg_guards (character drift + body language) first
        guards = (pack.get("prompt_identity") or {}).get("neg_guards") or []
        if guards:
            neg_parts.insert(0, ", ".join(guards))
            print(f"[identity] pack {pack['id']} neg_guards layered: {', '.join(guards)}")
    while "--neg-add" in extra:
        i = extra.index("--neg-add")
        if i + 1 >= len(extra):
            sys.exit("[artistic] `--neg-add` after `--` is missing its value")
        neg_parts.append(extra[i + 1])
        del extra[i:i + 2]
    if neg_parts:
        argv += ["--neg-add", ", ".join(dict.fromkeys(neg_parts))]

    for f in (card or {}).get("flags") or []:  # card-carried engine flags (e.g. --no-freeu for Harem)
        if f == "color_hijack":
            continue  # mechanism metadata consumed by the assembler, never an engine flag
        if f not in extra and f not in argv:
            argv.append(f)

    if args.ref:                               # optional identity lock (IPAdapter face) for a known character
        argv += ["--ip", "face", "--ip-image", args.ref]
    argv += extra
    # explicit router --base (validated, box-registered) or the boxb house ckpt goes after the
    # passthrough so a verbatim `-- --base` cannot smuggle a BOXA-only checkpoint (last-wins).
    if resolved_base:
        argv += ["--base", resolved_base]
    elif args.box == "boxb":
        argv += ["--base", BOXB_CKPT]
    fx = (card or {}).get("post_fx") or {}
    if fx.get("preset") not in (None, "", "none"):
        # Auto-applied by the --run path (see --no-fx); carried via env so build()'s
        # signature stays untouched for callers that treat it as env/argv only.
        env["LORAMAKE_POST_FX_PRESET"] = fx["preset"]
        if fx.get("overrides"):
            env["LORAMAKE_POST_FX_OVERRIDES"] = fx["overrides"]
    return env, argv


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _route_gates(box: str, host: str, args: argparse.Namespace) -> list[str]:
    return route_plan.route_gates(box, host, args, shared_gaming_host=_is_shared_gaming_host)


def _route_post_steps(intent: str, args: argparse.Namespace) -> list[str]:
    return route_plan.route_post_steps(intent, args)


def _resource_plan_to_fields(plan: _mr.ResourcePlan, box: str) -> dict:
    """Convert a compiled _mr.ResourcePlan into compact, box-selected route plan fields."""
    return route_plan.resource_plan_to_fields(plan, box)


def build_route_plan(intent: str, box: str, box_reason: str, env: dict[str, str],
                     argv: list[str], args: argparse.Namespace,
                     readiness: dict | None = None,
                     resource_plan: _mr.ResourcePlan | None = None) -> dict:
    """Machine-readable route plan: everything a fresh agent needs to fire or explain the route.

    Facade over ``loramake.route_plan.build_route_plan``: the engine-table entry, this tree's
    engine directory and outputs root, the effective host, the config hash, the timestamp and
    the shared-gaming-host predicate are bound here at call time.
    """
    spec = ENGINES[intent]
    host = env.get("BOXA_HOST") or _effective_host(box)
    return route_plan.build_route_plan(
        intent, box, box_reason, env, argv, args, readiness, resource_plan,
        spec=spec,
        engine_dir=ROOT,
        outputs_root=OUTPUTS_ROOT,
        host=host,
        config_hash=_effective_config_hash(intent, box, argv),
        generated_at=_utc_now(),
        schema=ROUTE_PLAN_SCHEMA,
        shared_gaming_host=_is_shared_gaming_host,
    )


def _print_route_plan(plan: dict) -> None:
    print(f"[route] {plan['intent']} -> {plan['engine']} on {plan['box']}", file=sys.stderr)
    print(f"[route] box: {plan['box_reason']}", file=sys.stderr)
    for gate in plan["pre_run_gates"]:
        print(f"[route] pre-run: {gate}", file=sys.stderr)
    for step in plan["post_run"]:
        print(f"[route] post-run: {step}", file=sys.stderr)


def _run_route_mode(args: argparse.Namespace, extra: list[str]) -> int:
    """Print-only route planner: validate, resolve, optionally probe, never launch."""
    resource_plan = None
    if _is_resource_tier(args.intent):
        resource_plan = _compile_resources_for_args(args, extra)
        if resource_plan.problems:
            sys.exit(f"[resource] {'; '.join(resource_plan.problems)}")
    args.box, box_reason = resolve_box(
        args.box, args.intent, use_cache=True, probe_timeout=PLAN_BOX_PROBE_TIMEOUT_S,
        eligible=resource_plan.eligible_boxes if resource_plan is not None else None,
    )
    env, argv = build(args.intent, args, extra, resource_plan)
    host = env.get("BOXA_HOST") or _effective_host(args.box)
    readiness = None
    if args.probe:
        ready, reason = probe_idle(host, timeout=PROBE_ONESHOT_TIMEOUT_S)
        readiness = {"ready": ready, "reason": reason, "host": host}
    plan = build_route_plan(args.intent, args.box, box_reason, env, argv, args, readiness,
                            resource_plan=resource_plan)
    if args.route_json:
        target = Path(args.route_json).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[route-json] {target}", file=sys.stderr)
    if args.route:
        _print_route_plan(plan)
    if args.probe:
        if readiness and readiness["ready"]:
            print(f"[probe] GO: {readiness['reason']}", file=sys.stderr)
        else:
            reason = (readiness or {}).get("reason", "probe unavailable")
            print(f"[probe] HOLD: {reason} (print-only; no GPU work was submitted)", file=sys.stderr)
            print(plan["command"])
            return 3
    print(plan["command"])
    return 0


_EXTERNAL_ROUTER_FLAGS = {
    "--provider", "--prompt", "--n", "--seed", "--box", "--run", "--json-out",
    "--confirm-paid-generation",
}


def _run_external_provider(args: argparse.Namespace, extra: list[str], router_argv: list[str]) -> int:
    """Run a paid provider behind the same gen.py JSON/result boundary as local engines."""
    started = time.monotonic()
    result = {
        "schema": "loramake.gen-result/v2",
        "attempt_id": os.environ.get("LORAMAKE_ATTEMPT_ID"),
        "intent": args.intent,
        "box": None,
        "provider": args.provider,
        "provider_status": "planned",
        "provider_request_id": None,
        "rc": 0,
        "new_files": [],
        "images": [],
        "survivors": None,
        "dropped": None,
        "best": None,
        "contact_sheet": None,
        "note": None,
        "prompt_ids": [],
        "free_ok": None,
        "timings": {"readiness_s": 0.0, "engine_s": 0.0, "free_s": 0.0,
                    "postprocess_s": 0.0, "total_s": 0.0},
    }

    def finish(rc: int) -> int:
        result["rc"] = rc
        result["timings"]["total_s"] = round(time.monotonic() - started, 3)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[json-out] {args.json_out}")
        return rc

    try:
        if args.intent != "best":
            raise image_providers.ProviderError(
                "provider_input", "external providers currently support the best tier only"
            )
        unsupported = sorted({
            token for token in router_argv
            if token.startswith("--") and token not in _EXTERNAL_ROUTER_FLAGS
        })
        if unsupported:
            raise image_providers.ProviderError(
                "provider_input",
                f"external provider does not support router flags: {', '.join(unsupported)}",
            )
        options = image_providers.parse_passthrough(args.provider, extra)
        request_plan = image_providers.plan(
            args.provider, args.prompt, n=args.n, seed=args.seed, options=options,
        )
        result["request_summary"] = request_plan
        result["config_hash"] = _effective_config_hash(
            args.intent,
            f"provider:{args.provider}",
            [json.dumps(request_plan, ensure_ascii=False, sort_keys=True)],
        )
        print("[provider-plan] " + json.dumps(request_plan, ensure_ascii=False, sort_keys=True))
        if not args.run:
            return 0
        if not args.confirm_paid_generation:
            raise image_providers.ProviderError(
                "provider_confirmation_required",
                "paid generation requires --confirm-paid-generation for this request",
            )
        engine_started = time.monotonic()
        outcome = image_providers.run_provider(
            args.provider,
            args.prompt,
            n=args.n,
            seed=args.seed,
            options=options,
            confirmed=True,
        )
        result["timings"]["engine_s"] = round(time.monotonic() - engine_started, 3)
        result.update(outcome)
        request_id = outcome.get("provider_request_id")
        result["prompt_ids"] = [request_id] if request_id else []
        result["note"] = f"{args.provider} generation succeeded"
        return finish(0)
    except image_providers.ProviderError as exc:
        result["provider_status"] = exc.provider_status
        result["provider_request_id"] = exc.request_id
        result["prompt_ids"] = [exc.request_id] if exc.request_id else []
        result["error_class"] = exc.error_class
        result["error"] = {
            "class": exc.error_class,
            "message": str(exc),
            "http_status": exc.http_status,
        }
        result["note"] = str(exc)
        print(f"[provider-error] {exc.error_class}: {exc}", file=sys.stderr)
        return finish(69)


DEAI_LINES = {"draft": "anima", "promote": "anima",
              "best": "illustrious", "artistic": "illustrious", "sequence": "illustrious",
              "variants": "illustrious", "regional": "illustrious"}
DEAI_NEG_FLAGS = ("--neg", "--neg-add")


def _argv_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv[:-1]) if tok == flag]


def _deai_report(intent: str, argv: list[str]) -> None:
    """Print deai_lint findings for the prompt the engine will receive to stderr. Advisory only:
    never raises, never changes the exit code, never touches stdout or --json-out."""
    line = DEAI_LINES.get(intent)
    prompt_flag = ENGINES.get(intent, {}).get("prompt")
    if not line or not prompt_flag or not (ROOT / "deai_lint.py").is_file():
        return  # the NSFW mirror does not carry deai_lint
    try:
        import deai_lint  # sibling; imported lazily so a lint problem cannot break gen.py
        prompts = _argv_values(argv, prompt_flag)
        if not prompts:
            return
        negative = ", ".join(v for flag in DEAI_NEG_FLAGS for v in _argv_values(argv, flag))
        findings = deai_lint.lint_prompt(line, prompts[-1], "", negative,
                                         deai_lint.load_rules(deai_lint.DEFAULT_RULES))
        if not findings:
            return
        errors = sum(f["severity"] == "error" for f in findings)
        print(f"[deai] {intent} ({line} line): {len(findings)} finding(s), {errors} error(s) — "
              f"advisory, render not blocked", file=sys.stderr)
        for f in findings:
            evidence = f" (evidence: {f['evidence']})" if f.get("evidence") else ""
            print(f"[deai]   {f['severity']} {f['rule']}: {f['message']}{evidence}", file=sys.stderr)
        print("[deai] method: docs/DEAI_CREATIVE_METHOD.md; silence with --no-deai-report", file=sys.stderr)
    except Exception as exc:  # advisory: a lint failure must never affect the render
        print(f"[deai] lint skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> None:
    raw = sys.argv[1:]
    extra = raw[raw.index("--") + 1:] if "--" in raw else []
    router_argv = raw[:raw.index("--")] if "--" in raw else raw

    ap = argparse.ArgumentParser(
        description="Intent -> engine router for loramake image generation. "
                    "Pick an intent; engine-specific flags go after `--`.")
    ap.add_argument("intent", nargs="?", choices=tuple(ENGINES), help="what you want to generate")
    ap.add_argument("--list", action="store_true", help="print the intent->engine decision tree and exit")
    ap.add_argument("--list-json", action="store_true",
                    help="print the route catalog as machine-readable JSON and exit")
    ap.add_argument("--route", action="store_true",
                    help="print a concise route plan (intent/engine/box/gates/post-run) to stderr, "
                         "then the exact command to stdout; print-only")
    ap.add_argument("--route-json", default=None, metavar="PATH",
                    help="write a machine-readable route plan to PATH and print the exact command; "
                         "print-only")
    ap.add_argument("--probe", action="store_true",
                    help="route plan only: add a one-shot GET readiness probe (queue/VRAM) and exit "
                         "non-zero on HOLD; never launches the engine or touches /free")
    ap.add_argument("--directions", action="store_true",
                    help="print the art direction cards (coherent slot bundles for artistic --direction) and exit")
    ap.add_argument("--box", choices=("auto", *HOSTS), default="auto",
                    help="which GPU box; auto (default) quick-scans BOXA first: if BOXA looks "
                         "gaming/busy route to BOXB immediately, otherwise use BOXA")
    ap.add_argument("--provider", choices=image_providers.PROVIDERS, default="local",
                    help="execution channel: local ComfyUI (default), civitai, or gpt_image")
    ap.add_argument("--confirm-paid-generation", action="store_true",
                    help="external providers only: explicit per-request paid generation confirmation")
    ap.add_argument("--base", default=None,
                    help="override the checkpoint via a model_quirks card stem (see `model_quirks.py "
                         "list`); the card must register the chosen --box. Not for draft/promote")
    ap.add_argument("--prompt", default=None, help="prompt (the global/scene base prompt for 'regional')")
    ap.add_argument("--ir", default=None, metavar="PATH",
                    help="optional Prompt IR JSON (compose first). Missing IR is lint/warn this "
                         "release; LORAMAKE_REQUIRE_IR=1 is not fail-closed")
    ap.add_argument("--task-json", default=None,
                    help="structured Pilot intents only: strict TaskSpec JSON (no prompt rewriting)")
    ap.add_argument("--source", default=None,
                    help="source image for 'reproduce' / draft image for 'promote' / "
                         "the picture being changed for 'revise'")
    ap.add_argument("--revision-json", default=None,
                    help="'revise' only: the freeze contract as JSON (changed_dimensions, "
                         "frozen_dimensions, minimal_delta, rung, scope, and the region mask "
                         "when the change repaints pixels)")
    ap.add_argument("--ref", default=None, help="identity-anchor (face) image for 'sequence'")
    ap.add_argument("--no-free", action="store_true",
                    help="skip the automatic POST /free after --run (e.g. when chaining runs "
                         "back-to-back and the model reload cost matters)")
    ap.add_argument("--free-now", action="store_true",
                    help="--run: release the box immediately instead of leasing it. The default "
                         "keeps the models warm and frees after LORAMAKE_VRAM_IDLE_S (300s) idle")
    ap.add_argument("--ccip-keep", action="store_true",
                    help="'sequence' only: hard-filter the result list to outputs that HELD the anchor "
                         "identity (CCIP), like the artistic line. Non-destructive: drifted files stay "
                         "on disk, they just leave survivors/. Default keeps the report-only behaviour.")
    ap.add_argument("--n", type=int, default=None, help="how many images / candidates")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lora", action="append", default=None, help="LoRA spec (repeatable), engine syntax")
    ap.add_argument("--refine", action="store_true", help="regional: chain fine-detail refine stage")
    ap.add_argument("--run", action="store_true", help="execute the command (default: just print it)")
    ap.add_argument("--json-out", default=None, metavar="PATH",
                    help="--run: also write a machine-readable result JSON (new files, gate stats, "
                         "survivors/best, contact sheet) to PATH — for callers that must not parse stdout")
    ap.add_argument("--no-deai-report", action="store_true",
                    help="skip the advisory deai_lint report (stderr) printed before the command; "
                         "for batch callers. The report never blocks or changes the exit code")
    ap.add_argument("--contact-sheet-out", default=None, metavar="PATH",
                    help="artistic only: contact-sheet output path (must not already exist)")
    ap.add_argument("--no-wait", action="store_true",
                    help="--run: skip the idle probe and fire immediately (default waits for an idle box)")
    ap.add_argument("--no-fx", action="store_true",
                    help="--run: skip the automatic post_fx finishing pass (raw outputs only)")
    ap.add_argument("--pick-best", action="store_true",
                    help="--run: QC-score the generated batch and print the single best-of-N path")
    ap.add_argument("--subjects", type=int, default=None,
                    help="--pick-best: expected subject count -> flag dropped/duplicated subjects")
    art = ap.add_argument_group("artistic intent (anti-slop single-char line)")
    art.add_argument("--style-mode", choices=("lora", "artist", "none"), default="lora",
                     help="lora (style LoRA carries the look; default) | artist (by-name tags) | none")
    art.add_argument("--artist", default=None, help="--style-mode artist: 'by ke-ta:1.0, by wlop:0.5'")
    art.add_argument("--style-lora", action="append", default=None,
                     help="style-role LoRA FILE[:STR] (repeatable; default strength 0.6)")
    art.add_argument("--aspect", default=None,
                     help="portrait|tall|square|landscape|wide|hi-portrait (default: the --direction "
                          "card's aspect, else portrait; ignored when a pose skeleton fixes --res)")
    art.add_argument("--detail", action="store_true", help="richer-detail sampler (dpmpp_3m_sde)")
    art.add_argument("--sfw", action="store_true",
                     help="gate outputs through the wd14 SFW content filter (drops explicit/questionable; "
                          "'sensitive' allowed) — the negative prompt alone doesn't guarantee a SFW image")
    art.add_argument("--medium", default=None, help="fix medium (default __medium__ wildcard; 'none' drops)")
    art.add_argument("--camera", default=None, help="fix framing (default __camera__ wildcard)")
    art.add_argument("--composition", default=None,
                     help="fix figure-vs-frame composition (default __composition__ wildcard; 'none' drops)")
    art.add_argument("--lighting", default=None, help="fix lighting (default __lighting__ wildcard)")
    art.add_argument("--pose", default=None, help="fix pose (default __pose__ wildcard)")
    art.add_argument("--color-mood", dest="color_mood", default=None,
                     help="fix palette mood (default __color_mood__ wildcard)")
    art.add_argument("--expression", default=None,
                     help="opt-in slot: omit = no expression segment; 'vary' = __expression__ per image; "
                          "a literal pins it")
    art.add_argument("--style", default=None,
                     help="aesthetic register or style card id (catalogs/style/registers.json, "
                          "catalogs/style/*.json) resolved on the artistic line; 'random' draws a "
                          "verified register with the seed. Prefix + LoRA + base pin + neg are "
                          "injected; explicit --style-lora / --base still win")
    art.add_argument("--density", default=None,
                     help="--style strength: light|medium|strong or 淡|中|浓 (default medium)")
    art.add_argument("--direction", default=None,
                     help="apply an art direction card (coherent medium/camera/composition/lighting/pose/"
                          "palette bundle; see `gen.py --directions`). Explicit per-slot flags override "
                          "the card; unpinned slots still roll their wildcard")
    art.add_argument("--character", default=None,
                     help="character pack id (characters/<id>.pack.json, schema v1.1): layers the pack's "
                          "neg_guards into the negative and — for cards declaring flags:color_hijack — "
                          "injects the pack's color_lock into the subject. A color_hijack card without "
                          "--character is an error (no silent bare run)")
    art.add_argument("--outfit", default=None,
                     help="N4 combo: pin an outfit card id or 'random' (activates seeded "
                          "outfit x action sampling; requires --character)")
    art.add_argument("--action", default=None,
                     help="N4 combo: pin a text-action id from wildcards/pose.txt or 'random'")
    art.add_argument("--scene", action="store_true",
                     help="N4 combo: also draw background + time-of-day pool entries into the prompt")
    art.add_argument("--pose-ref", default=None,
                     help="lock the pose to a pose-library skeleton slug (injects the verified openpose "
                          "ControlNet recipe + forces --res to the skeleton canvas; overrides a card's pose_ref)")
    art.add_argument("--no-pair", action="store_true",
                     help="--pose-ref: do NOT auto-fill open slots from the skeleton's pairing metadata")
    args = ap.parse_args(router_argv)

    route_mode = bool(args.route or args.route_json or args.probe)
    if args.directions:
        art_directions.cmd_list(args)
        return
    if args.list_json:
        print_list(as_json=True)
        return
    if args.list or not args.intent:
        print_list()
        return
    applied = apply_optional_ir(ir=args.ir, prompt=args.prompt, intent=args.intent)
    emit_ir_warnings(applied["warnings"])
    if applied["prompt"] is not None:
        args.prompt = applied["prompt"]
    if route_mode and args.run:
        sys.exit("--route/--route-json/--probe are print-only; drop --run to plan, then re-run with --run.")
    if route_mode and args.json_out:
        sys.exit("--json-out is a --run result file; use --route-json PATH for a route plan.")
    if route_mode and args.provider != "local":
        sys.exit("--route/--route-json/--probe currently support --provider local only; "
                 "for a paid provider run gen.py <intent> --provider <name> without --run to see its plan.")
    if args.contact_sheet_out is not None:
        if args.intent != "artistic":
            sys.exit("--contact-sheet-out only applies to 'artistic'.")
        raw_sheet = Path(args.contact_sheet_out).expanduser()
        absolute_sheet = raw_sheet if raw_sheet.is_absolute() else Path.cwd() / raw_sheet
        # lexists is intentional: a dangling symlink (or any other dirent) is still an occupied
        # custom target and must be rejected before build/idle/GPU work.  Do not apply this guard to
        # the historical fixed default path, which retains its overwrite behaviour.
        if os.path.lexists(absolute_sheet):
            sys.exit(f"--contact-sheet-out path already exists: {args.contact_sheet_out}")
        args.contact_sheet_out = str(absolute_sheet.resolve(strict=False))
    if args.pick_best and not args.run:
        sys.exit("--pick-best requires --run (it selects among freshly generated images).")
    if args.pick_best and args.intent == "sequence":
        sys.exit("'sequence' already runs a CCIP identity-consistency report; --pick-best does not apply.")
    if args.pick_best and args.intent == "gacha2.0":
        sys.exit("'gacha2.0' already runs a fail-closed wd14+qc+aesthetic gate; --pick-best does not apply.")
    if args.json_out and args.intent == "gacha2.0" and not args.run:
        sys.exit("--json-out is a --run result file; add --run so gacha2.0 can write the delivery winners.")
    if args.ccip_keep and args.intent != "sequence":
        sys.exit("--ccip-keep only applies to 'sequence' (the artistic line filters via --ref already).")

    if args.provider != "local":
        rc = _run_external_provider(args, extra, router_argv)
        if rc:
            raise SystemExit(rc)
        return
    if args.confirm_paid_generation:
        sys.exit("--confirm-paid-generation only applies to external providers.")

    if route_mode:
        rc = _run_route_mode(args, extra)
        if rc:
            raise SystemExit(rc)
        return

    # Compile resource constraints before resolving the box so auto/legacy policies respect
    # family, LoRA availability, base-card registration and pose-ref.
    resource_plan = None
    if _is_resource_tier(args.intent):
        resource_plan = _compile_resources_for_args(args, extra)
        if resource_plan.problems:
            sys.exit(f"[resource] {'; '.join(resource_plan.problems)}")

    # Plan with a short cross-process cache; --run always re-probes fresh.
    args.box, box_reason = resolve_box(
        args.box, args.intent,
        use_cache=not args.run,
        probe_timeout=PLAN_BOX_PROBE_TIMEOUT_S if not args.run else RUN_BOX_PROBE_TIMEOUT_S,
        eligible=resource_plan.eligible_boxes if resource_plan is not None else None,
    )
    print(f"[box] {args.box} — {box_reason}", file=sys.stderr)
    env, argv = build(args.intent, args, extra, resource_plan)
    if not args.no_deai_report:
        _deai_report(args.intent, argv)
    line = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + " ".join(shlex.quote(a) for a in argv)
    print(line.strip())
    if not args.run:
        return

    if os.environ.get("COMFY_WS_WAIT") is None:
        env["COMFY_WS_WAIT"] = "1"

    fx_preset = env.get("LORAMAKE_POST_FX_PRESET") or POST_FX_DEFAULT
    fx_overrides = env.get("LORAMAKE_POST_FX_OVERRIDES", "")
    fx_off = (args.no_fx or os.environ.get("LORAMAKE_POST_FX", "1") == "0"
              or str(fx_preset).lower() in ("none", "off"))
    if fx_off:
        env["LORAMAKE_POST_FX"] = "0"  # propagates --no-fx into engines that auto-apply it

    total_started = time.monotonic()
    readiness_started = time.monotonic()
    host = env.get("BOXA_HOST") or _effective_host(args.box)
    if not args.no_wait:  # never fire onto a busy / gaming GPU (shared box)
        fresh_auto_idle = (args.box == "boxa" and _AUTO_BOXA_IDLE_AT is not None
                           and time.monotonic() - _AUTO_BOXA_IDLE_AT < _AUTO_PROBE_REUSE_S)
        if fresh_auto_idle:
            print(f"  {args.box} idle (auto-box scan verified it moments ago)")
        else:
            verdict = wait_for_idle(host)
            ok, reason = verdict
            if is_unreachable(verdict):
                print(f"[abort] {args.box} unreachable: {reason} — stop the batch, the route is down",
                      file=sys.stderr)
                raise SystemExit(EXIT_BOX_UNREACHABLE)
            if not ok:
                sys.exit(f"[abort] {args.box} not idle: {reason} (use --no-wait to override)")
            print(f"  {args.box} {reason}")
    readiness_s = round(time.monotonic() - readiness_started, 3)
    # The router owns the post-run VRAM release decision (including --no-free); engines that
    # also release on their own honor this marker so a shared box is not released twice per run.
    env["LORAMAKE_ROUTER_MANAGES_FREE"] = "1"
    # Off the LAN only the webp preview crosses the link; the original stays on the box and its
    # location is recorded in the image meta (remote_original). An explicit env setting wins.
    if "LORAMAKE_FETCH_PREVIEW" not in os.environ and net_routes.is_remote(args.box):
        env["LORAMAKE_FETCH_PREVIEW"] = args.box

    select = args.pick_best                  # best/variants: aesthetic best-of-N crown
    surface = args.intent == "artistic"      # artistic: defect/identity filter + contact sheet for by-eye pick
    report = args.intent == "sequence"       # sequence: CCIP identity HOLD/DRIFT report
    post = select or surface or report or bool(args.json_out)
    # App jobs own an attempt id and require the precise engine manifest. The CLI adapter keeps the
    # historical output-tree snapshot fallback for older/direct engines.
    app_manifest_required = bool(os.environ.get("LORAMAKE_ATTEMPT_ID")) and args.intent in {
        "draft", "best", "artistic", "sequence", "anima_structured_pilot", "sdxl_structured_pilot",
        "forge", "forge_repro", "gacha2.0",
        "revise",
    }
    files_out: Path | None = None
    if post:  # per-run engine manifest (GEN_FILES_OUT protocol) — see _collect_run_files
        fd, tmp = tempfile.mkstemp(prefix="gen_files_", suffix=".txt")
        os.close(fd)
        files_out = Path(tmp)
        env["GEN_FILES_OUT"] = str(files_out)
    before = _png_snapshot() if post and not app_manifest_required else set()
    print(f"\n[run] {args.intent} on {args.box} ...", flush=True)
    engine_started = time.monotonic()
    rc = subprocess.run(argv, env={**os.environ, **env}).returncode
    engine_s = round(time.monotonic() - engine_started, 3)

    result = {"schema": "loramake.gen-result/v2",
              "attempt_id": os.environ.get("LORAMAKE_ATTEMPT_ID"),
              "intent": args.intent, "box": args.box, "rc": rc, "new_files": [],
              "survivors": None, "dropped": None, "best": None, "contact_sheet": None,
              "note": None, "prompt_ids": [],
              "config_hash": _effective_config_hash(args.intent, args.box, argv),
              "free_ok": None, "free": None,
              "timings": {"readiness_s": readiness_s, "engine_s": engine_s,
                          "free_s": 0.0, "postprocess_s": 0.0, "total_s": 0.0}}
    post_started = time.monotonic()

    def _run_files() -> list[str]:
        return _collect_run_files(
            files_out, before, allow_snapshot_fallback=not app_manifest_required,
        )

    def _emit(code_or_msg=None):
        result["prompt_ids"] = _prompt_ids_from_events(os.environ.get("GEN_EVENTS_OUT"))
        result["timings"]["postprocess_s"] = round(time.monotonic() - post_started, 3)
        if not args.no_free:
            free_started = time.monotonic()
            free = _release_or_lease(host, args.box, free_now=args.free_now)
            result["free_ok"] = free.pop("free_ok", None)
            result["free"] = free
            result["timings"]["free_s"] = round(time.monotonic() - free_started, 3)
        result["timings"]["total_s"] = round(time.monotonic() - total_started, 3)
        if files_out is not None:  # every post-run path exits through here — reap the manifest
            files_out.unlink(missing_ok=True)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[json-out] {args.json_out}")
        if code_or_msg is not None:
            sys.exit(code_or_msg)

    if not (select or surface or report):
        result["new_files"] = _run_files() if args.json_out else []
        if app_manifest_required and rc == 0 and not result["new_files"]:
            result["rc"] = 74
            result["note"] = "engine produced no GEN_FILES_OUT entries; output scan disabled for app jobs"
            _emit("[manifest] app engine reported no PNGs through GEN_FILES_OUT")
        if not fx_off:
            result["new_files"] += _auto_fx(result["new_files"], fx_preset, fx_overrides)
        _emit(rc)
        return
    if rc != 0:
        result["note"] = f"engine exited {rc}; post-run analysis skipped"
        _emit(f"[{args.intent}] engine exited {rc}; skipping post-run analysis.")

    candidates = _run_files()
    result["new_files"] = candidates
    if not fx_off:
        result["new_files"] += _auto_fx(candidates, fx_preset, fx_overrides)
    if not candidates:
        if app_manifest_required:
            result["rc"] = 74
            result["note"] = "engine produced no GEN_FILES_OUT entries; output scan disabled for app jobs"
            _emit(f"[{args.intent}] engine reported no PNGs through GEN_FILES_OUT")
        result["note"] = "no new PNGs under outputs/"
        _emit(f"[{args.intent}] no new PNGs under outputs/ — nothing to analyse (did the engine save?).")

    if surface:  # artistic: filter defects/identity, then surface survivors for a by-eye pick
        result.update(_surface_artistic(candidates, args))
        _emit()
        return
    if select:
        best = _pick_best(candidates, args.subjects)
        if best is None:
            result["note"] = "pick-best scored no candidates (imgutils venv missing?)"
            _emit(f"[pick-best] scored none of {len(candidates)} candidate(s) — imgutils venv missing? "
                  f"set IMGUTILS_VENV_PY. Candidates left at: {OUTPUTS_ROOT}")
        result["best"] = best
        print(f"\n[best-of-{len(candidates)}] (aesthetic) -> {best}")
        _emit()
        return
    # sequence: report which outputs held the anchor identity (CCIP) vs which drifted — non-destructive
    if args.ccip_keep:
        held = _ccip_keep(args.ref, candidates)
        result["survivors"] = held
        result["dropped"] = {"qc": 0, "sfw": 0, "ccip": len(candidates) - len(held)}
        result["note"] = f"{len(held)}/{len(candidates)} held the anchor identity (drifted files stay on disk)"
        print(f"\n[consistency] {result['note']}:")
        for p in held:
            print(f"  {p}")
        _emit()
        return
    print(f"\n[consistency] {len(candidates)} output(s) vs anchor {args.ref}:")
    subprocess.run([sys.executable, str(ROOT / "consistency_check.py"), "--ref", args.ref, *candidates])
    result["note"] = f"CCIP report printed vs anchor {args.ref}"
    _emit()


if __name__ == "__main__":
    main()
