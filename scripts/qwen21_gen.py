#!/usr/bin/env python3
"""Qwen-Image-2.1 text-to-image and reference-image edit on the BOXA sidecar ComfyUI (BOXA :8189, via 127.0.0.1:18189).

The default invocation is a local PLAN: it prints the exact jobs and output paths and performs no
network request. ``--run`` asserts the sidecar runtime contract, preflights the three fixed loader
assets, waits for the shared GPU to be idle, submits the jobs and saves the returned PNGs.

Recipe: ``configs/qwen21/recipes.json`` names one t2i graph and one edit graph per recipe; the loader
names (unet / clip / vae) and the sampler defaults are read from the graph, never duplicated here
(notes: comfyui-boxa/comfyui-workflows/qwen21_t2i_notes.md, qwen21_edit_notes.md). Graphs are loaded and
patched, never rebuilt.

t2i: the canvas comes from ``--ratio/--size`` (validated table) or ``--res``.
edit (``--ref IMAGE``): the reference is normalized (EXIF, RGB, first frame, longest side capped),
uploaded into the sidecar's own ``input/`` and spliced in through ``TextEncodeQwenImage21``; the node
sizes the canvas from the reference's aspect at about ``resolution``² pixels, so no canvas flag applies.

Prompt dialect is natural language. At cfg 1.0 ComfyUI skips the unconditional pass, so a negative
prompt is only accepted together with ``--cfg > 1``.

Concurrency boundary: the idle probe goes to production ``:8188`` (same GPU, same gaming guard and
VRAM headroom rule as every other BOXA line) and the sidecar queue must be empty. When production itself
is unreachable, the sidecar's own VRAM reading stands in for the gaming guard. That is a cooperative
check inside an explicit probe window, not a mutual exclusion between the two instances.

Output is rooted at this repository: ``outputs/qwen21/by_model/<DRIVING_MODEL>/qwen21_gen``
(``LORAMAKE_OUTPUTS`` moves the root — Studio channel jobs render into their own tree). PNGs carry a
``loramake`` iTXt record and a ``.meta.json`` sidecar; same-seed files get a ``__vN`` suffix instead of
being overwritten. ``--keep-into DIR`` moves each PNG + sidecar into DIR with the same no-clobber rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import PROBE_VRAM, probe_idle, wait_for_idle  # noqa: E402  # shared-box idle policy, probed on production :8188
from krea2_gen import (  # noqa: E402  # sibling natural-language line: output claims + argparse converters
    _fast_delay,
    _filename_token,
    _nonnegative_seed,
    _positive_finite_float,
    _positive_int,
    claim_output_path,
    model_slug,
    no_clobber_path,
)
from loramake.comfy import QueueStateError, _request_json, cancel_prompt, looks_like_webp, observe_queue, upload_image  # noqa: E402
from loramake.qwen21_canvas import CANVAS, DEFAULT_RATIO, DEFAULT_SIZE, RATIOS, SIZES  # noqa: E402
from loramake.qwen21_endpoints import resolve_qwen21_host  # noqa: E402
from boxa_comfy import (  # noqa: E402
    BOXA_HOST,
    free_vram,
    looks_like_png,
    preflight_assets,
    save_image_with_meta,
    submit_and_fetch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_PATH = REPO_ROOT / "configs" / "qwen21" / "recipes.json"
GRAPH_PATH = REPO_ROOT / "comfyui-boxa" / "comfyui-workflows" / "qwen21_t2i_api.json"
EDIT_GRAPH_PATH = REPO_ROOT / "comfyui-boxa" / "comfyui-workflows" / "qwen21_edit_api.json"
DEFAULT_MODEL = "claude"

# Tokens that must appear in the sidecar's argv (from /system_stats) before anything is submitted.
RUNTIME_ARGV_REQUIRED = ("8189", "--use-ck-attention")
GRID = 16  # QwenImage21.spacial_downscale_ratio
RES_MIN, RES_MAX = 512, 2048
# Edit mode: TextEncodeQwenImage21.resolution rescales every reference to ~resolution² pixels on a 32
# grid. 1024 (the node default) renders full-frame noise once a VAE is attached (ComfyUI issue #16435).
EDIT_RES_GRID = 32
EDIT_RES_MIN, EDIT_RES_MAX = 512, 2048
EDIT_RES_NOISE_TRAP = 1024
REF_MAX_SIDE = 2048  # staged reference cap; the encoder rescales to ~resolution² anyway
REF_STAGING = "_refs"  # <outputs root>/qwen21/_refs/<sha256[:16]>.png
UNREACHABLE_REASON = "system_stats unreachable"  # gen.probe_idle's verdict on a dead production instance
# One production probe per render; a dead :8188 must not cost the 8 s launch budget (over a remote link a
# live one answers in ~1.5 s, on the LAN in 20 ms).
IDLE_PROBE_TIMEOUT_S = 4.0
# Sidecar fallback threshold (production unreachable). Models now stay resident, which leaves ~45% free at
# rest, so gen.py's 40% would flag our own weights as a game. ponytail: 35% catches any game holding
# >= 1.6 GB; a canvas-aware budget if 2K renders ever start OOM-ing behind a small game.
SIDECAR_IDLE_VRAM = 0.35
PNG_SIG = b"\x89PNG\r\n\x1a\n"

NODE = {"unet": "1", "clip": "2", "vae": "3", "encode": "4", "latent": "5",
        "sampler": "6", "decode": "7", "save": "8", "load_image": "9"}
EXPECTED_CLASSES = {
    "1": "UNETLoader", "2": "CLIPLoader", "3": "VAELoader", "4": "TextEncodeQwenImage21",
    "5": "EmptyLatentImage", "6": "KSampler", "7": "VAEDecode", "8": "SaveImage",
}
EXPECTED_CLASSES_EDIT = {
    "1": "UNETLoader", "2": "CLIPLoader", "3": "VAELoader", "4": "TextEncodeQwenImage21",
    "6": "KSampler", "7": "VAEDecode", "8": "SaveImage", "9": "LoadImage",
}
REF_SLOT = "images.image_1"  # autogrow slot: the dotted key is the only API-format spelling that lands


def qwen21_root() -> Path:
    # LORAMAKE_OUTPUTS moves the write root (Studio channel jobs render into their own tree);
    # relative overrides anchor to the repo root, same rule as loramake.config._path_from_env.
    raw = os.environ.get("LORAMAKE_OUTPUTS")
    root = Path(raw).expanduser() if raw else REPO_ROOT / "outputs"
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / "qwen21"


def output_dir_for_model(value: str) -> Path:
    return qwen21_root() / "by_model" / model_slug(value) / "qwen21_gen"


def load_recipes(path: Path = RECIPES_PATH) -> dict:
    """The recipe table; refuses a file without a ``recipes`` map and a ``default`` inside it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    recipes = data.get("recipes")
    if not isinstance(recipes, dict) or not recipes or data.get("default") not in recipes:
        raise ValueError(f"recipes file needs a 'recipes' table and a 'default' naming one of them: {path}")
    return data


def resolve_recipe(name: str | None, data: dict) -> tuple[str, dict]:
    chosen = name or data["default"]
    recipe = data["recipes"].get(chosen)
    if not isinstance(recipe, dict):
        raise ValueError(f"recipe {chosen!r} is not one of: {', '.join(data['recipes'])}")
    for key in ("graph_t2i", "graph_edit", "edit_resolution"):
        if key not in recipe:
            raise ValueError(f"recipe {chosen!r} lacks {key!r}")
    return chosen, recipe


def recipe_graph_path(recipe: dict, key: str) -> Path:
    """Recipe graph paths are repository-relative, anchored on the recipes file (REPO_ROOT is patchable)."""
    path = Path(recipe[key])
    return path if path.is_absolute() else RECIPES_PATH.parents[2] / path


def graph_models(graph: dict) -> dict:
    """Loader names as the checked-in graph pins them: the recipe truth for preflight and metadata."""
    return {
        "unet": graph[NODE["unet"]]["inputs"]["unet_name"],
        "clip": graph[NODE["clip"]]["inputs"]["clip_name"],
        "clip_type": graph[NODE["clip"]]["inputs"]["type"],
        "vae": graph[NODE["vae"]]["inputs"]["vae_name"],
    }


def graph_sampling(graph: dict) -> dict:
    """Sampler defaults from node 6; ``--steps/--cfg`` override them per run."""
    sampler = graph[NODE["sampler"]]["inputs"]
    return {"steps": int(sampler["steps"]), "cfg": float(sampler["cfg"]),
            "sampler": sampler["sampler_name"], "scheduler": sampler["scheduler"]}


def load_graph(path: Path = GRAPH_PATH, expected: dict = EXPECTED_CLASSES) -> dict:
    """Load a checked-in API graph and refuse any shape drift.

    ``expected`` is the class table of the mode (t2i or edit). The edit table also pins the wiring the
    patcher relies on, because a class check alone cannot see a dropped link.
    """
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    for node_id, class_type in expected.items():
        found = (graph.get(node_id) or {}).get("class_type")
        if found != class_type:
            raise ValueError(f"graph node {node_id} is {found!r}, expected {class_type!r}: {path}")
    try:
        graph_models(graph)
        graph_sampling(graph)
    except KeyError as exc:
        raise ValueError(f"graph lacks input {exc}: {path}") from None
    if NODE["load_image"] in expected:
        encode = graph[NODE["encode"]]["inputs"]
        if graph[NODE["sampler"]]["inputs"].get("latent_image") != [NODE["encode"], 2]:
            raise ValueError(f"edit graph node 6 latent_image must come from node 4 output 2: {path}")
        if "vae" not in encode or encode.get(REF_SLOT) != [NODE["load_image"], 0]:
            raise ValueError(f"edit graph node 4 needs 'vae' and '{REF_SLOT}': [\"9\", 0]: {path}")
    return graph


def resolve_canvas(ratio: str, size: str) -> tuple[int, int]:
    try:
        return CANVAS[(ratio, size)]
    except KeyError:
        supported = ", ".join(f"{r}/{s}" for r, s in CANVAS)
        raise ValueError(f"canvas {ratio}/{size} is not validated; supported: {supported}") from None


def parse_res(value: str) -> tuple[int, int]:
    """Argparse converter: explicit WIDTHxHEIGHT on the 16 grid, within the validated range."""
    parts = value.strip().lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution dimensions must be integers") from exc
    if any(dim < RES_MIN or dim > RES_MAX for dim in (width, height)):
        raise argparse.ArgumentTypeError(f"resolution must be {RES_MIN}-{RES_MAX} pixels per side")
    if width % GRID or height % GRID:
        raise argparse.ArgumentTypeError(
            f"resolution must be a multiple of {GRID} (the model renders off-grid sizes silently at another size)"
        )
    return width, height


def validate_resolution(value: int) -> int:
    """Edit-mode ``TextEncodeQwenImage21.resolution``: 32 grid inside the validated band; 1024 is the noise trap."""
    if value == EDIT_RES_NOISE_TRAP:
        raise ValueError("resolution 1024 renders full-frame noise with reference latents "
                         "(ComfyUI issue #16435); use 1056")
    if value < EDIT_RES_MIN or value > EDIT_RES_MAX or value % EDIT_RES_GRID:
        raise ValueError(f"resolution must be {EDIT_RES_MIN}-{EDIT_RES_MAX} on the {EDIT_RES_GRID} grid "
                         "(0 = native size is not validated on this line yet)")
    return value


def _edit_resolution_arg(value: str) -> int:
    try:
        return validate_resolution(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def normalize_ref(path: Path, staging: Path) -> Path:
    """Stage one reference as an RGB PNG that the sidecar's LoadImage reads the way we see it.

    First frame only, EXIF orientation applied, alpha dropped, longest side capped at REF_MAX_SIDE (the
    encoder rescales to ~resolution² anyway, so nothing is lost and the upload stays small). The staged
    file is named by the source's content hash, so re-editing the same file reuses one upload.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as opened:
            opened.seek(0)
            image = ImageOps.exif_transpose(opened) or opened
            image = image.convert("RGB")
            if max(image.size) > REF_MAX_SIDE:
                image.thumbnail((REF_MAX_SIDE, REF_MAX_SIDE), Image.LANCZOS)
            staging.mkdir(parents=True, exist_ok=True)
            staged = staging / f"{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}.png"
            image.save(staged, "PNG")
    except Exception as exc:  # decoder errors are many-typed; this is the trust boundary for user files
        raise ValueError(f"reference image unreadable ({type(exc).__name__}: {exc}); "
                         "PNG / JPEG / WebP / GIF are supported, HEIC is not") from exc
    return staged


def upload_ref(staged: Path, host: str) -> str:
    """Upload the staged reference into the sidecar's input/ and return the LoadImage name."""
    if os.environ.get("LORAMAKE_SKIP_UPLOAD") == "1":
        raise RuntimeError("LORAMAKE_SKIP_UPLOAD=1 would skip uploading a freshly staged reference; "
                           "unset it for edit runs")
    name = upload_image(str(staged), host)
    if not name:
        raise RuntimeError(f"sidecar {host} did not accept the reference upload {staged.name}")
    return name


def png_dims(raw: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR: in edit mode the node decides the canvas, we only read it back."""
    if raw[:8] != PNG_SIG or raw[12:16] != b"IHDR":
        raise RuntimeError("fetched output is not a PNG with an IHDR header")
    width, height = struct.unpack(">II", raw[16:24])
    return int(width), int(height)


def image_dims(raw: bytes) -> tuple[int, int]:
    """Width and height of a fetched PNG or WebP (the webp preview keeps the render's pixel size)."""
    if raw[:8] == PNG_SIG:
        return png_dims(raw)
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(raw)) as img:
        return int(img.size[0]), int(img.size[1])


# /view?preview=webp;80 makes ComfyUI re-encode the output server-side: 94 KB instead of a 1.7 MB PNG
# (2026-09-22, 768x768). Over a remote link that is the difference between 0.5 s and minutes; the PNG
# stays in the sidecar's output/ under the name recorded as source_image.
FETCH_FORMATS = {"png": (".png", None, looks_like_png), "webp": (".webp", {"preview": "webp;80"}, looks_like_webp)}


def patch_graph(graph: dict, *, prompt: str, negative: str, seed: int, steps: int, cfg: float,
                width: int | None = None, height: int | None = None, prefix: str,
                ref_name: str | None = None, edit_resolution: int | None = None) -> dict:
    """Return a patched deep copy.

    t2i: node 5 carries the canvas; node 4's ``resolution`` is left alone (a no-op without references).
    edit (``ref_name`` set): node 9 loads the uploaded reference and node 4's ``resolution`` sizes the
    canvas from the reference's aspect; that graph has no node 5.
    LoRA hook (not wired): a ``LoraLoaderModelOnly`` chain would sit between node 1 and node 6's
    ``model`` input, as in krea2_gen.build_graph; only a non-convrot recipe can carry it.
    """
    g = copy.deepcopy(graph)
    encode = g[NODE["encode"]]["inputs"]
    encode["prompt"] = prompt
    encode["negative_prompt"] = negative
    if ref_name is None:
        g[NODE["latent"]]["inputs"]["width"] = width
        g[NODE["latent"]]["inputs"]["height"] = height
    else:
        g[NODE["load_image"]]["inputs"]["image"] = ref_name
        encode["resolution"] = edit_resolution
    sampler = g[NODE["sampler"]]["inputs"]
    sampler["seed"] = seed
    sampler["steps"] = steps
    sampler["cfg"] = cfg
    g[NODE["save"]]["inputs"]["filename_prefix"] = prefix
    return g


def fetch_system_stats(host: str) -> dict:
    return _request_json(host, "/system_stats", timeout=20.0)


def runtime_problems(stats: dict) -> list[str]:
    """Contract check against /system_stats: the answering instance must be the ck sidecar."""
    argv = [str(item) for item in ((stats.get("system") or {}).get("argv") or [])]
    return [f"argv lacks {token!r}" for token in RUNTIME_ARGV_REQUIRED if token not in argv]


def runtime_record(stats: dict) -> dict:
    system = stats.get("system") or {}
    return {
        "comfyui_version": system.get("comfyui_version"),
        "pytorch_version": system.get("pytorch_version"),
        "argv": [str(item) for item in (system.get("argv") or [])],
    }


def _quiet(future) -> dict | None:
    """The future's value, or None when it failed (the caller re-raises that failure in order)."""
    try:
        return future.result()
    except Exception:
        return None


def shared_gpu_guard(idle_host: str, qwen_host: str, stats: dict | None = None) -> tuple[bool, str]:
    """Production's idle policy first; when production itself is unreachable, the sidecar's VRAM stands in.

    Both instances share the one GPU, so a game or a live-but-busy production job shows up as low free
    VRAM on the sidecar too. ``stats`` is the runtime probe's /system_stats, seconds old, reused so the
    remote link is not paid twice; without it the sidecar is fetched here.
    A reachable-but-busy production keeps the ordinary wait loop.
    """
    ok, reason = probe_idle(idle_host, timeout=IDLE_PROBE_TIMEOUT_S)
    if ok:
        return True, reason
    if reason != UNREACHABLE_REASON:
        return wait_for_idle(idle_host)
    try:
        stats = stats if stats is not None else fetch_system_stats(qwen_host)
    except QueueStateError as exc:
        return False, f"production {reason}; sidecar stats failed too: {exc}"
    device = (stats.get("devices") or [{}])[0]
    total, free = device.get("vram_total") or 0, device.get("vram_free") or 0
    if total <= 0:
        return False, f"production {reason}; sidecar reports no VRAM total"
    ratio = free / total
    figures = f"{ratio:.0%} free ({free // 1048576}MB/{total // 1048576}MB)"
    if ratio < SIDECAR_IDLE_VRAM:
        return False, f"production {reason}; sidecar VRAM only {figures}"
    return True, f"production {reason}; sidecar VRAM {figures}"


def sidecar_queue_empty(host: str) -> tuple[bool, str]:
    try:
        seen = observe_queue(host, timeout=10.0)
    except QueueStateError as exc:
        return False, f"sidecar queue unreachable: {exc}"
    running, pending = len(seen["running"]), len(seen["pending"])
    if running or pending:
        return False, f"sidecar queue busy (running={running}, pending={pending})"
    return True, "sidecar queue empty"


def output_path(outdir: Path, *, label: str, steps: int, canvas: str, index: int, seed: int,
                edit: bool = False, ext: str = ".png") -> Path:
    label_token = _filename_token(label, fallback="") if label else ""
    tag = f"_{label_token}" if label_token else ""
    mode = "_edit" if edit else ""
    return outdir / f"qwen21{mode}{tag}_{steps}st_{canvas}_{index:02d}_{seed}{ext}"


def keep_into(png_path: Path, into: Path) -> Path:
    """Move one PNG + its sidecar into ``into`` as a group, never overwriting; returns the new PNG path."""
    into.mkdir(parents=True, exist_ok=True)
    target, _revision = no_clobber_path(into / png_path.name)
    src_meta = Path(str(png_path) + ".meta.json")
    dst_meta = Path(str(target) + ".meta.json")
    shutil.move(str(png_path), str(target))
    if src_meta.is_file():
        try:
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict):
            meta["kept_from"] = str(png_path)
            meta["kept_into"] = str(target)
            dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            src_meta.unlink()
        else:
            shutil.move(str(src_meta), str(dst_meta))
    return target


def _abort_remote(host: str, prompt_id: str | None) -> dict | None:
    """Best-effort cancel of our own prompt; the client leaving never counts as the remote stopping."""
    if not prompt_id:
        return None
    try:
        return cancel_prompt(host, prompt_id, timeout=20.0)
    except QueueStateError as exc:
        return {"ok": False, "action": "unreachable", "prompt_id": prompt_id, "reason": str(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview or run Qwen-Image-2.1 t2i / reference edit on the BOXA sidecar")
    parser.add_argument("--prompt", required=True, help="natural-language prompt (no tag scaffold is added)")
    parser.add_argument("--neg", default="", help="negative prompt; only accepted with --cfg > 1")
    parser.add_argument("--n", type=_positive_int, default=1, help="number of images")
    parser.add_argument("--seed", type=_nonnegative_seed, default=None, help="base seed (random if omitted)")
    parser.add_argument("--ratio", choices=RATIOS, default=None, help=f"t2i aspect ratio (validated table; default {DEFAULT_RATIO})")
    parser.add_argument("--size", choices=SIZES, default=None, help=f"t2i size tier (validated table; default {DEFAULT_SIZE})")
    parser.add_argument("--res", type=parse_res, default=None, metavar="WIDTHxHEIGHT",
                        help=f"t2i explicit canvas, {RES_MIN}-{RES_MAX} on the {GRID} grid; overrides --ratio/--size")
    parser.add_argument("--ref", default=None, metavar="IMAGE",
                        help="edit mode: reference image (any PIL-readable file); the canvas follows its aspect")
    parser.add_argument("--resolution", type=_edit_resolution_arg, default=None,
                        help=f"edit mode: TextEncodeQwenImage21 resolution, {EDIT_RES_MIN}-{EDIT_RES_MAX} on the "
                             f"{EDIT_RES_GRID} grid, never {EDIT_RES_NOISE_TRAP} (default: recipe edit_resolution)")
    parser.add_argument("--steps", type=_positive_int, default=None, help="sampling steps (default: the graph's, 16)")
    parser.add_argument("--cfg", type=_positive_finite_float, default=None, help="CFG (default: the graph's, 1.0)")
    parser.add_argument("--recipe", default=None, help="recipe id from configs/qwen21/recipes.json (default: its 'default')")
    parser.add_argument("--graph", default=None, help="API graph file overriding the recipe's graph for this mode")
    parser.add_argument("--box", choices=("boxa",), default="boxa", help="only the BOXA sidecar exists for this line")
    parser.add_argument("--model", type=model_slug, default=os.environ.get("DRIVING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--label", default="", help="extra token in the output filename")
    parser.add_argument("--keep-into", default=None, metavar="DIR", help="move each PNG + sidecar into DIR (no clobber)")
    parser.add_argument("--fetch", choices=sorted(FETCH_FORMATS), default="png",
                        help="what to download: the full PNG (default) or ComfyUI's webp;80 preview "
                             "(~1/18 the bytes; use on a remote link, the PNG stays on the sidecar)")
    parser.add_argument("--json-out", default=None, metavar="FILE", help="write a result JSON (new_files, runs)")
    parser.add_argument("--run", action="store_true", help="submit (default: local preview only)")
    parser.add_argument("--no-wait", action="store_true", help="--run: skip the shared-GPU idle guard")
    # Models stay loaded between renders (user, 2026-09-22): release them when switching models or
    # walking away, with `freeboxa`, not after every image. --no-free is kept as a no-op for old callers.
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--free", action="store_true", help="--run: POST /free on the sidecar afterwards")
    cleanup.add_argument("--no-free", action="store_true", help=argparse.SUPPRESS)
    return parser


def _write_result(path: str | None, payload: dict) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    edit = args.ref is not None
    if edit and (args.ratio or args.size or args.res):
        parser.error("--ref takes the canvas from the reference image; drop --ratio/--size/--res")
    if not edit and args.resolution is not None:
        parser.error("--resolution only applies together with --ref")
    ref_path = Path(args.ref).expanduser() if edit else None
    if ref_path is not None and not ref_path.is_file():
        parser.error(f"--ref image not found: {args.ref}")
    try:
        recipe_name, recipe = resolve_recipe(args.recipe, load_recipes())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"recipes unusable: {exc}")
    graph_path = Path(args.graph) if args.graph else recipe_graph_path(recipe, "graph_edit" if edit else "graph_t2i")
    try:
        graph = load_graph(graph_path, EXPECTED_CLASSES_EDIT if edit else EXPECTED_CLASSES)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"graph unusable: {exc}")
    models, sampling = graph_models(graph), graph_sampling(graph)
    steps = args.steps if args.steps is not None else sampling["steps"]
    cfg = args.cfg if args.cfg is not None else sampling["cfg"]
    if args.neg.strip() and cfg <= 1.0:
        parser.error("--neg needs --cfg > 1: at cfg 1.0 ComfyUI skips the unconditional pass and the "
                     "negative prompt is ignored")
    width = height = None
    edit_resolution = None
    if edit:
        try:
            edit_resolution = validate_resolution(int(args.resolution if args.resolution is not None
                                                      else recipe["edit_resolution"]))
        except (TypeError, ValueError) as exc:
            parser.error(f"edit resolution: {exc}")
    else:
        try:
            width, height = args.res if args.res else resolve_canvas(args.ratio or DEFAULT_RATIO,
                                                                     args.size or DEFAULT_SIZE)
        except ValueError as exc:
            parser.error(str(exc))
    canvas = "auto" if edit else f"{width}x{height}"

    fetch_ext, view_params, validate_fetched = FETCH_FORMATS[args.fetch]
    base_seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2_000_000_000
    if base_seed + args.n - 1 >= 2**64:
        parser.error("base seed plus --n exceeds ComfyUI's maximum seed")
    outdir = output_dir_for_model(args.model)
    keep_dir = Path(args.keep_into).expanduser() if args.keep_into else None
    jobs = []
    for index in range(args.n):
        seed = base_seed + index
        base_path = output_path(outdir, label=args.label, steps=steps, canvas=canvas, index=index, seed=seed,
                                edit=edit, ext=fetch_ext)
        planned, revision = no_clobber_path(base_path)
        jobs.append({"index": index, "seed": seed, "base_path": base_path, "planned_path": planned,
                     "revision": revision})

    qwen_host = resolve_qwen21_host(args.box, explicit=os.environ.get("QWEN21_HOST"))
    idle_host = BOXA_HOST
    mode = "RUN" if args.run else "PLAN"
    print(f"qwen21-gen {mode} {'edit' if edit else 't2i'} recipe={recipe_name} unet={models['unet']} "
          f"clip={models['clip']} vae={models['vae']} steps={steps} cfg={cfg} "
          f"{sampling['sampler']}/{sampling['scheduler']} canvas={canvas}"
          + (f" (from ref @ {edit_resolution}) ref={ref_path}" if edit else "")
          + f" model={args.model} n={args.n} sidecar={qwen_host} idle_probe={idle_host}")
    for job in jobs:
        print(f"  [{job['index']:02d} seed={job['seed']}] -> {job['planned_path']}"
              + (f" (keep-into {keep_dir})" if keep_dir else ""))
    if not args.run:
        print("preview only: no network request or file write; pass --run to execute")
        return 0

    # Network begins here. 1) runtime contract, 2) fixed assets, 3) reference upload, 4) shared-GPU guard,
    # 5) sidecar queue. All five are independent reads (plus the upload), so they go out together: over a
    # remote link every request costs 1-2 s of round trip and the serial chain used to be the biggest fixed
    # cost. Verdicts are still read in the documented order so the abort messages do not change.
    def _stage_ref() -> tuple[Path, str]:
        staged = normalize_ref(ref_path, qwen21_root() / REF_STAGING)
        return staged, upload_ref(staged, qwen_host)

    with ThreadPoolExecutor(max_workers=5) as pool:
        stats_f = pool.submit(fetch_system_stats, qwen_host)
        assets = pool.submit(preflight_assets, "qwen21", models["unet"], [], clips=[models["clip"]],
                             vaes=[models["vae"]], host=qwen_host)
        ref = pool.submit(_stage_ref) if edit else None
        guard = (pool.submit(lambda: shared_gpu_guard(idle_host, qwen_host, _quiet(stats_f)))
                 if not args.no_wait else None)
        queue = pool.submit(sidecar_queue_empty, qwen_host)
    try:
        stats = stats_f.result()
    except QueueStateError as exc:
        print(f"[abort] sidecar unreachable at {qwen_host}: {exc}", file=sys.stderr)
        return 2
    problems = runtime_problems(stats)
    if problems:
        print(f"[abort] sidecar runtime contract mismatch: {'; '.join(problems)}", file=sys.stderr)
        return 2
    runtime = runtime_record(stats)
    print(f"  sidecar {runtime['comfyui_version']} torch {runtime['pytorch_version']} argv ok")
    assets.result()
    ref_name = None
    if ref is not None:
        try:
            staged, ref_name = ref.result()
        except (ValueError, RuntimeError) as exc:
            print(f"[abort] reference: {exc}", file=sys.stderr)
            return 2
        print(f"  ref {ref_path} -> {staged.name} -> sidecar input {ref_name}")
    if guard is not None:
        ok, reason = guard.result()
        if not ok:
            print(f"[abort] shared GPU not idle: {reason} (use --no-wait to override)", file=sys.stderr)
            return 2
        print(f"  {idle_host} {reason}")
    ok, reason = queue.result()
    if not ok:
        print(f"[abort] {reason}", file=sys.stderr)
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    runs: list[dict] = []
    failures = 0
    interrupted = False
    cleanup_ok = True
    total = 0.0
    try:
        for job in jobs:
            started = time.monotonic()
            prompt_id: dict[str, str | None] = {"id": None}

            def _sink(record: dict, _slot=prompt_id) -> None:
                if record.get("event") == "prompt_submitted":
                    _slot["id"] = record.get("prompt_id")

            try:
                patched = patch_graph(graph, prompt=args.prompt, negative=args.neg, seed=job["seed"],
                                      steps=steps, cfg=cfg, width=width, height=height,
                                      prefix="qwen21_edit" if edit else "qwen21_gen",
                                      ref_name=ref_name, edit_resolution=edit_resolution)
                results = submit_and_fetch(patched, label=f"qwen21 {job['index']:02d}", host=qwen_host,
                                           delay=_fast_delay, validate=validate_fetched, event_sink=_sink,
                                           view_params=view_params)
                if not results:
                    remote = _abort_remote(qwen_host, prompt_id["id"])
                    raise RuntimeError(f"no valid {args.fetch} returned (remote cleanup: {remote})")
                raw, history_item = results[0]
                if not validate_fetched(raw):
                    raise RuntimeError(f"fetched output is empty or not a valid {args.fetch}")
                elapsed = time.monotonic() - started
                out_w, out_h = image_dims(raw) if edit else (width, height)
                base_path = job["base_path"]
                if edit:  # the node chose the canvas; name the file after what came back
                    base_path = output_path(outdir, label=args.label, steps=steps, canvas=f"{out_w}x{out_h}",
                                            index=job["index"], seed=job["seed"], edit=True, ext=fetch_ext)
                while True:
                    with claim_output_path(base_path) as claim:
                        destination, revision = claim.path, claim.version
                        metadata = {
                            "schema_version": "loramake.qwen21_gen.v1",
                            "engine": "qwen21_gen",
                            "gen_engine": "comfyui:boxa-sidecar/qwen21",
                            "line": "qwen21_gen",
                            "mode": "edit" if edit else "t2i",
                            "recipe": recipe_name,
                            "box": args.box,
                            "host": qwen_host,
                            "idle_probe_host": idle_host,
                            "runtime": runtime,
                            "model": models["unet"],
                            "clip": models["clip"],
                            "clip_type": models["clip_type"],
                            "vae": models["vae"],
                            "seed": job["seed"],
                            "steps": steps,
                            "cfg": cfg,
                            "sampler": sampling["sampler"],
                            "scheduler": sampling["scheduler"],
                            "prompt": args.prompt,
                            "negative": args.neg,
                            "resolution": f"{out_w}x{out_h}",
                            "ratio": None if (edit or args.res) else (args.ratio or DEFAULT_RATIO),
                            "size": None if (edit or args.res) else (args.size or DEFAULT_SIZE),
                            "ref": str(ref_path) if edit else None,
                            "ref_upload": ref_name,
                            "edit_resolution": edit_resolution,
                            "graph": str(graph_path),
                            "elapsed_s": round(elapsed, 1),
                            "driving_model": args.model,
                            "output_revision": revision,
                            "prompt_id": prompt_id["id"],
                            "source_image": history_item.get("filename") if isinstance(history_item, dict) else None,
                            "fetch": args.fetch,
                        }
                        try:
                            save_image_with_meta(destination, raw, metadata, overwrite=False)
                        except FileExistsError:
                            continue
                    break
                final_path = keep_into(destination, keep_dir) if keep_dir else destination
                written.append(str(final_path))
                runs.append({"index": job["index"], "seed": job["seed"], "elapsed_s": round(elapsed, 1),
                             "path": str(final_path), "prompt_id": prompt_id["id"]})
                print(f"  [{job['index']:02d} seed={job['seed']}] {elapsed:.1f}s {out_w}x{out_h} -> {final_path}")
            except KeyboardInterrupt:
                interrupted = True
                remote = _abort_remote(qwen_host, prompt_id["id"])
                runs.append({"index": job["index"], "seed": job["seed"], "interrupted": True,
                             "prompt_id": prompt_id["id"], "remote_cleanup": remote})
                print(f"  [{job['index']:02d} seed={job['seed']}] INTERRUPTED (remote cleanup: {remote})",
                      file=sys.stderr)
                break
            except Exception as exc:
                failures += 1
                runs.append({"index": job["index"], "seed": job["seed"], "error": f"{type(exc).__name__}: {exc}",
                             "prompt_id": prompt_id["id"]})
                print(f"  [{job['index']:02d} seed={job['seed']}] FAILED ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
            finally:
                total += time.monotonic() - started
    finally:
        if not args.free:
            print("  /free: skipped (models stay loaded; run freeboxa to release the GPU)")
        else:
            try:
                cleanup_ok = free_vram(qwen_host)
            except Exception:
                cleanup_ok = False
            print("  /free:", "ok" if cleanup_ok else "FAILED")
        # rc travels inside the result: app/jobs.py reads the JSON, never the exit status.
        # a failed /free is reported in the result, never a job failure: the images are already on disk.
        rc = 130 if interrupted else (0 if len(written) == args.n and failures == 0 else 1)
        _write_result(args.json_out, {
            "schema": "loramake.qwen21_gen.result.v1",
            "rc": rc,
            "new_files": written,
            "runs": runs,
            "interrupted": interrupted,
            "failures": failures,
            "cleanup_ok": cleanup_ok,
            "sidecar": qwen_host,
        })

    status = "interrupted" if interrupted else ("ok" if cleanup_ok else "cleanup failed")
    print(f"done: {len(written)}/{args.n} image(s), {failures} failed, {status}, {total:.1f}s total")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
