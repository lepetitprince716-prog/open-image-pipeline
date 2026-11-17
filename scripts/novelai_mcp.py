#!/usr/bin/env python3
"""Standalone stdio MCP server ``novelai``: NovelAI txt2img for any agent.

Three tools, no repository-specific data. Anyone can point an MCP client at
``scripts/novelai-mcp`` with ``NAI_ACCESS_TOKEN`` set and generate images:

    nai_generate   txt2img over the full parameter surface; free tier locked
    nai_models     model alias table and the free-tier rule (offline)
    nai_budget     today's Anlas ledger total (offline)

Generation shells out to ``scripts/nai_gen.py``. That client reports errors with
``SystemExit`` and prints result paths on stdout, which is the MCP wire under
stdio, so it runs in a subprocess and never in this process. Its pure functions
(model aliases, cost estimation, ledger totals) are imported directly.

The free tier is the default: NovelAI bills nothing for a default size at 28
steps or fewer, so ``nai_generate`` refuses anything with a non-zero estimate
unless the caller passes ``paid=true``. ``nai_gen.py`` keeps its own fail-closed
daily ledger on top of that, which ``paid=true`` does not bypass.

CLI twin for agents without an MCP client:

    scripts/novelai-mcp tools
    scripts/novelai-mcp call nai_models
    scripts/novelai-mcp call nai_generate '{"prompt": "1girl, solo", "dry_run": true}'
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 12):  # pydantic builds the tool schema from these
    from typing import NotRequired, TypedDict
else:  # on <3.12 it rejects typing.TypedDict and wants the backport
    from typing_extensions import NotRequired, TypedDict

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nai_gen  # noqa: E402

SERVER_NAME = "novelai"
NAI_GEN = SCRIPTS / "nai_gen.py"
DEFAULT_MODEL = "v5-full"
MAX_COUNT = 8
RENDER_TIMEOUT_S = 300.0
STDERR_TAIL = 800

INSTRUCTIONS = (
    "NovelAI image generation over the official API. nai_generate is txt2img with the full "
    "parameter surface (model, multi-character prompts with centers, size, steps, scale, "
    "sampler, noise schedule, seed). The free tier is locked in by default: a default size at "
    "<=28 steps costs 0 Anlas, and anything dearer is refused unless paid=true. Use "
    "dry_run=true to see the resolved model, size and Anlas estimate without a network call or "
    "a token. nai_models lists the model aliases; nai_budget reports today's Anlas spend. "
    "Needs NAI_ACCESS_TOKEN in the environment."
)


class NaiError(RuntimeError):
    """A tool-level failure, reported to the caller verbatim."""


class Center(TypedDict):
    """Where a character sits on the canvas, each coordinate in 0..1."""

    x: float
    y: float


class Character(TypedDict):
    """One character in a multi-character scene.

    Declared as a TypedDict so the generated tool schema names these fields:
    a bare ``list[dict]`` reaches the model as an object with no properties.
    """

    prompt: str
    center: NotRequired[Center]


Size = Literal["portrait", "landscape", "square"]


# -- helpers -----------------------------------------------------------------------


def _ledger_path() -> Path | None:
    """The Anlas ledger both tools use. ``NAI_ANLAS_LEDGER=none`` opts out."""
    raw = os.environ.get("NAI_ANLAS_LEDGER", nai_gen.DEFAULT_ANLAS_LEDGER)
    if raw in nai_gen.ANLAS_LEDGER_OFF:
        return None
    path = Path(raw)
    # MCP hosts launch a server with an arbitrary CWD; anchor relative paths to the repo.
    return path if path.is_absolute() else REPO / path


def _output_root() -> Path:
    raw = os.environ.get("NOVELAI_OUTPUT_DIR", "outputs/nai/mcp")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO / path


def _resolve_model(name: str) -> str:
    try:
        return nai_gen.resolve_model(name)
    except SystemExit as exc:  # resolve_model reports unknown aliases this way
        raise NaiError(str(exc)) from None


def _resolve_size(size: str, width: int | None, height: int | None) -> tuple[int, int]:
    if size not in nai_gen.DEFAULT_SIZES:
        raise NaiError(f"size must be one of {', '.join(sorted(nai_gen.DEFAULT_SIZES))}")
    default_w, default_h = nai_gen.DEFAULT_SIZES[size]
    w, h = int(width or default_w), int(height or default_h)
    if w <= 0 or h <= 0 or w % 64 or h % 64:
        raise NaiError(f"width and height must be positive multiples of 64 (got {w}x{h})")
    return w, h


def _build_characters(characters: list[Character] | None) -> list[dict]:
    """Validate the job's character list: prompts required, centers all-or-none."""
    if not characters:
        return []
    if not isinstance(characters, list):
        raise NaiError("characters must be a list of {prompt, center?} objects")
    out = []
    for i, entry in enumerate(characters, 1):
        if not isinstance(entry, dict) or not str(entry.get("prompt", "")).strip():
            raise NaiError(f"character {i} needs a non-empty 'prompt'")
        item = {"prompt": str(entry["prompt"]).strip()}
        if entry.get("center") is not None:
            try:
                item["center"] = nai_gen.validate_center(entry["center"], f"character {i}")
            except SystemExit as exc:
                raise NaiError(str(exc)) from None
        out.append(item)
    placed = sum(1 for c in out if "center" in c)
    # build_payload refuses mixed placement with SystemExit; catch it before spawning.
    if 0 < placed < len(out):
        raise NaiError(
            f"give every character a center or none ({placed} of {len(out)} have one); "
            "mixed placement would silently pin the rest to the middle"
        )
    return out


def _estimate(model_id: str, width: int, height: int, steps: int) -> float:
    return nai_gen.estimate_generation_cost(model_id, width, height, steps)


def _check_free_tier(model_id: str, width: int, height: int, steps: int, paid: bool) -> float:
    cost = _estimate(model_id, width, height, steps)
    if cost > 0 and not paid:
        raise NaiError(
            f"{model_id} at {width}x{height} / {steps} steps is estimated at {cost} Anlas per "
            "image; the free tier is one image of at most 1024x1024 pixels in any shape (every named "
            "size fits) at <=28 steps. Pass paid=true to spend Anlas."
        )
    return cost


def _gen_argv(job_path: Path, out_dir: Path, seed: int | None, width: int, height: int,
              steps: int, scale: float, cfg_rescale: float, sampler: str,
              noise_schedule: str, uc_preset: int, quality_toggle: bool) -> list[str]:
    ledger = _ledger_path()
    argv = [
        sys.executable, str(NAI_GEN),
        "--job", str(job_path), "--out", str(out_dir),
        "--width", str(width), "--height", str(height), "--steps", str(steps),
        "--scale", str(scale), "--cfg-rescale", str(cfg_rescale),
        "--sampler", sampler, "--noise-schedule", noise_schedule,
        "--uc-preset", str(uc_preset),
        "--anlas-ledger", str(ledger) if ledger else "none",
    ]
    if quality_toggle:
        argv.append("--quality-toggle")
    if seed is not None:
        argv += ["--seed", str(seed)]
    return argv


def _run_once(argv: list[str]) -> dict:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=RENDER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise NaiError(f"nai_gen.py did not finish within {RENDER_TIMEOUT_S:.0f}s") from None
    files = [ln.strip() for ln in done.stdout.splitlines() if ln.strip().endswith(".png")]
    entry: dict = {"returncode": done.returncode, "files": files}
    if done.returncode != 0:
        entry["error"] = done.stderr.strip()[-STDERR_TAIL:]
    return entry


# -- tools -------------------------------------------------------------------------


def tool_generate(
    prompt: str,
    negative: str | None = None,
    model: str = DEFAULT_MODEL,
    characters: list[Character] | None = None,
    size: Size = "portrait",
    width: int | None = None,
    height: int | None = None,
    steps: int = 28,
    scale: float = 5.0,
    cfg_rescale: float = 0.0,
    sampler: str = "k_euler_ancestral",
    noise_schedule: str = "karras",
    seed: int | None = None,
    uc_preset: int = 3,
    quality_toggle: bool = False,
    count: int = 1,
    paid: bool = False,
    dry_run: bool = False,
    out: str | None = None,
) -> dict:
    if not str(prompt or "").strip():
        raise NaiError("prompt must not be empty")
    if sampler not in nai_gen.SAMPLERS:
        raise NaiError(f"sampler must be one of {', '.join(nai_gen.SAMPLERS)}")
    if noise_schedule not in nai_gen.NOISE_SCHEDULES:
        raise NaiError(f"noise_schedule must be one of {', '.join(nai_gen.NOISE_SCHEDULES)}")
    if uc_preset not in (0, 1, 2, 3):
        raise NaiError("uc_preset must be 0, 1, 2 or 3")
    if steps < 1:
        raise NaiError("steps must be at least 1")
    count = int(count)
    if not 1 <= count <= MAX_COUNT:
        raise NaiError(f"count must be between 1 and {MAX_COUNT}")

    model_id = _resolve_model(model)
    w, h = _resolve_size(size, width, height)
    chars = _build_characters(characters)
    unit_cost = _check_free_tier(model_id, w, h, steps, paid)

    job = {
        "model": model_id,
        "base": str(prompt).strip(),
        "characters": chars,
        "negative": negative if negative is not None else nai_gen.DEFAULT_NEGATIVE,
    }
    summary = {
        "model": model_id, "size": [w, h], "steps": steps, "seed": seed,
        "count": count, "anlas_per_image": unit_cost, "anlas_total": unit_cost * count,
        "free_tier": nai_gen.is_opus_free_eligible(model_id, w, h, steps),
        "paid": paid, "job": job,
    }
    if dry_run:
        # Offline preview: no subprocess, no network, no token. nai_gen.py's own
        # --dry-run prints the payload on stdout and the estimate on stderr, neither
        # of which survives the .png line filter below.
        return {"action": "generate", "dry_run": True, **summary}

    out_dir = Path(out).expanduser() if out else _output_root() / time.strftime("%Y%m%d-%H%M%S")
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results = []
    for i in range(count):
        # nai_gen.py renders one image per call; walk the seed so a batch is not N twins.
        this_seed = None if seed is None else int(seed) + i
        results.append({"seed": this_seed, **_run_once(_gen_argv(
            job_path, out_dir, this_seed, w, h, steps, scale, cfg_rescale,
            sampler, noise_schedule, uc_preset, quality_toggle))})

    files = [f for r in results for f in r["files"]]
    failed = [r for r in results if r["returncode"] != 0]
    if not files and failed:
        raise NaiError("nai_gen.py failed: " + (failed[0].get("error") or "no output"))
    return {"action": "generate", "dry_run": False, **summary,
            "out_dir": str(out_dir), "files": files, "results": results}


def tool_models() -> dict:
    free_sizes = [f"{w}x{h}" for w, h in nai_gen.DEFAULT_SIZES.values()]
    return {
        "default": DEFAULT_MODEL,
        "aliases": dict(nai_gen.MODELS),
        "raw_ids": "any nai-diffusion-* id passes through for models newer than this table",
        "samplers": list(nai_gen.SAMPLERS),
        "noise_schedules": list(nai_gen.NOISE_SCHEDULES),
        "free_tier": {
            "sizes": free_sizes,
            "max_pixels": nai_gen.OPUS_FREE_MAX_PIXELS,
            "max_steps": 28,
            "note": "one image of at most max_pixels in any shape (every listed size fits) at "
                    "<=28 steps costs 0 Anlas on V3 and newer; anything else needs paid=true",
        },
    }


def tool_budget(date: str | None = None) -> dict:
    path = _ledger_path()
    if path is None:
        return {"ledger": None, "note": "ledger disabled via NAI_ANLAS_LEDGER=none"}
    day = date or nai_gen._ledger_date()
    budget = float(os.environ.get("NAI_ANLAS_BUDGET", nai_gen.DEFAULT_ANLAS_BUDGET))
    used = nai_gen.used_today(path, day) if path.exists() else 0.0
    return {
        "ledger": str(path), "date": day, "used": used, "budget": budget,
        "remaining": budget - used, "exists": path.exists(),
    }


# -- registration ------------------------------------------------------------------


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def register(mcp) -> None:
    """Attach the three NovelAI tools to an MCPServer."""

    @mcp.tool(name="nai_generate", structured_output=False)
    def nai_generate(
        prompt: str,
        negative: str | None = None,
        model: str = DEFAULT_MODEL,
        characters: list[Character] | None = None,
        size: Size = "portrait",
        width: int | None = None,
        height: int | None = None,
        steps: int = 28,
        scale: float = 5.0,
        cfg_rescale: float = 0.0,
        sampler: str = "k_euler_ancestral",
        noise_schedule: str = "karras",
        seed: int | None = None,
        uc_preset: int = 3,
        quality_toggle: bool = False,
        count: int = 1,
        paid: bool = False,
        dry_run: bool = False,
        out: str | None = None,
    ) -> str:
        """NovelAI txt2img. Free tier locked: a default size (832x1216 / 1216x832 / 1024x1024) at <=28 steps costs 0 Anlas, and a dearer request is refused unless paid=true. dry_run=true returns the resolved model, size and Anlas estimate with no network call and no token. characters is [{prompt, center:{x,y}}] for multi-character scenes — give every character a center or none. count renders that many images, walking the seed. Saves PNGs and returns their paths."""
        return _dump(tool_generate(
            prompt, negative, model, characters, size, width, height, steps, scale,
            cfg_rescale, sampler, noise_schedule, seed, uc_preset, quality_toggle,
            count, paid, dry_run, out))

    @mcp.tool(name="nai_models", structured_output=False)
    def nai_models() -> str:
        """Model aliases NovelAI accepts (v5-full is the default), the sampler and noise-schedule tables, and the free-tier rule. Offline."""
        return _dump(tool_models())

    @mcp.tool(name="nai_budget", structured_output=False)
    def nai_budget(date: str | None = None) -> str:
        """Anlas spent today against the daily budget, from the same fail-closed ledger nai_generate writes. Offline."""
        return _dump(tool_budget(date))


# -- entry points ------------------------------------------------------------------


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
    register(mcp)
    mcp.run(transport="stdio")


def cli(argv: list[str]) -> int:
    if argv and argv[0] == "tools":
        collector = _Collector()
        register(collector)
        for name, fn in collector.tools.items():
            print(f"{name:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    if len(argv) >= 2 and argv[0] == "call":
        collector = _Collector()
        register(collector)
        name = argv[1]
        if name not in collector.tools:
            print(f"unknown tool {name}; run `tools` for the list", file=sys.stderr)
            return 2
        try:
            args = json.loads(argv[2]) if len(argv) > 2 else {}
            if not isinstance(args, dict):
                raise NaiError("args must be a JSON object")
            print(collector.tools[name](**args))
        except (NaiError, json.JSONDecodeError, TypeError, KeyError) as exc:
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
