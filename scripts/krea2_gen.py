#!/usr/bin/env python3
"""Krea-2 Turbo generation line (Z3 by default, ``--box zzz`` for the 4060 Ti worker).

The default invocation is a local preview: it expands prompts and prints the exact jobs/output paths,
but performs no HTTP requests and creates no files. Pass ``--run`` to preflight the fixed Krea-2 assets,
wait for an idle box, submit the jobs, and save the returned PNGs.

Recipe (validated in ``outputs/_analysis/krea2_ab/REPORT.md``):
  UNETLoader(darkBeast INT8 ConvRot) + CLIPLoader(qwen3vl_4b_fp8_scaled, type=krea2)
  + VAELoader(qwen_image_vae) -> KSampler(8 steps, cfg 1.0, euler/simple).

Output is always rooted at this repository, independent of the caller's cwd:
``outputs/krea2/by_model/<DRIVING_MODEL>/krea2_gen``. PNGs carry a ``loramake`` iTXt record and a
``.meta.json`` sidecar. Existing same-seed files are retained and the new output gets a ``__vN`` suffix.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import wait_for_idle  # noqa: E402  # reuse the repo's shared-box idle policy
from wildcard import WildcardEngine, WildcardError  # noqa: E402
from z3_comfy import (  # noqa: E402  # shared driver helpers (loramake.comfy)
    Z3_HOST,
    ZZZ_HOST,
    free_vram,
    looks_like_png,
    preflight_assets,
    save_image_with_meta,
    submit_and_fetch,
)

BOX_HOSTS = {"z3": Z3_HOST, "zzz": ZZZ_HOST}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "claude"
MODEL_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

KREA2 = {
    "unet": "darkBeastINT8Convrot2_krea211INT8Convrot.safetensors",
    "clip": "qwen3vl_4b_fp8_scaled.safetensors",
    "clip_type": "krea2",
    "vae": "qwen_image_vae.safetensors",
    "steps": 8,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
}
RES_PRESETS = {
    "square": (1024, 1024),
    "portrait": (1024, 1408),
    "landscape": (1408, 1024),
    "hd": (1536, 1536),
    "max": (2048, 2048),
}


def model_slug(value: str) -> str:
    """Normalize a driver slug while rejecting path/control syntax."""
    slug = value.strip().lower()
    if not MODEL_SLUG_RE.fullmatch(slug) or slug in {".", ".."} or ".." in slug:
        raise argparse.ArgumentTypeError(
            "model must be a 1-64 character lowercase slug using letters, digits, '.', '_' or '-' "
            "(path separators and '..' are not allowed)"
        )
    return slug


def output_dir_for_model(value: str) -> Path:
    """Return the repo-rooted output bucket for a validated driving-model slug."""
    slug = model_slug(value)
    return REPO_ROOT / "outputs" / "krea2" / "by_model" / slug / "krea2_gen"


# Preserve the old import-time constants for callers that inspect them. The CLI still validates the raw
# environment value through argparse, so an invalid DRIVING_MODEL is reported instead of silently used.
try:
    MODEL = model_slug(os.environ.get("DRIVING_MODEL", DEFAULT_MODEL))
except argparse.ArgumentTypeError:
    MODEL = DEFAULT_MODEL
OUTDIR = str(output_dir_for_model(MODEL))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than zero") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def _nonnegative_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer") from exc
    if parsed < 0 or parsed >= 2**64:
        raise argparse.ArgumentTypeError("seed must be in the range 0..2^64-1")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def resolve_res(value: str) -> tuple[int, int]:
    """Argparse converter for a supported preset or usable Krea-2 ``WIDTHxHEIGHT`` shape."""
    raw = value.strip().lower()
    if raw in RES_PRESETS:
        return RES_PRESETS[raw]
    parts = raw.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "resolution must be WIDTHxHEIGHT or a preset: " + ", ".join(RES_PRESETS)
        )
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution dimensions must be integers") from exc
    if any(dimension < 1024 or dimension > 2048 for dimension in (width, height)):
        raise argparse.ArgumentTypeError("resolution must be 1024-2048 pixels per side")
    if width % 64 or height % 64:
        raise argparse.ArgumentTypeError("resolution dimensions must be multiples of 64")
    return width, height


def parse_lora(value: str) -> tuple[str, float]:
    """Accept 'FILE' or 'FILE:STRENGTH' (strength in (0, 2])."""
    if ":" in value:
        name, strength_s = value.rsplit(":", 1)
        try:
            strength = float(strength_s)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid LoRA strength: {strength_s!r}") from exc
        if not (0.0 < strength <= 2.0):
            raise argparse.ArgumentTypeError("LoRA strength must be in (0, 2]")
    else:
        name, strength = value, 1.0
    if not name.strip():
        raise argparse.ArgumentTypeError("LoRA name must not be empty")
    return name, strength


def build_graph(
    pos: str,
    neg: str,
    seed: int,
    w: int,
    h: int,
    steps: int,
    cfg: float,
    unet: str,
    loras: list[tuple[str, float]] | None = None,
) -> dict:
    graph: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": KREA2["clip"], "type": KREA2["clip_type"], "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": KREA2["sampler"],
                "scheduler": KREA2["scheduler"],
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "krea2_gen"}},
    }
    # DiT LoRA 链：LoraLoaderModelOnly 依次接在 UNET 输出上（参考 Wan 线 ANIME_VIDEO_PLAYBOOK.md）
    model_ref: list = ["1", 0]
    for i, (lname, lstr) in enumerate(loras or []):
        nid = f"10_{i}"
        graph[nid] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": lname, "strength_model": lstr},
        }
        model_ref = [nid, 0]
    graph["7"]["inputs"]["model"] = model_ref
    if neg:
        graph["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}}
    else:
        graph["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
    return graph


def _fast_delay(elapsed: float) -> float:
    """Tight early polling for ~5s jobs, relaxed once something long is clearly running."""
    if elapsed < 10:
        return 0.5
    if elapsed < 60:
        return 2.0
    return 5.0


def _filename_token(value: str, *, fallback: str) -> str:
    """Make a display label/model basename safe as one filename component."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("._-")
    return (token or fallback)[:96]


def output_path(outdir: Path, *, label: str, unet: str, steps: int, index: int, seed: int) -> Path:
    label_token = _filename_token(label, fallback="") if label else ""
    tag = f"_{label_token}" if label_token else ""
    unet_name = unet.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    unet_token = _filename_token(unet_name, fallback="model")
    stem = f"krea2{tag}_{unet_token}_{steps}st_{index:02d}_{seed}"
    return outdir / f"{stem}.png"


def _artifact_occupied(path: Path) -> bool:
    """Treat an image, its sidecar, or even a dangling symlink as an occupied output name."""
    return os.path.lexists(path) or os.path.lexists(Path(str(path) + ".meta.json"))


def _claim_file(path: Path) -> Path:
    return path.with_name(f".{path.name}.claim")


def _versioned_path(path: Path, version: int) -> Path:
    return path if version == 1 else path.with_name(f"{path.stem}__v{version}{path.suffix}")


def _occupied(path: Path) -> bool:
    return _artifact_occupied(path) or os.path.lexists(_claim_file(path))


def no_clobber_path(path: Path) -> tuple[Path, int]:
    """Return ``path`` or the first traceable ``__vN`` sibling that is not occupied."""
    for version in range(1, 10_001):
        candidate = _versioned_path(path, version)
        if not _occupied(candidate):
            return candidate, version
    raise RuntimeError(f"could not allocate a non-clobbering output name near {path}")


class OutputClaim:
    """Filesystem claim held across PNG plus sidecar publication for cooperating processes."""

    def __init__(self, path: Path, version: int, claim_path: Path, fd: int) -> None:
        self.path = path
        self.version = version
        self.claim_path = claim_path
        self.fd = fd
        stat = os.fstat(fd)
        self._identity = (stat.st_dev, stat.st_ino)
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            try:
                stat = self.claim_path.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == self._identity:
                    self.claim_path.unlink()
            except OSError:
                pass
        finally:
            os.close(self.fd)

    def __enter__(self) -> "OutputClaim":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


def claim_output_path(path: Path) -> OutputClaim:
    """Atomically claim a free output revision; concurrent callers receive distinct ``__vN`` paths."""
    for version in range(1, 10_001):
        candidate = _versioned_path(path, version)
        if _artifact_occupied(candidate):
            continue
        claim_path = _claim_file(candidate)
        try:
            fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode())
            os.fsync(fd)
            claim = OutputClaim(candidate, version, claim_path, fd)
        except Exception:
            os.close(fd)
            claim_path.unlink(missing_ok=True)
            raise
        # Close the check-then-claim race with a second check while the exclusive claim is held.
        if _artifact_occupied(candidate):
            claim.release()
            continue
        return claim
    raise RuntimeError(f"could not atomically claim an output name near {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview or run Krea-2 Turbo generation on Z3")
    parser.add_argument("--prompt", required=True, help="natural-language caption (no tag scaffold is added)")
    parser.add_argument("--neg", default="", help="negative prompt; only meaningful with --cfg > 1")
    parser.add_argument("--n", type=_positive_int, default=1, help="number of images (positive integer)")
    parser.add_argument("--seed", type=_nonnegative_seed, default=None, help="base seed (random if omitted)")
    parser.add_argument(
        "--res",
        type=resolve_res,
        default="square",
        metavar="WIDTHxHEIGHT|PRESET",
        help="1024-2048, multiples of 64; presets: " + " / ".join(RES_PRESETS),
    )
    parser.add_argument(
        "--steps", type=_positive_int, default=KREA2["steps"], help="positive sampling steps (Turbo default 8)"
    )
    parser.add_argument(
        "--cfg", type=_positive_finite_float, default=KREA2["cfg"], help="finite CFG > 0 (Turbo default 1.0)"
    )
    parser.add_argument("--unet", default=KREA2["unet"], help="Krea-2 diffusion model file on the box")
    parser.add_argument(
        "--lora", type=parse_lora, action="append", default=None,
        metavar="FILE[:STRENGTH]",
        help="LoraLoaderModelOnly name (repeatable; 'FILE:STRENGTH' to set strength, default 1.0)",
    )
    parser.add_argument(
        "--box", choices=sorted(BOX_HOSTS), default="z3",
        help="target box (z3 = RTX 5080 primary; zzz = 4060 Ti worker, needs ComfyUI >= 0.26 + the models)",
    )
    parser.add_argument("--model", type=model_slug, default=os.environ.get("DRIVING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--label", default="", help="extra token in the output filename")
    parser.add_argument("--run", action="store_true", help="submit to Z3 (default: print a local preview only)")
    parser.add_argument(
        "--no-wait", action="store_true", help="--run: skip the default wait for an idle Z3 and submit immediately"
    )
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--no-free", action="store_true", help="--run: keep models/VRAM loaded instead of POST /free"
    )
    cleanup.add_argument(
        "--free", dest="no_free", action="store_false", help="--run: POST /free (legacy explicit form; now default)"
    )
    parser.set_defaults(no_free=False)
    parser.add_argument("--no-wildcard", action="store_true", help="disable {a|b}/__file__ prompt expansion")
    parser.add_argument(
        "--wildcard-dir", default=None, help="dir holding __name__.txt wildcard files (default: repo wildcards/)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not args.unet.strip():
        parser.error("--unet must not be empty")

    width, height = args.res
    base_seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2_000_000_000
    if base_seed + args.n - 1 >= 2**64:
        parser.error("base seed plus --n exceeds ComfyUI's maximum seed")

    outdir = output_dir_for_model(args.model)
    wildcard = None if args.no_wildcard else WildcardEngine(args.wildcard_dir)
    jobs: list[dict] = []
    try:
        for index in range(args.n):
            seed = base_seed + index
            prompt = wildcard.expand(args.prompt, seed) if wildcard else args.prompt
            base_path = output_path(
                outdir, label=args.label, unet=args.unet, steps=args.steps, index=index, seed=seed
            )
            planned_path, revision = no_clobber_path(base_path)
            jobs.append(
                {
                    "index": index,
                    "seed": seed,
                    "prompt": prompt,
                    "base_path": base_path,
                    "planned_path": planned_path,
                    "revision": revision,
                }
            )
    except WildcardError as exc:
        parser.error(f"wildcard expansion failed: {exc}")

    loras: list[tuple[str, float]] = list(args.lora or [])
    host = BOX_HOSTS[args.box]
    mode = "RUN" if args.run else "PLAN"
    print(
        f"krea2-gen {mode} unet={args.unet} clip={KREA2['clip']} vae={KREA2['vae']} "
        f"steps={args.steps} cfg={args.cfg} {KREA2['sampler']}/{KREA2['scheduler']} "
        f"{width}x{height} model={args.model} n={args.n} box={args.box} host={host}"
    )
    for job in jobs:
        print(f"  [{job['index']:02d} seed={job['seed']}] -> {job['planned_path']}")
        if job["prompt"] != args.prompt:
            print(f"        prompt: {job['prompt']}")
    if not args.run:
        print("preview only: no network request or file write; pass --run to execute")
        return 0

    # Network begins only after --run. Check every fixed loader asset before waiting/submitting.
    preflight_assets(
        "krea2",
        args.unet,
        [name for name, _ in loras],
        clips=[KREA2["clip"]],
        vaes=[KREA2["vae"]],
        host=host,
    )
    if not args.no_wait:
        ok, reason = wait_for_idle(host)
        if not ok:
            print(f"[abort] {args.box} not idle: {reason} (use --no-wait to override)", file=sys.stderr)
            return 2
        print(f"  {args.box} {reason}")

    outdir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    written = 0
    failures = 0
    cleanup_ok = True

    # Arm cleanup immediately before the first possible submission. This guarantees cleanup for submit,
    # fetch, validation, and save exceptions without unloading somebody else's model after an idle abort.
    try:
        for job in jobs:
            started = time.monotonic()
            try:
                graph = build_graph(
                    job["prompt"],
                    args.neg,
                    job["seed"],
                    width,
                    height,
                    args.steps,
                    args.cfg,
                    args.unet,
                    loras=loras,
                )
                results = submit_and_fetch(
                    graph,
                    label=f"krea2 {job['index']:02d}",
                    host=host,
                    delay=_fast_delay,
                    validate=looks_like_png,
                )
                if not results:
                    raise RuntimeError("no valid PNG returned")
                raw, history_item = results[0]
                if not looks_like_png(raw):
                    raise RuntimeError("fetched output is empty or not a valid PNG")

                elapsed = time.monotonic() - started
                while True:
                    with claim_output_path(job["base_path"]) as claim:
                        destination, revision = claim.path, claim.version
                        metadata = {
                            "schema_version": "loramake.krea2_gen.v1",
                            "engine": "krea2_gen",
                            "gen_engine": f"comfyui:{args.box}/krea2",
                            "box": args.box,
                            "model": args.unet,
                            "loras": [{"name": name, "strength": strength} for name, strength in loras],
                            "host": host,
                            "seed": job["seed"],
                            "steps": args.steps,
                            "cfg": args.cfg,
                            "sampler": KREA2["sampler"],
                            "scheduler": KREA2["scheduler"],
                            "prompt": job["prompt"],
                            "negative": args.neg,
                            "resolution": f"{width}x{height}",
                            "clip": KREA2["clip"],
                            "clip_type": KREA2["clip_type"],
                            "vae": KREA2["vae"],
                            "elapsed_s": round(elapsed, 1),
                            "driving_model": args.model,
                            "line": "krea2_gen",
                            "output_revision": revision,
                            "source_image": (
                                history_item.get("filename") if isinstance(history_item, dict) else None
                            ),
                            # Backward-compatible mirrors used by existing Krea-2 sidecars/readers.
                            "unet": args.unet,
                            "res": [width, height],
                            "neg": args.neg,
                        }
                        try:
                            save_image_with_meta(destination, raw, metadata, overwrite=False)
                        except FileExistsError:
                            # A non-cooperating writer won after our claim; preserve it and claim __vN.
                            continue
                    break
                written += 1
                print(f"  [{job['index']:02d} seed={job['seed']}] {elapsed:.1f}s -> {destination}")
            except Exception as exc:
                failures += 1
                elapsed = time.monotonic() - started
                print(
                    f"  [{job['index']:02d} seed={job['seed']}] FAILED "
                    f"({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
            finally:
                total += time.monotonic() - started
    finally:
        if args.no_free:
            print("  /free: skipped (--no-free)")
        else:
            try:
                cleanup_ok = free_vram(host)
            except Exception:
                cleanup_ok = False
            print("  /free:", "ok" if cleanup_ok else "FAILED")

    cleanup_status = "ok" if cleanup_ok else "failed"
    print(f"done: {written}/{args.n} image(s), {failures} failed, cleanup={cleanup_status}, {total:.1f}s total")
    return 0 if written == args.n and failures == 0 and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
