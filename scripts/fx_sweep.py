#!/usr/bin/env python3
"""Parameter-sweep contact sheets: render one source image through a grid of retouch.py / post_fx.py
parameter combinations and tile the variants (plus the untouched original) into one labeled contact
sheet, so a human can pick a parameter setting by eye instead of guessing values blind. Pure
post-processing glue over the three sibling modules — no GPU, no new image maths of its own.

Each --sweep flag is one grid axis: a dataclass field name (from retouch.Retouch or post_fx.Grade) or
the literal "preset", paired with a comma-separated list of values. The grid is the cartesian product
of all axes. --set fixes a field to one value across every variant. The default --preset is "none" (not
the tool's own default preset) because a sweep is meant to explore deltas from neutral unless the user
names a starting look.

Usage:
  fx_sweep.py in.png --sweep shadows=0.0,0.4,0.8 --sweep clarity=0.0,0.2
  fx_sweep.py in.png --tool post_fx --sweep preset=none,film,teal-orange --sweep grain=0.0,0.03
  fx_sweep.py in.png --preset fix --set denoise=0.3 --sweep sharpen=0.0,0.3,0.6 --cols 4
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import types
import typing
from pathlib import Path

from PIL import Image

import contact_sheet
import post_fx
import retouch

_MAX_STEM_LEN = 40
_TOOL_DATACLASS: dict[str, type] = {"retouch": retouch.Retouch, "post_fx": post_fx.Grade}
_TOOL_PRESETS: dict[str, dict[str, typing.Any]] = {"retouch": retouch.PRESETS, "post_fx": post_fx.PRESETS}
_METADATA_FIELDS = {"name"}  # label-only fields with zero rendering effect (e.g. Grade.name); sweeping
                              # them would produce visually-identical variants under different filenames,
                              # so they're excluded from the sweepable/settable field set entirely
_COMMA_UNSAFE_SWEEP_FIELDS = {"curve"}  # str fields whose own spec syntax uses commas as an internal
                                          # delimiter (e.g. "0:0,128:140,255:230"); --sweep's comma-
                                          # separated value list would mistake control points for sweep
                                          # values, so these are blocked from --sweep with a clear error
                                          # and left available via --set (a single raw string, untouched)


def _rename_dir_noreplace(src: Path, dst: Path) -> None:
    """Atomically publish a directory while preserving any destination entry created by a racer.

    Plain POSIX rename may replace an empty destination directory, so use each supported platform's
    no-replace primitive rather than a check-then-rename sequence.
    """
    if os.name == "nt":
        os.rename(src, dst)  # Windows rename fails when dst already exists.
        return

    libc = ctypes.CDLL(None, use_errno=True)
    src_bytes = os.fsencode(src)
    dst_bytes = os.fsencode(dst)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(src_bytes, dst_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_exclusive = libc.renameat2
        rename_exclusive.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(-100, src_bytes, -100, dst_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError(f"atomic no-replace directory publish is unsupported on {sys.platform}")

    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error, os.strerror(error), dst)
    raise OSError(error, os.strerror(error), dst)


def _field_kind(annotation: object) -> str | None:
    """Classify a resolved dataclass field annotation into a sweep-parseable kind, or None if the
    field type isn't one this tool knows how to parse from a CLI string (e.g. the tuple colour tints
    on Grade) — such fields are simply excluded from the valid-params set."""
    if annotation is bool:
        return "bool"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _field_kind(args[0])
    return None


def _valid_fields(tool: str) -> dict[str, str]:
    """{field_name: kind} for the tool's dataclass, kind in {"float", "bool", "str"}."""
    cls = _TOOL_DATACLASS[tool]
    hints = typing.get_type_hints(cls)
    fields: dict[str, str] = {}
    for f in dataclasses.fields(cls):
        if f.name in _METADATA_FIELDS:
            continue
        kind = _field_kind(hints[f.name])
        if kind is not None:
            fields[f.name] = kind
    return fields


def _parse_value(kind: str, raw: str) -> float | bool | str:
    if kind == "float":
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"cannot parse {raw!r} as float") from exc
        if not math.isfinite(value):
            raise ValueError(f"cannot parse {raw!r} as a finite float")
        return value
    if kind == "bool":
        token = raw.strip().lower()
        if token in ("true", "1"):
            return True
        if token in ("false", "0"):
            return False
        raise ValueError(f"cannot parse {raw!r} as bool (expected true/false/1/0)")
    return raw  # "str" kind, including Optional[str] fields like curve: passed through unchanged


def _parse_param(tool: str, param: str, raw: str, valid_fields: dict[str, str],
                  allow_preset: bool) -> float | bool | str:
    """Parse one raw value for `param`, or treat it as a preset name when allow_preset and
    param == "preset". Raises ValueError listing the valid params on any failure (unknown name or
    unparsable value) so the CLI can surface one clear message either way."""
    choices = sorted(valid_fields) + (["preset"] if allow_preset else [])
    if allow_preset and param == "preset":
        presets = _TOOL_PRESETS[tool]
        if raw not in presets:
            raise ValueError(f"unknown preset {raw!r} for tool {tool!r}; valid presets: "
                              f"{sorted(presets)}")
        return raw
    if param not in valid_fields:
        raise ValueError(f"unknown param {param!r} for tool {tool!r}; valid params: "
                          f"{', '.join(choices)}")
    try:
        return _parse_value(valid_fields[param], raw)
    except ValueError as exc:
        raise ValueError(f"{exc}; valid params: {', '.join(choices)}") from exc


def _variant_stem(combo_raw: dict[str, str]) -> str:
    """Filename stem built from the swept params only, e.g. 'shadows0.2_clarity0.3'. Abbreviates each
    param name to its first 4 characters if the full stem would exceed _MAX_STEM_LEN."""
    stem = "_".join(f"{name}{value}" for name, value in combo_raw.items())
    if len(stem) > _MAX_STEM_LEN:
        stem = "_".join(f"{name[:4]}{value}" for name, value in combo_raw.items())
    return stem


def _dedupe_stem(stem: str, seen: dict[str, int]) -> str:
    """Guards against two combos abbreviating to the same stem (e.g. after the 4-char truncation)
    silently overwriting each other's variant file on disk."""
    seen[stem] = seen.get(stem, 0) + 1
    return stem if seen[stem] == 1 else f"{stem}_{seen[stem]}"


def _render(tool: str, img: Image.Image, obj: typing.Any, seed: int) -> Image.Image:
    if tool == "retouch":
        return retouch.retouch_image(img, obj)
    return post_fx.grade_image(img, obj, seed)


def run_sweep(image: str | Path, tool: str = "retouch", preset: str = "none",
              sweeps: dict[str, list[str]] | None = None, fixed: dict[str, str] | None = None,
              out_dir: str | Path | None = None, thumb: int = 512, cols: int | None = None,
              seed: int = 0, max_combos: int = 36) -> Path:
    """Render `image` through the cartesian product of `sweeps` (each value list parsed per the
    tool dataclass field's type, or as preset names for the literal axis "preset"), tile the results
    plus the untouched original into a contact sheet, and return its path. Raises ValueError on an
    unknown tool/preset/param, an unparsable value, a missing sweep, or a grid bigger than
    `max_combos` (never silently truncated)."""
    if tool not in _TOOL_DATACLASS:
        raise ValueError(f"unknown tool {tool!r}; expected one of {sorted(_TOOL_DATACLASS)}")
    presets = _TOOL_PRESETS[tool]
    valid_fields = _valid_fields(tool)
    if preset not in presets:
        raise ValueError(f"unknown preset {preset!r} for tool {tool!r}; valid presets: "
                          f"{sorted(presets)}")
    if not sweeps:
        raise ValueError("at least one --sweep axis is required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"--seed must be a non-negative integer, got {seed!r}")

    overlap = set(sweeps) & set(fixed or {})
    if overlap:
        raise ValueError(f"param(s) {sorted(overlap)} given to both --sweep and --set; a param can only "
                          f"be fixed (--set) or swept (--sweep), not both — --set would be silently "
                          f"overwritten by --sweep")

    axis_names = list(sweeps.keys())
    axis_raw: dict[str, list[str]] = {}
    axis_parsed: dict[str, list[float | bool | str]] = {}
    for name, raw_values in sweeps.items():
        if not raw_values:
            raise ValueError(f"sweep axis {name!r} has no values")
        if name in _COMMA_UNSAFE_SWEEP_FIELDS:
            raise ValueError(f"{name!r} cannot be swept via --sweep: its own spec syntax uses commas, "
                              f"which collide with the --sweep value-list delimiter. Fix it with "
                              f"--set {name}=... instead (a single value, parsed as one raw string).")
        axis_raw[name] = list(raw_values)
        axis_parsed[name] = [_parse_param(tool, name, v, valid_fields, allow_preset=True)
                              for v in raw_values]

    combo_count = math.prod(len(v) for v in axis_raw.values())
    if max_combos < 1:
        raise ValueError(f"--max-combos must be >= 1, got {max_combos}")
    if combo_count > max_combos:
        raise ValueError(f"sweep grid has {combo_count} combinations, exceeding --max-combos "
                          f"{max_combos}")

    parsed_fixed: dict[str, float | bool | str] = {}
    for name, raw in (fixed or {}).items():
        parsed_fixed[name] = _parse_param(tool, name, raw, valid_fields, allow_preset=False)

    if thumb < 1:
        raise ValueError(f"--thumb must be >= 1, got {thumb}")
    if cols is not None and cols < 1:
        raise ValueError(f"--cols must be >= 1, got {cols}")

    src = Path(image)
    if not src.is_file():
        raise ValueError(f"not a file: {src}")
    out = Path(out_dir) if out_dir else src.parent / f"{src.stem}_sweep"
    if src.resolve().parent == out.resolve():
        # original.png (or a variant named like the source) would overwrite the input in place
        raise ValueError(f"output dir {out} contains the input image; pick a different --out-dir")
    paired_axes = [list(zip(axis_raw[name], axis_parsed[name])) for name in axis_names]
    prepared: list[tuple[dict[str, str], typing.Any, str, str]] = []
    stem_counts: dict[str, int] = {}
    for combo in itertools.product(*paired_axes):
        combo_raw = dict(zip(axis_names, (c[0] for c in combo)))
        combo_parsed = dict(zip(axis_names, (c[1] for c in combo)))

        base_name = str(combo_parsed.get("preset", preset))
        obj = presets[base_name]
        if parsed_fixed:
            obj = dataclasses.replace(obj, **parsed_fixed)
        field_updates = {k: v for k, v in combo_parsed.items() if k != "preset"}
        if field_updates:
            obj = dataclasses.replace(obj, **field_updates)

        stem = _dedupe_stem(_variant_stem(combo_raw), stem_counts)
        if tool == "retouch":
            # Validate every effective config (including a fixed curve) before creating output paths.
            retouch._validate_retouch(obj)
        else:
            post_fx._validate_grade(obj, seed)
        prepared.append((combo_raw, obj, base_name, stem))

    if out.exists() or out.is_symlink():
        raise ValueError(f"output dir already exists: {out}; pick a fresh --out-dir")

    source_bytes = src.read_bytes()
    try:
        with Image.open(io.BytesIO(source_bytes)) as opened:
            img = opened.convert("RGB")
            img.load()
    except OSError as exc:
        raise ValueError(f"input is not a readable image: {src}") from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    published = False
    try:
        original_path = staging / "original.png"
        img.save(original_path)
        variant_paths: list[Path] = []
        variant_records: list[dict[str, object]] = []
        for combo_raw, obj, base_name, stem in prepared:
            variant_path = staging / f"{stem}.png"
            _render(tool, img, obj, seed).save(variant_path)
            variant_paths.append(variant_path)
            variant_records.append({
                "file": variant_path.name,
                "sha256": hashlib.sha256(variant_path.read_bytes()).hexdigest(),
                "swept": combo_raw,
                "effective_preset": base_name,
                "effective_config": dataclasses.asdict(obj),
            })

        # Keep the Cartesian grid contiguous in row-major order. The untouched source is useful as a
        # reference tile, but placing it first shifts every 2-D axis by one cell and breaks row/column reads.
        variant_paths.append(original_path)
        sheet_path = contact_sheet.build_sheet([str(p) for p in variant_paths],
                                                str(staging / "sheet.png"),
                                                cols=cols, thumb=thumb)
        manifest = {
            "schema": "loramake.fx_sweep.v1",
            "input": str(src.resolve()),
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "tool": tool,
            "base_preset": preset,
            "fixed": dict(fixed or {}),
            "sweeps": {name: list(values) for name, values in sweeps.items()},
            "seed": seed,
            "max_combos": max_combos,
            "thumb": thumb,
            "cols": cols,
            "variants": variant_records,
            "sheet_order": [p.name for p in variant_paths],
            "sheet": Path(sheet_path).name,
            "sheet_sha256": hashlib.sha256(Path(sheet_path).read_bytes()).hexdigest(),
            "original": original_path.name,
            "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
        try:
            _rename_dir_noreplace(staging, out)
        except FileExistsError as exc:
            raise ValueError(f"output dir appeared during rendering: {out}; staged run was not published") \
                from exc
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return out / "sheet.png"


def _parse_sweep_arg(token: str) -> tuple[str, list[str]]:
    name, sep, rest = token.partition("=")
    if not sep or not rest:
        raise ValueError(f"invalid --sweep spec {token!r}; expected 'param=v1,v2,...'")
    values = [v.strip() for v in rest.split(",")]
    if any(not v for v in values):
        raise ValueError(f"invalid --sweep spec {token!r}: empty value in list")
    return name.strip(), values


def _parse_set_arg(token: str) -> tuple[str, str]:
    name, sep, rest = token.partition("=")
    if not sep or not rest:
        raise ValueError(f"invalid --set spec {token!r}; expected 'param=value'")
    return name.strip(), rest.strip()


def _collect_sweep_args(tokens: list[str] | None) -> dict[str, list[str]]:
    sweeps: dict[str, list[str]] = {}
    for token in tokens or []:
        name, values = _parse_sweep_arg(token)
        if name in sweeps:
            raise ValueError(f"duplicate --sweep param {name!r}; each grid axis must be named once")
        sweeps[name] = values
    return sweeps


def _collect_set_args(tokens: list[str] | None) -> dict[str, str]:
    fixed: dict[str, str] = {}
    for token in tokens or []:
        name, value = _parse_set_arg(token)
        if name in fixed:
            raise ValueError(f"duplicate --set param {name!r}; each fixed value must be named once")
        fixed[name] = value
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a parameter-sweep grid through retouch.py or "
                                             "post_fx.py and tile the variants (plus the untouched "
                                             "original) into one contact sheet for by-eye picking.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--tool", choices=tuple(_TOOL_DATACLASS), default="retouch",
                    help="tool to sweep (default retouch)")
    ap.add_argument("--preset", default="none",
                    help="base preset (default 'none' — sweeps explore deltas from neutral)")
    ap.add_argument("--sweep", action="append", default=None, metavar="param=v1,v2,...",
                    help="one grid axis; repeatable. param is a tool dataclass field or 'preset'. "
                         "'curve' can't be swept this way (its own spec uses commas) — fix it with "
                         "--set curve=... instead")
    ap.add_argument("--set", dest="set_params", action="append", default=None, metavar="param=value",
                    help="fix a param across all variants; repeatable")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="output directory (default: <image stem>_sweep next to the input)")
    ap.add_argument("--thumb", type=int, default=512, help="max thumbnail edge px (default 512)")
    ap.add_argument("--cols", type=int, default=None, help="contact sheet columns (default ~sqrt of count)")
    ap.add_argument("--max-combos", dest="max_combos", type=int, default=36,
                    help="refuse to render more than this many grid combinations (default 36)")
    ap.add_argument("--seed", type=int, default=0, help="post_fx grain seed (ignored by retouch)")
    args = ap.parse_args()

    try:
        sweeps = _collect_sweep_args(args.sweep)
        fixed = _collect_set_args(args.set_params)

        sheet = run_sweep(args.image, tool=args.tool, preset=args.preset, sweeps=sweeps, fixed=fixed,
                          out_dir=args.out_dir, thumb=args.thumb, cols=args.cols, seed=args.seed,
                          max_combos=args.max_combos)
    except ValueError as exc:
        sys.exit(str(exc))

    combo_count = math.prod(len(v) for v in sweeps.values())
    print(f"{combo_count} variants")
    print(str(sheet))


if __name__ == "__main__":
    main()
