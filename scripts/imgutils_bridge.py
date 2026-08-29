#!/usr/bin/env python3
"""Read-only bridge to an external imgutils virtualenv for operations the default
interpreter cannot perform: NovelAI stealth-LSB metadata, WD14 tags, and CCIP
character-similarity scoring.

This script is meant to be executed *by the imgutils venv interpreter* as a
subprocess (see ``run_bridge`` in ``image_recipe.py``), not imported. It emits a
single JSON object on stdout and never raises past ``main`` — every failure mode
degrades to ``{"ok": false, "reason": ...}`` so the caller can warn and continue.

Operations:
  lsb  <image>          -> {"ok", "metadata": <decoded LSB JSON>|null, "reason"}
  wd14 <image>          -> {"ok", "rating": {...}, "general": {...}, "character": {...}, "reason"}
                            (single-image legacy shape, unchanged for existing callers)
  wd14 <img1> <img2>...  -> {"ok", "results": {<path>: {"ok","rating","general","character",
                            "input_identity","reason"}}, "reason"}
                            (2+ paths -> validated multi-image batch; one tagger load, N images)
  ccip <image> <image2> -> {"ok", "difference": float, "threshold": float,
                            "same_character": bool, "reason"}  (lower difference = closer)
  ccip <ref> <img1> <img2>...
                         -> {"ok", "results": {<path>: {legacy ccip shape + input_identity}},
                             "reason"} (one metric-model load, N candidates)
  qc   <image>          -> {"ok", "aesthetic_percentile": float, "aesthetic_label": str,
                            "completeness": str, "monochrome": float, "faces": int, "hands": int,
                            "persons": int, "reason"}  (no-reference quality signals; percentile ranks)
  qc   <img1> <img2>... -> {"ok", "results": {<path>: {legacy qc shape + input_identity}},
                            "reason"} (one process/model load, N images)
  segment <image> <out> -> {"ok", "out": <path>, "size": [w, h], "reason"}
                           (isnetis anime-subject matting; writes an RGBA PNG with transparent bg)
  detect <img1> [img2 ...] [--with-wd14]
                         -> {"ok", "results": {<path>: {"hands","eyes","faces","heads","persons",
                            "text","completeness","wd14"?,"errors"?,"input_identity"}}, "reason"}
                            Runs hands/eyes/faces/heads/persons/text/feet detectors plus anime_completeness
                            (feet = NudeNet FEET_* classes relabeled "foot", conf >= 0.3)
                            over MULTIPLE images in ONE process, so ONNX sessions load once and are
                            reused for the whole batch (imgutils caches sessions per model internally;
                            a 576-image sweep must not reload per image). Each detection box is
                            {"bbox":[x0,y0,x1,y1] int, "label": str, "confidence": float}.
                            --with-wd14 additionally attaches WD14 tags per image (general_threshold
                            lowered to 0.1 so low-confidence defect-tag calibration is visible) so a
                            caller can get detections + tags in the SAME subprocess call. Per-detector
                            and per-image failures are isolated into that image's "errors" dict and
                            retain partial diagnostics, while making the top-level "ok" false.
                            Every input is decoded before inference. A .png argument whose bytes are
                            JPEG (or any suffix/content mismatch) is rejected, as are duplicate path
                            aliases. Every result carries canonical path + SHA-256 + byte/dimension/
                            format identity. Output remains keyed by the full input path so callers
                            can look up a given path directly without stem collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Offline-first: every model this bridge needs must already be in the local HF cache.
# A cache miss then fails in seconds (LocalEntryNotFoundError) instead of hanging for
# minutes on a stalled hub download inside a gate. Provision a missing model explicitly:
#   HF_HUB_OFFLINE=0 <venv-python> imgutils_bridge.py wd14 some.png
# Must run before the lazy imgutils imports pull in huggingface_hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# Names huggingface_hub has used for its offline cache-miss error (renamed once already).
_HF_CACHE_MISS_NAMES = frozenset({"LocalEntryNotFoundError", "EntryNotFoundError"})


def _is_hf_cache_miss(exc: BaseException) -> bool:
    """Detect a huggingface_hub offline cache miss without importing huggingface_hub.

    Walks type(exc).__mro__ (so a renamed subclass still matches) and the __cause__ /
    __context__ chain (so a wrapper exception still surfaces the provisioning hint),
    requiring the class to come from huggingface_hub itself."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for cls in type(current).__mro__:
            module = getattr(cls, "__module__", "") or ""
            if cls.__name__ in _HF_CACHE_MISS_NAMES and module.startswith("huggingface_hub"):
                return True
        for chained in (current.__cause__, current.__context__):
            if chained is not None:
                stack.append(chained)
    return False


def _op_reason(prefix: str, exc: Exception) -> str:
    """Failure reason for an op, naming the offline-cache miss explicitly."""
    if _is_hf_cache_miss(exc) or type(exc).__name__ == "LocalEntryNotFoundError":
        return (f"{prefix}: model not in local HF cache — bridge is offline-first; "
                "provision once with HF_HUB_OFFLINE=0")
    return f"{prefix}: {type(exc).__name__}"


class BatchInputError(ValueError):
    """A batch cannot be scored without every input being uniquely and fully validated."""


_FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(argument: str) -> dict[str, Any]:
    """Decode an image and return its collision-resistant input identity.

    ``PIL.Image.verify`` checks container integrity, then a second open/load checks pixel decoding.
    The detected format must match a recognized filename suffix; in particular a JPEG renamed to
    ``*.png`` is rejected rather than flowing into a PNG-only batch under a misleading name.
    """
    path = Path(argument).expanduser()
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise BatchInputError(f"input not found: {argument}") from exc
    if not canonical.is_file():
        raise BatchInputError(f"input is not a regular file: {argument}")
    stat_before = canonical.stat()
    try:
        from PIL import Image
    except Exception as exc:
        raise BatchInputError(f"Pillow import failed: {type(exc).__name__}") from exc
    try:
        with Image.open(canonical) as image:
            detected_format = str(image.format or "").upper()
            width, height = image.size
            image.verify()
        with Image.open(canonical) as image:
            image.load()
    except Exception as exc:
        raise BatchInputError(f"image decode failed for {argument}: {type(exc).__name__}: {exc}") from exc
    expected_format = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if expected_format is None:
        raise BatchInputError(f"unsupported image suffix for {argument}: {path.suffix or '<none>'}")
    if detected_format != expected_format:
        raise BatchInputError(
            f"image suffix/content mismatch for {argument}: suffix expects {expected_format}, bytes are {detected_format or 'unknown'}"
        )
    digest = _sha256(canonical)
    stat_after = canonical.stat()
    before_signature = (stat_before.st_dev, stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns)
    after_signature = (stat_after.st_dev, stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns)
    if before_signature != after_signature:
        raise BatchInputError(f"input changed during validation: {argument}")
    return {
        "argument": argument,
        "canonical_path": str(canonical),
        "sha256": digest,
        "bytes": int(stat_after.st_size),
        "mtime_ns": int(stat_after.st_mtime_ns),
        "device": int(stat_after.st_dev),
        "inode": int(stat_after.st_ino),
        "format": detected_format,
        "width": int(width),
        "height": int(height),
    }


def _validate_batch_images(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Validate a whole batch before model import/inference and reject key/path collisions."""
    if not paths:
        raise BatchInputError("batch requires at least one image")
    if len(paths) != len(set(paths)):
        raise BatchInputError("duplicate input argument would collide in results")
    identities: dict[str, dict[str, Any]] = {}
    canonical_seen: dict[str, str] = {}
    for argument in paths:
        identity = _inspect_image(argument)
        canonical = identity["canonical_path"]
        if canonical in canonical_seen:
            raise BatchInputError(
                f"duplicate input path alias: {argument} and {canonical_seen[canonical]} resolve to {canonical}"
            )
        canonical_seen[canonical] = argument
        identities[argument] = identity
    return identities


def op_lsb(path: str) -> dict[str, Any]:
    try:
        from imgutils.metadata import LSBReadError, read_lsb_metadata
    except Exception as exc:  # venv missing / wrong interpreter
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        metadata = read_lsb_metadata(path)
    except LSBReadError:
        return {"ok": True, "metadata": None, "reason": "no stealth LSB payload"}
    except Exception as exc:
        return {"ok": False, "reason": f"lsb read error: {type(exc).__name__}"}
    return {"ok": True, "metadata": metadata, "reason": ""}


def op_wd14(path: str) -> dict[str, Any]:
    try:
        from imgutils.tagging import get_wd14_tags
    except Exception as exc:
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        rating, general, character = get_wd14_tags(path)
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("wd14 error", exc)}

    def rounded(tags: dict[str, Any]) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(tags).items()}

    return {
        "ok": True,
        "rating": rounded(rating),
        "general": rounded(general),
        "character": rounded(character),
        "reason": "",
    }


def op_teen(path: str) -> dict[str, Any]:
    """Anime age-band classification (deepghs anime_teen): label in
    {contentious, safe_teen, non_teen} plus the full per-class score map."""
    try:
        from imgutils.validate import anime_teen, anime_teen_score
    except Exception as exc:
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        label, score = anime_teen(path)
        scores = anime_teen_score(path)
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("teen error", exc)}
    return {
        "ok": True,
        "label": str(label),
        "score": round(float(score), 4),
        "scores": {str(k): round(float(v), 4) for k, v in dict(scores).items()},
        "reason": "",
    }


def op_teen_multi(paths: list[str]) -> dict[str, Any]:
    """Multi-image variant of op_teen: one classifier load, N images, one process. Per-image shape
    matches op_teen's own {"ok","label","score","scores","reason"}."""
    try:
        identities = _validate_batch_images(paths)
    except BatchInputError as exc:
        return {"ok": False, "results": {}, "reason": str(exc)}
    try:
        from imgutils.validate import anime_teen, anime_teen_score
    except Exception as exc:
        return {"ok": False, "results": {}, "reason": f"imgutils import failed: {type(exc).__name__}"}
    results: dict[str, Any] = {}
    for path in paths:
        try:
            label, score = anime_teen(path)
            scores = anime_teen_score(path)
            results[path] = {
                "ok": True,
                "label": str(label),
                "score": round(float(score), 4),
                "scores": {str(k): round(float(v), 4) for k, v in dict(scores).items()},
                "input_identity": identities[path],
                "reason": "",
            }
        except Exception as exc:
            results[path] = {"ok": False, "input_identity": identities[path],
                             "reason": _op_reason("teen error", exc)}
    failed = [path for path, result in results.items() if result.get("ok") is not True]
    return {
        "ok": not failed,
        "results": results,
        "reason": "" if not failed else f"teen failed for {len(failed)} image(s)",
    }


def op_wd14_multi(paths: list[str]) -> dict[str, Any]:
    """Multi-image variant of op_wd14: one WD14 tagger load, N images, one process. Per-image shape
    matches op_wd14's own {"ok","rating","general","character","reason"} so callers can reuse the same
    downstream parsing for either the single- or multi-image bridge shape."""
    try:
        identities = _validate_batch_images(paths)
    except BatchInputError as exc:
        return {"ok": False, "results": {}, "reason": str(exc)}
    try:
        from imgutils.tagging import get_wd14_tags
    except Exception as exc:
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}

    def rounded(tags: dict[str, Any]) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(tags).items()}

    results: dict[str, Any] = {}
    for path in paths:
        try:
            rating, general, character = get_wd14_tags(path)
            results[path] = {"ok": True, "rating": rounded(rating), "general": rounded(general),
                              "character": rounded(character), "input_identity": identities[path],
                              "reason": ""}
        except Exception as exc:
            results[path] = {"ok": False, "input_identity": identities[path],
                             "reason": _op_reason("wd14 error", exc)}
    failed = [path for path, result in results.items() if result.get("ok") is not True]
    return {
        "ok": not failed,
        "results": results,
        "reason": "" if not failed else f"wd14 failed for {len(failed)} image(s)",
    }


def _boxes(detections: Any) -> list[dict[str, Any]]:
    """Normalize an imgutils detect_* result (list of (bbox, label, confidence)) into the bridge's
    region shape. Malformed inference output fails closed instead of disappearing as zero boxes."""
    if not isinstance(detections, (list, tuple)):
        raise ValueError("detector output is not a list/tuple")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(detections):
        try:
            bbox, label, conf = item
            coords = [int(v) for v in bbox]
            confidence = float(conf)
        except Exception as exc:
            raise ValueError(f"malformed detector entry at index {index}") from exc
        if len(coords) != 4 or not math.isfinite(confidence):
            raise ValueError(f"invalid detector entry at index {index}")
        out.append({"bbox": coords, "label": str(label), "confidence": round(confidence, 4)})
    return out


def op_detect(paths: list[str], with_wd14: bool = False) -> dict[str, Any]:
    """Run all defect-relevant detectors (hands/eyes/faces/heads/persons/text/feet) plus anime_completeness
    over MULTIPLE images in a single process — see the module docstring `detect` entry for the full
    contract. Inputs are decoded and collision-checked before any model import. Detector failures
    remain explicit per-image errors; the caller must treat any such record as an inference failure,
    never as a clean scan."""
    try:
        identities = _validate_batch_images(paths)
    except BatchInputError as exc:
        return {"ok": False, "results": {}, "reason": str(exc)}
    try:
        from imgutils.detect import detect_eyes, detect_faces, detect_hands, detect_heads, detect_person
    except Exception as exc:  # venv missing / wrong interpreter — nothing else will work either
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}

    detectors = [
        ("hands", detect_hands, {"level": "s", "conf_threshold": 0.35}),
        ("eyes", detect_eyes, {"level": "s", "conf_threshold": 0.3}),
        ("faces", detect_faces, {"level": "s", "conf_threshold": 0.25}),
        ("heads", detect_heads, {}),
        ("persons", detect_person, {"level": "m"}),
    ]
    text_import_error = None
    try:
        from imgutils.detect import detect_text
        detectors.append(("text", detect_text, {}))
    except Exception as exc:
        text_import_error = f"imgutils import failed: {type(exc).__name__}"

    # feet: no dedicated imgutils foot detector exists; NudeNet's FEET_* classes give tight
    # foot bboxes (verified on-sample 2026-07-16). Other NudeNet classes are dropped here.
    feet_import_error = None
    try:
        from imgutils.detect.nudenet import detect_with_nudenet

        def _detect_feet(path: str, conf_threshold: float = 0.3):
            return [(bbox, "foot", conf) for bbox, label, conf in detect_with_nudenet(path)
                    if label.startswith("FEET_") and conf >= conf_threshold]

        detectors.append(("feet", _detect_feet, {}))
    except Exception as exc:
        feet_import_error = f"imgutils import failed: {type(exc).__name__}"

    try:
        from imgutils.validate import anime_completeness
    except Exception as exc:
        anime_completeness = None
        completeness_import_error = f"imgutils import failed: {type(exc).__name__}"
    else:
        completeness_import_error = None

    wd14_fn = None
    wd14_import_error = None
    if with_wd14:
        try:
            from imgutils.tagging import get_wd14_tags
            wd14_fn = get_wd14_tags
        except Exception as exc:
            wd14_import_error = f"imgutils import failed: {type(exc).__name__}"

    results: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {"input_identity": identities[path]}
        errors: dict[str, str] = {}
        for name, fn, kwargs in detectors:
            try:
                entry[name] = _boxes(fn(path, **kwargs))
            except Exception as exc:
                entry[name] = []
                errors[name] = f"{type(exc).__name__}: {exc}"[:160]
        if text_import_error:
            entry["text"] = []
            errors["text"] = text_import_error
        if feet_import_error:
            entry["feet"] = []
            errors["feet"] = feet_import_error

        if anime_completeness is None:
            entry["completeness"] = None
            errors["completeness"] = completeness_import_error
        else:
            try:
                label, score = anime_completeness(path)
                entry["completeness"] = {"label": str(label), "score": round(float(score), 4)}
            except Exception as exc:
                entry["completeness"] = None
                errors["completeness"] = f"{type(exc).__name__}: {exc}"[:160]

        if with_wd14:
            if wd14_fn is None:
                errors["wd14"] = wd14_import_error
            else:
                try:
                    # general_threshold lowered from the library default (0.35) so low-confidence
                    # defect-tag calibration (down to a 0.1 floor) is actually visible to the caller.
                    rating, general, character = wd14_fn(path, general_threshold=0.1)
                    entry["wd14"] = {
                        "rating": {str(k): round(float(v), 4) for k, v in dict(rating).items()},
                        "general": {str(k): round(float(v), 4) for k, v in dict(general).items()},
                        "character": {str(k): round(float(v), 4) for k, v in dict(character).items()},
                    }
                except Exception as exc:
                    errors["wd14"] = f"{type(exc).__name__}: {exc}"[:160]

        if errors:
            entry["errors"] = errors
        results[path] = entry
    failed = [path for path, result in results.items() if result.get("errors")]
    return {
        "ok": not failed,
        "results": results,
        "reason": "" if not failed else f"detector inference failed for {len(failed)} image(s)",
    }


def op_ccip(path_a: str, path_b: str | None = None) -> dict[str, Any]:
    """CCIP character-similarity distance between two images. Lower = closer; a
    distance below ``ccip_default_threshold()`` means the same character."""
    if not path_b:
        return {"ok": False, "reason": "ccip requires two image paths"}
    try:
        from imgutils.metrics import ccip_default_threshold, ccip_difference
    except Exception as exc:  # venv missing / wrong interpreter
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        difference = float(ccip_difference(path_a, path_b))
        threshold = float(ccip_default_threshold())
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("ccip error", exc)}
    return {
        "ok": True,
        "difference": round(difference, 6),
        "threshold": round(threshold, 6),
        "same_character": difference < threshold,
        "reason": "",
    }


def op_ccip_multi(reference: str, paths: list[str]) -> dict[str, Any]:
    """Batch CCIP with the single-image result shape preserved under each candidate key."""
    try:
        _inspect_image(reference)
        identities = _validate_batch_images(paths)
    except BatchInputError as exc:
        return {"ok": False, "results": {}, "reason": str(exc)}
    try:
        from imgutils.metrics import ccip_default_threshold, ccip_difference
        threshold = float(ccip_default_threshold())
    except Exception as exc:
        return {"ok": False, "results": {},
                "reason": f"imgutils import failed: {type(exc).__name__}"}
    results: dict[str, Any] = {}
    for path in paths:
        try:
            difference = float(ccip_difference(reference, path))
            results[path] = {
                "ok": True,
                "difference": round(difference, 6),
                "threshold": round(threshold, 6),
                "same_character": difference < threshold,
                "input_identity": identities[path],
                "reason": "",
            }
        except Exception as exc:
            results[path] = {"ok": False, "input_identity": identities[path],
                             "reason": _op_reason("ccip error", exc)}
    failed = [path for path, result in results.items() if result.get("ok") is not True]
    return {"ok": not failed, "results": results,
            "reason": "" if not failed else f"ccip failed for {len(failed)} image(s)"}


def op_graycorr(path_a: str, path_b: str | None = None) -> dict[str, Any]:
    """Grayscale Pearson correlation between two images at a common downscale.

    The composition-fidelity metric of the Forge repro line (docs/forge_line.md):
    both images are converted to L and resized to a shared 256x256 grid, then the
    pixel vectors are correlated. 1.0 = identical composition; the same-kernel
    cross-machine reference band is ~0.75-0.79."""
    if not path_b:
        return {"ok": False, "reason": "graycorr requires two image paths"}
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return {"ok": False, "reason": f"PIL/numpy import failed: {type(exc).__name__}"}
    try:
        vectors = []
        for path in (path_a, path_b):
            with Image.open(path) as im:
                gray = im.convert("L").resize((256, 256), Image.LANCZOS)
            vectors.append(np.asarray(gray, dtype=np.float64).ravel())
        a, b = vectors
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            return {"ok": False, "reason": "flat image (zero variance) — correlation undefined"}
        correlation = float(np.corrcoef(a, b)[0, 1])
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("graycorr error", exc)}
    return {"ok": True, "correlation": round(correlation, 6), "reason": ""}


def op_colorcorr(path_a: str, path_b: str | None = None) -> dict[str, Any]:
    """Colour-palette correlation between two images: per-channel RGB histograms
    (64 bins each, density-normalized) concatenated and Pearson-correlated.

    The palette half of the style signal in the reproduction loop — graycorr sees
    spatial layout, this sees colour distribution (composition-invariant)."""
    if not path_b:
        return {"ok": False, "reason": "colorcorr requires two image paths"}
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return {"ok": False, "reason": f"PIL/numpy import failed: {type(exc).__name__}"}
    try:
        vectors = []
        for path in (path_a, path_b):
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB").resize((256, 256), Image.LANCZOS),
                                 dtype=np.float64)
            hists = [np.histogram(rgb[..., ch], bins=64, range=(0, 255), density=True)[0]
                     for ch in range(3)]
            vectors.append(np.concatenate(hists))
        a, b = vectors
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            return {"ok": False, "reason": "flat histogram (zero variance) — correlation undefined"}
        correlation = float(np.corrcoef(a, b)[0, 1])
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("colorcorr error", exc)}
    return {"ok": True, "correlation": round(correlation, 6), "reason": ""}


def op_lpips(path_a: str, path_b: str | None = None) -> dict[str, Any]:
    """LPIPS perceptual difference between two images. Lower = perceptually closer.

    Advisory metric of the reproduction loop (never a gate): unlike graycorr it sees
    texture and colour, so it complements the grayscale composition signal."""
    if not path_b:
        return {"ok": False, "reason": "lpips requires two image paths"}
    try:
        from imgutils.metrics import lpips_difference
    except Exception as exc:  # venv missing / wrong interpreter
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        difference = float(lpips_difference(path_a, path_b))
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("lpips error", exc)}
    return {"ok": True, "difference": round(difference, 6), "reason": ""}


def op_qc(path: str) -> dict[str, Any]:
    """No-reference quality signals for best-of-N / QC gating.

    `aesthetic_percentile` (0..1, calibrated against a reference distribution) is the primary
    quality metric — the discriminative one. The coarse `aesthetic_label` (worst..masterpiece)
    SATURATES at 'best'/'masterpiece' for most decent gens, so it is reported for readability but
    not used for ranking. `completeness` (polished/rough/monochrome) and `monochrome` (0..1) catch
    defect classes the aesthetic misses (sketchy/unfinished, accidental greyscale). face/hand/person
    counts feed anatomy-sanity flags; faces is the reliable subject count (person detection over-
    fires on background art). NOTE: none of these judge anatomy correctness (bad hands) or the right
    subject count — that needs the caller's --subjects check, and ultimately CCIP / a human."""
    try:
        from imgutils.detect import detect_faces, detect_hands, detect_person
        from imgutils.metrics import anime_dbaesthetic
        from imgutils.validate import anime_completeness, get_monochrome_score
    except Exception as exc:  # venv missing / wrong interpreter
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        label, percentile = anime_dbaesthetic(path)  # default fmt = (label, percentile)
        completeness, _ = anime_completeness(path)
        out = {"ok": True, "aesthetic_percentile": round(float(percentile), 4),
               "aesthetic_label": str(label), "completeness": str(completeness),
               "monochrome": round(float(get_monochrome_score(path)), 4),
               "faces": len(detect_faces(path)), "hands": len(detect_hands(path)),
               "persons": len(detect_person(path)), "reason": ""}
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("qc error", exc)}
    return out


def op_qc_multi(paths: list[str]) -> dict[str, Any]:
    """Batch no-reference QC: validate first, import once, then reuse cached model sessions."""
    try:
        identities = _validate_batch_images(paths)
    except BatchInputError as exc:
        return {"ok": False, "results": {}, "reason": str(exc)}
    try:
        from imgutils.detect import detect_faces, detect_hands, detect_person
        from imgutils.metrics import anime_dbaesthetic
        from imgutils.validate import anime_completeness, get_monochrome_score
    except Exception as exc:
        return {"ok": False, "results": {},
                "reason": f"imgutils import failed: {type(exc).__name__}"}
    results: dict[str, Any] = {}
    for path in paths:
        try:
            label, percentile = anime_dbaesthetic(path)
            completeness, _ = anime_completeness(path)
            results[path] = {
                "ok": True,
                "aesthetic_percentile": round(float(percentile), 4),
                "aesthetic_label": str(label),
                "completeness": str(completeness),
                "monochrome": round(float(get_monochrome_score(path)), 4),
                "faces": len(detect_faces(path)),
                "hands": len(detect_hands(path)),
                "persons": len(detect_person(path)),
                "input_identity": identities[path],
                "reason": "",
            }
        except Exception as exc:
            results[path] = {"ok": False, "input_identity": identities[path],
                             "reason": _op_reason("qc error", exc)}
    failed = [path for path, result in results.items() if result.get("ok") is not True]
    return {"ok": not failed, "results": results,
            "reason": "" if not failed else f"qc failed for {len(failed)} image(s)"}


def op_segment(path: str, out: str | None = None) -> dict[str, Any]:
    """Matte the anime subject onto a transparent background (isnetis) and write an RGBA PNG."""
    if not out:
        return {"ok": False, "reason": "segment requires an output path"}
    try:
        from imgutils.segment import segment_rgba_with_isnetis
    except Exception as exc:  # venv missing / wrong interpreter
        return {"ok": False, "reason": f"imgutils import failed: {type(exc).__name__}"}
    try:
        _mask, rgba = segment_rgba_with_isnetis(path)
        rgba.save(out)
    except Exception as exc:
        return {"ok": False, "reason": _op_reason("segment error", exc)}
    return {"ok": True, "out": out, "size": list(rgba.size), "reason": ""}


# "detect" is dispatched specially in main() (it takes a path list, not a single image) but is listed
# here too so it appears in `op` choices.
OPS = {"lsb": op_lsb, "wd14": op_wd14, "teen": op_teen, "ccip": op_ccip, "qc": op_qc,
       "segment": op_segment, "detect": op_detect, "graycorr": op_graycorr,
       "lpips": op_lpips, "colorcorr": op_colorcorr}


def main() -> None:
    parser = argparse.ArgumentParser(description="imgutils venv bridge")
    parser.add_argument("op", choices=sorted(OPS))
    parser.add_argument("image")
    parser.add_argument("image2", nargs="?",
                        help="second image (ccip) / output path (segment) / extra image (detect, wd14)")
    parser.add_argument("more_images", nargs="*",
                        help="additional image paths — detect / wd14 multi-image batch only")
    parser.add_argument("--with-wd14", action="store_true",
                        help="detect op only: also attach WD14 tags per image (one subprocess call)")
    args = parser.parse_args()
    try:
        if args.op == "detect":
            paths = [args.image] + ([args.image2] if args.image2 else []) + list(args.more_images)
            out = op_detect(paths, with_wd14=args.with_wd14)
        elif args.op == "wd14" and (args.image2 or args.more_images):
            paths = [args.image] + ([args.image2] if args.image2 else []) + list(args.more_images)
            out = op_wd14_multi(paths)
        elif args.op == "qc" and (args.image2 or args.more_images):
            paths = [args.image] + ([args.image2] if args.image2 else []) + list(args.more_images)
            out = op_qc_multi(paths)
        elif args.op == "teen" and (args.image2 or args.more_images):
            paths = [args.image] + ([args.image2] if args.image2 else []) + list(args.more_images)
            out = op_teen_multi(paths)
        elif args.op == "ccip" and args.more_images:
            paths = ([args.image2] if args.image2 else []) + list(args.more_images)
            out = op_ccip_multi(args.image, paths)
        elif args.op in ("ccip", "segment", "graycorr", "lpips", "colorcorr"):
            out = OPS[args.op](args.image, args.image2)
        else:
            out = OPS[args.op](args.image)
    except Exception as exc:  # last-resort guard — caller only ever sees clean JSON
        out = {"ok": False, "reason": f"unexpected: {type(exc).__name__}: {exc}"}
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
