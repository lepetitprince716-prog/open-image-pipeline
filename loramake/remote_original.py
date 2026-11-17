"""Fetch the full-resolution original of an image that arrived as a preview.

Off the LAN, engines keep only ComfyUI's webp preview (re-encoded to PNG) and record where the
original lives in the image's ``.meta.json`` as ``remote_original`` (box, filename, subfolder, type).
``fetch`` pulls that PNG from the box and replaces the local preview in place, so every existing
reference (job results, gallery paths) keeps pointing at the same file.

Some originals must never cross the link: the sidecar carries ``originals_stay_remote: true``
(the MOSS line writes it), or ``LORAMAKE_ORIGINALS_STAY_REMOTE=1`` holds every original on its box.
``fetch`` refuses those without touching the network; ``stays_remote`` lets callers say why.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from loramake import comfy, net_routes

STAY_REMOTE_ENV = "LORAMAKE_ORIGINALS_STAY_REMOTE"
STAYS_REMOTE_DETAIL = ("the original stays on the render box (originals_stay_remote); "
                       "only its preview is kept here")


def _meta_path(image: Path) -> Path:
    return Path(str(image) + ".meta.json")


def _meta(image: Path) -> dict:
    try:
        meta = json.loads(_meta_path(Path(image)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def pending(image: Path) -> dict | None:
    """The ``remote_original`` record when ``image`` is still a preview, else None."""
    record = _meta(image).get("remote_original")
    if not isinstance(record, dict) or not record.get("box") or not record.get("filename"):
        return None
    return record


def stays_remote(image: Path) -> bool:
    """True when ``image`` is a preview whose original must not be fetched from its box."""
    if pending(image) is None:
        return False
    return os.environ.get(STAY_REMOTE_ENV) == "1" or _meta(image).get("originals_stay_remote") is True


def fetch(image: Path) -> bool:
    """Replace a local preview with its original. True when ``image`` is now (or already was) full
    size; False when the box is unreachable or the original stays there (``stays_remote``)."""
    image = Path(image)
    record = pending(image)
    if record is None:
        return True
    if stays_remote(image):
        return False
    meta = _meta(image)
    host = net_routes.comfy_host(record["box"]).rstrip("/")
    raw = comfy.curl(["-G", f"{host}/view",
                      "--data-urlencode", f"filename={record['filename']}",
                      "--data-urlencode", f"subfolder={record.get('subfolder') or ''}",
                      "--data-urlencode", f"type={record.get('type') or 'output'}"],
                     binary=True, host=host)
    if not comfy.looks_like_png(raw):
        print(f"  original of {image.name} not fetched from {record['box']} ({len(raw)} bytes)",
              file=sys.stderr)
        return False
    meta.pop("remote_original", None)
    meta.pop("fetched_preview", None)
    meta["original_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    comfy.save_image_with_meta(image, raw, meta, overwrite=True)
    return True
