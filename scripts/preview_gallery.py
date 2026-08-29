#!/usr/bin/env python3
"""Generate a static HTML gallery for sidebar dev-server preview.

Images are referenced by their repo-root-relative URL and served over HTTP by
serve_preview.py (127.0.0.1:8770), so the page loads instantly with no base64 inlining.
This Claude-side port is kept distinct from the Hermes gallery server (127.0.0.1:18180)
so the two preview interfaces never collide.

Usage:
    python3 scripts/preview_gallery.py [--title TITLE] IMG [IMG ...]

Writes outputs/_preview/index.html and prints its server URL path.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from config import OUTPUTS, ROOT

REPO = ROOT
OUT = OUTPUTS / "_preview" / "index.html"


def rel_url(path: Path) -> str:
    """Repo-root-relative server URL (leading slash, POSIX separators)."""
    return "/" + path.resolve().relative_to(REPO).as_posix()


def _img_dims(path: Path) -> tuple[int, int] | None:
    """Intrinsic (width, height) of any image format via a header-only read; None if unreadable.
    Uses PIL (already a project dependency) so JPEG/WebP images saved with a .png suffix — e.g.
    Civitai outputs — get dimensions too, not just true PNGs."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def build_card(path: Path) -> str:
    url = rel_url(path)
    name = html.escape(path.name)
    # Intrinsic dimensions let the eager-loaded grid reserve each cell's aspect ratio up front, so the
    # page doesn't reflow/jump as images arrive. decoding="async" keeps decode off the main thread.
    # loading stays "eager" — lazy did not load reliably in the sidebar webview (commit 2ea19f6).
    dims = _img_dims(path)
    size_attr = f' width="{dims[0]}" height="{dims[1]}"' if dims else ""
    return (
        f'<figure><a href="{url}" target="_blank" rel="noopener">'
        f'<img loading="eager" decoding="async"{size_attr} src="{url}" alt="{name}"></a>'
        f"<figcaption>{name}</figcaption></figure>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Image preview")
    parser.add_argument("--full", action="store_true",
                        help="single-column view: each image at full width, scroll vertically through them")
    parser.add_argument("images", nargs="+",
                        help="image paths (repo-relative, or absolute under this repo)")
    args = parser.parse_args()

    cards = []
    for raw in args.images:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO / raw
        if not path.exists():
            continue
        # The preview server serves from the repo root, so rel_url() can only build a URL for files
        # under REPO. Skip anything outside it with a clear note instead of crashing in relative_to().
        try:
            path.resolve().relative_to(REPO)
        except ValueError:
            print(f"  skip (outside repo, cannot be served): {raw}", file=sys.stderr)
            continue
        cards.append(build_card(path))

    title = html.escape(args.title)
    # --full: one image per row at full panel width (scroll vertically). Default: responsive thumbnail grid.
    grid_cols = "1fr" if args.full else "repeat(auto-fill,minmax(220px,1fr))"
    full_css = "figure{max-width:1100px;width:100%;margin:0 auto}" if args.full else ""
    doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title><style>"
        ":root{color-scheme:light dark}"
        "body{margin:0;padding:16px;font:14px/1.5 -apple-system,system-ui,sans-serif;"
        "background:#0b0b0d;color:#e7e7ea}"
        "h1{font-size:15px;font-weight:500;margin:0 0 14px;color:#a9a9b2}"
        f".grid{{display:grid;gap:14px;grid-template-columns:{grid_cols}}}"
        "figure{margin:0;background:#161619;border:1px solid #26262b;border-radius:12px;overflow:hidden}"
        "img{display:block;width:100%;height:auto}"
        "figcaption{padding:8px 10px;font-size:12px;color:#8a8a93;"
        "font-family:ui-monospace,monospace;word-break:break-all}"
        "a{color:inherit;text-decoration:none}"
        f"{full_css}"
        "</style></head><body>"
        f"<h1>{title} &middot; {len(cards)} image(s)</h1>"
        f'<div class="grid">{"".join(cards)}</div>'
        "</body></html>"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(rel_url(OUT))


if __name__ == "__main__":
    main()
