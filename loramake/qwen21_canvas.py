"""Qwen-Image-2.1 canvas table, shared by the engine and the app.

Only user-judged canvases are listed (2026-09-21, six canvases + 2K); an unlisted
combination is an error, not a guess. 832x1216 / 1216x832 are the nearest 16-grid
canvases to 2:3 / 3:2 (0.684 / 1.462), not exact.

Pure data with no imports, so ``app.brief`` can read the table without pulling in
``scripts/qwen21_gen.py`` (and ``gen.py`` behind it).
"""
from __future__ import annotations

RATIOS = ("1:1", "2:3", "3:2")
SIZES = ("draft", "std", "large", "2k")
DEFAULT_RATIO, DEFAULT_SIZE = "1:1", "std"
# (ratio, size) -> (width, height)
CANVAS: dict[tuple[str, str], tuple[int, int]] = {
    ("1:1", "draft"): (768, 768),
    ("1:1", "std"): (1024, 1024),
    ("1:1", "2k"): (2048, 2048),
    ("2:3", "std"): (832, 1216),
    ("2:3", "large"): (1024, 1536),
    ("3:2", "std"): (1216, 832),
    ("3:2", "large"): (1536, 1024),
}
