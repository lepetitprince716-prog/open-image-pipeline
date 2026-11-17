"""Content zone (SFW / NSFW) of derived files: thumbnails, frame dumps, caches, staging.

A derived file belongs to the zone of its source. Ratings follow directory boundaries,
never content inspection, so the test is a path test:

- ``sfw_relpath`` is the allowlist. Only a source under an SFW render root, with no
  NSFW-marked path component, counts as SFW. Everything else -- nsfw/, trash, review
  snapshots, probe trees, other checkouts, paths outside the repository -- belongs to
  the NSFW side (fail closed), and its derivatives go under NSFW_CACHE.
- ``nsfw_marked`` is the denylist, for places that cannot use the allowlist, such as
  copies of media from outside the repository, remote references or archive prefixes.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import OUTPUT_CHARACTER, OUTPUTS, ROOT

# The SFW render roots; scripts/gallery_scan.py builds its SFW gallery sources from them.
# The character tree, the Anima free-exploration line home (boxa_anima_fast.py OUTDIR) and the
# MOSS-Image previews (scripts/moss_gen.py writes a preview only after its wd14 gate passes it).
SFW_ROOTS = (OUTPUTS / OUTPUT_CHARACTER, OUTPUTS / "anima", OUTPUTS / "moss")
# Derivatives of anything that is not provably SFW land here, never under outputs/.
NSFW_CACHE = ROOT / "nsfw" / "app_cache"


def nsfw_marked(path: str) -> bool:
    """Whether any component of a path names an NSFW tree (``nsfw``, ``qwennsfw``,
    ``nsfw_scout_...``)."""
    return any("nsfw" in part.casefold() for part in re.split(r"[\\/]", path))


def sfw_relpath(path: Path) -> str | None:
    """``outputs/<root>/...`` for a source under an SFW render root, else None.

    The returned path lets an SFW derivative name its source, so it can be traced back.
    Symlinks are judged by their target.
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        shown = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        shown = resolved.as_posix()
    if nsfw_marked(shown):
        return None
    for root in SFW_ROOTS:
        try:
            rel = resolved.relative_to(root.resolve())
        except (OSError, RuntimeError, ValueError):
            continue
        if rel.parts:
            return f"outputs/{root.name}/{rel.as_posix()}"
    return None
