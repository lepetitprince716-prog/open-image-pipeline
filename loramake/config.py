"""Import-time configuration constants shared by loramake scripts."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from loramake.comfy_endpoints import DIRECT_COMFY_HOSTS, resolve_comfy_host


def _repo_root() -> Path:
    """Return the repository root containing ``loramake/__init__.py``.

    Walk upward from this file location with a small fixed limit.  This does not
    depend on ``.git`` because linked worktrees store ``.git`` as a file.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:5]):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate repo root containing loramake/__init__.py")


Z3_HOST = resolve_comfy_host("z3", explicit=os.environ.get("Z3_HOST"))
ZZZ_HOST = resolve_comfy_host("zzz", explicit=os.environ.get("ZZZ_HOST"))
# Fall back to the canonical table rather than repeating its literal here:
# comfy_endpoints is meant to be the only place a host address is written down.
NP = urlparse(Z3_HOST).hostname or urlparse(DIRECT_COMFY_HOSTS["z3"]).hostname
POLL_TIMEOUT = float(os.environ.get("Z3_POLL_TIMEOUT", "1800"))  # wall-clock seconds before a poll gives up

ROOT = _repo_root()


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve an optional project-local path override without changing defaults.

    Consumer projects can borrow loramake's engines while keeping every render in
    their own repository. Relative overrides remain anchored to the loramake root
    so subprocesses do not change behaviour merely because their cwd changed.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


OUTPUTS = _path_from_env("LORAMAKE_OUTPUTS", ROOT / "outputs")
OUTPUT_CHARACTER = os.environ.get("LORAMAKE_OUTPUT_CHARACTER", "character").strip() or "character"


def by_model_dir(character: str, line: str = "", *, model: str | None = None) -> Path:
    """Return ``outputs/<character>/by_model/<model>[/<line>]``.

    ``model=None`` reads ``DRIVING_MODEL`` at call time and defaults to
    ``claude``.  No normalization is applied here; callers pass their existing
    import-time model value when they need to preserve script-specific behavior.
    """
    m = model if model is not None else os.environ.get("DRIVING_MODEL", "claude")
    base = OUTPUTS / character / "by_model" / m
    return base / line if line else base
