#!/usr/bin/env python3
"""Compatibility shim for the shared ComfyUI driver helpers now in loramake.comfy."""
from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:5]):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate repo root containing loramake/__init__.py")


root = _repo_root()
root_s = str(root)
if root_s not in sys.path:
    sys.path.insert(0, root_s)

from loramake import comfy  # noqa: E402

sys.modules[__name__] = comfy
