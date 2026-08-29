"""Soft-archive marks for catalog LoRAs.

Deletion never touches weight files or the generated z3_index.json (both owned by
other tools); archiving is a reversible mark kept in a private JSON sidecar. Marked
entries stay in the catalog projection with archived=true so the models page can
show and undo them, while pickers hide them by default.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "app" / "data" / "lora_archive.json"
PATH = DEFAULT_PATH

_lock = threading.Lock()


def _load(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def archived_files(path: Path | None = None) -> set[str]:
    """Engine-spec file names currently marked archived."""
    return set(_load(Path(path or PATH)))


def set_archived(file: str, archived: bool, path: Path | None = None) -> bool:
    """Mark or unmark one catalog file. Returns the resulting archived state."""
    target = Path(path or PATH)
    with _lock:
        data = _load(target)
        if archived:
            data[file] = {"archived_at": time.time()}
        else:
            data.pop(file, None)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(target)
    return archived
