"""Round ledger derived from what already exists on disk.

A round is one review batch. Everything the ledger shows is read from the files the round
left behind, so nothing has to be maintained by hand: the batch manifest and result, the
Buzz ledger lines that name the batch, the spec that produced it (probe, deck or explore
pool), the gate report next to the renders, and the registry realizations and NovelAI
library cards whose evidence cites the batch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import threading

SPEC_GLOBS = ("configs/civitai_probes/*.json",)
GATE_REPORTS = ("gates_report.json", "gate_report.json")
PROBE_SPEC = "probe_spec.json"


_JSON_LOCK = threading.Lock()
_JSON_CACHE: dict[str, tuple[tuple[int, int], dict | list | None]] = {}


def _read_json(path: Path) -> dict | list | None:
    try:
        sig = (path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        return None
    key = str(path)
    with _JSON_LOCK:
        hit = _JSON_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        data = None
    with _JSON_LOCK:
        _JSON_CACHE[key] = (sig, data)
    return data


def _verdicts(result: dict | None) -> dict:
    if not isinstance(result, dict):
        return {"love": 0, "yes": 0, "no": 0, "comment": 0}
    return {"love": len(result.get("l", [])), "yes": len(result.get("y", [])),
            "no": len(result.get("n", [])), "comment": len(result.get("c", {}))}


_LEDGER_LOCK = threading.Lock()
_LEDGER_CACHE: dict[str, tuple[tuple[int, int], dict[str, dict]]] = {}


def spend_by_batch(ledger_path: Path) -> dict[str, dict]:
    """Buzz spent and renders charged per batch, from the JSONL ledger."""
    totals: dict[str, dict] = {}
    if not ledger_path.is_file():
        return totals
    sig = (ledger_path.stat().st_mtime_ns, ledger_path.stat().st_size)
    key = str(ledger_path)
    with _LEDGER_LOCK:
        hit = _LEDGER_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        batch = record.get("batch")
        if not isinstance(batch, str):
            continue
        entry = totals.setdefault(batch, {"buzz": 0, "renders": 0})
        cost = record.get("cost_buzz")
        entry["buzz"] += int(cost) if isinstance(cost, (int, float)) else 0
        entry["renders"] += 1
    with _LEDGER_LOCK:
        _LEDGER_CACHE[key] = (sig, totals)
    return totals


def specs_by_batch(repo: Path) -> dict[str, str]:
    """Spec file (relative path) per batch id, for every spec kind that names its batch."""
    found: dict[str, str] = {}
    for pattern in SPEC_GLOBS:
        for path in sorted(repo.glob(pattern)):
            data = _read_json(path)
            if isinstance(data, dict):
                batch = data.get("batch") or data.get("review_batch")
                if isinstance(batch, str) and batch not in found:
                    found[batch] = str(path.relative_to(repo))
    return found


def registry_citations(cards_dir: Path) -> dict[str, int]:
    """Realizations per batch whose evidence ref points at the batch."""
    counts: dict[str, int] = {}
    for path in sorted(cards_dir.glob("*.json")):
        card = _read_json(path)
        if not isinstance(card, dict):
            continue
        for realizations in (card.get("realizations") or {}).values():
            for real in realizations if isinstance(realizations, list) else []:
                ref = (real.get("evidence") or {}).get("ref") if isinstance(real, dict) else None
                if isinstance(ref, str) and "/review_batches/" in ref:
                    batch = ref.rsplit("/review_batches/", 1)[1].strip("/")
                    counts[batch] = counts.get(batch, 0) + 1
    return counts


def library_citations(library_path: Path, batches: list[str]) -> dict[str, int]:
    """NovelAI library cards whose note names the batch."""
    counts: dict[str, int] = {}
    data = _read_json(library_path) if library_path.is_file() else None
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, dict):
        return counts
    for card in cards.values():
        note = card.get("note", "") if isinstance(card, dict) else ""
        for batch in batches:
            if batch in note:
                counts[batch] = counts.get(batch, 0) + 1
    return counts


def _first_item_parents(repo: Path, manifest: dict) -> list[Path]:
    """Directories above the batch's first image, nearest first, stopping at the repository."""
    items = manifest.get("items") or []
    first = items[0].get("path") if items and isinstance(items[0], dict) else None
    if not isinstance(first, str):
        return []
    parents = []
    for parent in (repo / first).parents:
        parents.append(parent)
        if parent == repo:
            break
    return parents


def gate_summary(repo: Path, manifest: dict) -> dict | None:
    """Overall pass/fail counts from the gate report next to the batch's renders, when present."""
    for parent in _first_item_parents(repo, manifest):
        for name in GATE_REPORTS:
            data = _read_json(parent / name) if (parent / name).is_file() else None
            if not isinstance(data, dict):
                continue
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else summary
            out = {key: overall[key] for key in ("pass", "fail", "unverified") if isinstance(overall.get(key), int)}
            return out or None
    return None


def spec_for(repo: Path, batch: str, manifest: dict, specs: dict[str, str], decks: dict[str, str]) -> str | None:
    """The spec that produced a batch: by batch id, else a probe_spec.json or concept deck above its renders."""
    if batch in specs:
        return specs[batch]
    for parent in _first_item_parents(repo, manifest):
        probe = parent / PROBE_SPEC
        if probe.is_file():
            return str(probe.relative_to(repo))
        for deck, deck_path in decks.items():
            if deck.startswith(parent.name) and parent != repo:
                return deck_path
    return None


def decks_by_id(repo: Path) -> dict[str, str]:
    """Concept deck id -> deck file, for batches rendered from a deck."""
    decks: dict[str, str] = {}
    for path in sorted(repo.glob("configs/concept_decks/*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("deck"), str):
            decks[data["deck"]] = str(path.relative_to(repo))
    return decks


def collect(*, review_root: Path, repo: Path, ledger_path: Path, cards_dir: Path, library_path: Path) -> list[dict]:
    """Every review batch as a round record, newest first."""
    if not review_root.is_dir():
        return []
    spend = spend_by_batch(ledger_path)
    specs = specs_by_batch(repo)
    decks = decks_by_id(repo)
    registry = registry_citations(cards_dir)
    batches = sorted(entry.name for entry in review_root.iterdir()
                     if entry.is_dir() and not entry.is_symlink() and (entry / "manifest.json").is_file())
    library = library_citations(library_path, batches)
    rounds: list[dict] = []
    for batch in batches:
        batch_dir = review_root / batch
        manifest = _read_json(batch_dir / "manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("items"), list):
            continue
        result = _read_json(batch_dir / "result.json") if (batch_dir / "result.json").is_file() else None
        verdicts = _verdicts(result)
        total = len(manifest["items"])
        decided = sum(verdicts.values())
        created = manifest.get("created_at")
        if not isinstance(created, str):
            created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((batch_dir / "manifest.json").stat().st_mtime))
        mode = manifest.get("mode", "review")
        rounds.append({
            "batch": batch, "created": created, "mode": mode, "total": total, "decided": decided,
            "pending": mode == "review" and decided < total, "verdicts": verdicts,
            "spend": spend.get(batch), "spec": spec_for(repo, batch, manifest, specs, decks),
            "gates": gate_summary(repo, manifest),
            "writeback": {"registry": registry.get(batch, 0), "library": library.get(batch, 0)},
        })
    rounds.sort(key=lambda row: row["created"], reverse=True)
    return rounds
