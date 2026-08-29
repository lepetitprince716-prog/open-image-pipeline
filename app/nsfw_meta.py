"""NSFW meta loader for app (M6).

Loads NSFW presets from nsfw/app_presets.json (Grok content handoff).
- Tolerant: skips bad entries, ignores unknown fields.
- Schema per handoff: {id, name_cn, intent, tier, prompt, loras?, n?, box?, note?}
- CLI: --check (exit 0/1 + summary), default prints count.

Consumed by server.py /api/nsfw/meta (and future presets UI tab).
Graceful when file missing: returns {"presets": []}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from . import brief as brief_mod
else:  # direct ``python app/nsfw_meta.py`` execution
    import brief as brief_mod

ROOT = Path(__file__).resolve().parent.parent
PRESETS_PATH = ROOT / "nsfw" / "app_presets.json"

TIERS = brief_mod.TIERS
BOXES = brief_mod.BOXES


def _load_raw() -> dict:
    if not PRESETS_PATH.exists():
        return {"version": 1, "presets": []}
    try:
        return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[nsfw_meta] failed to load {PRESETS_PATH}: {e}", file=sys.stderr)
        return {"version": 1, "presets": []}


def presets() -> list[dict]:
    """Return validated/tolerant list of preset dicts ready for Brief construction."""
    raw = _load_raw()
    out: list[dict] = []
    for item in raw.get("presets", []) or []:
        if not isinstance(item, dict):
            continue
        pid = item.get("id")
        name = item.get("name_cn")
        intent = item.get("intent")
        tier = item.get("tier")
        prompt = item.get("prompt")
        if not (pid and name and intent and tier in TIERS and isinstance(prompt, str) and prompt.strip()):
            # skip bad entry silently (tolerant)
            continue
        p: dict = {
            "id": str(pid),
            "name_cn": str(name),
            "intent": str(intent),
            "tier": tier,
            "prompt": prompt.strip(),
        }
        loras = item.get("loras")
        if isinstance(loras, list) and all(isinstance(x, str) and x.strip() for x in loras):
            p["loras"] = [x.strip() for x in loras][:brief_mod.MAX_EXTRA_LORAS]
        n = item.get("n")
        if isinstance(n, int) and 1 <= n <= brief_mod.MAX_N:
            p["n"] = n
        box = item.get("box")
        if box in BOXES:
            p["box"] = box
        note = item.get("note")
        if isinstance(note, str) and note.strip():
            p["note"] = note.strip()
        out.append(p)
    return out


def as_meta() -> dict:
    return {"presets": presets()}


def cmd_check(_args: argparse.Namespace) -> None:
    raw = _load_raw().get("presets", []) or []
    ps = presets()
    skipped = len(raw) - len(ps)
    if skipped:
        print(f"[nsfw_meta] FAIL: {skipped} of {len(raw)} entries invalid "
              f"(need id/name_cn/intent, tier in {TIERS}, non-empty prompt)", file=sys.stderr)
        raise SystemExit(1)
    print(f"[nsfw_meta] OK: {len(ps)} valid presets loaded from {PRESETS_PATH}")
    if ps:
        print("  ids:", [p["id"] for p in ps])


def main() -> None:
    ap = argparse.ArgumentParser(description="NSFW presets meta (for /api/nsfw/meta + UI)")
    ap.add_argument("--check", action="store_true", help="validate + print summary then exit")
    ap.add_argument("--json", action="store_true", help="emit full {presets: [...]} to stdout")
    args = ap.parse_args()

    if args.check:
        cmd_check(args)
        # also basic sanity on source file presence for CI
        if not PRESETS_PATH.exists():
            print("[nsfw_meta][WARN] presets file missing (will serve empty)", file=sys.stderr)
        return

    if args.json:
        print(json.dumps(as_meta(), ensure_ascii=False, indent=2))
        return

    # default human
    ps = presets()
    print(f"NSFW presets: {len(ps)} (source: {PRESETS_PATH})")
    for p in ps:
        print(f"  - {p['id']}: {p['name_cn']}")


if __name__ == "__main__":
    main()
