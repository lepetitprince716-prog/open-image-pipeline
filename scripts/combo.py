#!/usr/bin/env python3
"""N4 random-combo engine: outfit x action (x scene) sampling with reproduction.

Five-layer decoupling plan N4: a seeded RNG samples a character-neutral outfit card
(scripts/outfit_cards.json) and a text action (wildcards/pose.txt entries with inline
metadata) under two filters — the action's motion_class must be allowed by the outfit
card's motion_allow, and the SFW bit must match the caller's line. The drawn combo is
recorded so a good image can be exactly replayed: same seed + same combo + same command.

Slot derivation keeps the draws independent per dimension (seed -> "outfit"/"action"/
"scene" sub-seeds) and fully deterministic. v1 scene = pool sampling only (no scene cards).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WILDCARDS = ROOT / "wildcards"

MOTION_CLASSES = ("static", "seated", "reclining", "dance", "dynamic")


def _subseed(seed: int, label: str) -> int:
    """Deterministic per-dimension derivation from the caller seed (no builtin hash():
    PYTHONHASHSEED would break cross-process reproducibility)."""
    digest = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")
    return (seed * 0x9E3779B1 + digest) & 0xFFFFFFFF


def _pool_lines(name: str) -> list[str]:
    """Non-comment pool lines with their inline metadata parsed."""
    path = WILDCARDS / f"{name}.txt"
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        text, _, meta = line.partition(" # ")
        tags: dict[str, str] = {}
        for part in meta.split(","):
            part = part.strip()
            if ":" in part:
                key, _, value = part.partition(":")
                tags[key.strip()] = value.strip()
        entries.append({"text": text.strip(), "id": tags.get("id"), "motion_class": tags.get("motion"),
                        "register": tags.get("register"), "raw": line})
    return entries


def sample(seed: int, *, outfit_id: str | None = None, action_id: str | None = None,
           scene: bool = False, sfw_only: bool = True,
           exclude_register: set[str] | None = None) -> dict:
    """Draw one combo. ``outfit_id`` / ``action_id`` pin a specific card/action (None =
    random); ``scene`` draws background + time-of-day pool entries. Every draw uses a
    sub-seed derived from ``seed``, so the same arguments replay the same combo."""
    import outfit_cards as oc

    exclude = set(exclude_register or ())
    outfits = {oid: c for oid, c in oc.cards().items()
               if (not sfw_only or c["sfw"])}
    if not outfits:
        raise ValueError("no outfit cards match the sfw filter")

    if outfit_id is not None:
        outfit = oc.get(outfit_id)
        if outfit is None or (sfw_only and not outfit["sfw"]):
            raise ValueError(f"outfit {outfit_id!r} unavailable under sfw_only={sfw_only}")
    else:
        rng = random.Random(_subseed(seed, "outfit"))
        outfit_id = rng.choice(sorted(outfits))
        outfit = {"id": outfit_id, **outfits[outfit_id]}

    allowed_motion = set(outfit["motion_allow"])
    actions = [e for e in _pool_lines("pose")
               if e["motion_class"] in allowed_motion
               and (e["register"] is None or e["register"] not in exclude)]
    if not actions:
        raise ValueError(f"no actions compatible with outfit {outfit['id']} motion_allow "
                         f"{sorted(allowed_motion)}")
    if action_id is not None:
        action = next((e for e in actions if e["id"] == action_id), None)
        if action is None:
            raise ValueError(f"action {action_id!r} incompatible with outfit "
                             f"{outfit['id']} or unknown")
    else:
        action = random.Random(_subseed(seed, "action")).choice(actions)

    scene_draw = {}
    if scene:
        rng = random.Random(_subseed(seed, "scene"))
        bg = [e["text"] for e in _pool_lines("background")]
        tod = [e["text"] for e in _pool_lines("time_of_day")]
        scene_draw = {"background": rng.choice(bg), "time_of_day": rng.choice(tod)}

    return {
        "schema": "loramake.combo.v1",
        "seed": seed,
        "outfit_id": outfit["id"],
        "outfit_name_cn": outfit.get("name_cn"),
        "action_id": action.get("id"),
        "action_text": action["text"],
        "motion_class": action.get("motion_class"),
        "scene": scene_draw,
        "filters": {"motion_allow": sorted(allowed_motion), "sfw_only": sfw_only,
                    "exclude_register": sorted(exclude)},
    }


def sample_many(seed: int, n: int, **kwargs) -> list[dict]:
    """n independent combo draws (each with its own sub-seeds from seed+i)."""
    return [sample(seed + i, **kwargs) for i in range(n)]


def cmd_sample(args: argparse.Namespace) -> None:
    combos = sample_many(args.seed, args.n, outfit_id=args.outfit, action_id=args.action,
                         scene=args.scene, sfw_only=not args.nsfw)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(combos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in combos:
        scene = f" + bg={c['scene']['background']}" if c["scene"] else ""
        print(f"[combo] seed={c['seed']} outfit={c['outfit_id']} "
              f"action={c['action_id']} ({c['motion_class']}){scene}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--outfit", default=None, help="pin an outfit card id (default: random)")
    ap.add_argument("--action", default=None, help="pin an action id (default: random)")
    ap.add_argument("--scene", action="store_true", help="also draw background + time-of-day")
    ap.add_argument("--nsfw", action="store_true", help="allow sfw=False outfit cards")
    ap.add_argument("--json-out", default=None, help="write the draws to a JSON manifest")
    args = ap.parse_args()
    cmd_sample(args)


if __name__ == "__main__":
    main()
