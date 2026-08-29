#!/usr/bin/env python3
"""Build an offline, deterministic video direction package from a JSON brief."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import OUTPUTS, ROOT  # inserts the repository root on sys.path
from loramake.video_direction import (
    VideoDirectionError,
    atomic_publish_bundle,
    build_storyboard_manifest,
    json_bytes,
    load_json_object,
    normalize_direction,
    render_commands,
    render_shot_list,
)


def load_brief(path: str | Path) -> dict:
    _, value = load_json_object(path, label="brief JSON")
    return value


def plan_project(brief_path: str | Path, project_dir: str | Path | None = None) -> dict[str, Path]:
    source = Path(brief_path).expanduser().resolve()
    direction = normalize_direction(load_brief(source), base_dir=source.parent)
    target_dir = (
        Path(project_dir).expanduser().resolve()
        if project_dir is not None
        else (OUTPUTS / "_analysis" / "video_projects" / direction["project"]["id"]).resolve()
    )
    direction_path = target_dir / "direction.json"
    direction_raw = json_bytes(direction)
    manifest = build_storyboard_manifest(
        direction,
        direction_path=direction_path,
        direction_sha256=hashlib.sha256(direction_raw).hexdigest(),
        repo_root=ROOT,
        output_root=OUTPUTS,
    )
    paths = {
        "direction": direction_path,
        "shot_list": target_dir / "shot_list.md",
        "storyboard_manifest": target_dir / "storyboard_manifest.json",
        "commands": target_dir / "commands.sh",
    }
    atomic_publish_bundle(
        {
            paths["direction"]: direction_raw,
            paths["shot_list"]: render_shot_list(direction).encode("utf-8"),
            paths["storyboard_manifest"]: json_bytes(manifest),
            paths["commands"]: render_commands(manifest, repo_root=ROOT).encode("utf-8"),
        }
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="normalize a JSON brief without media or network execution")
    plan.add_argument("--brief", required=True)
    plan.add_argument("--project-dir", "--out", dest="project_dir")
    args = parser.parse_args(argv)
    try:
        paths = plan_project(args.brief, args.project_dir)
    except VideoDirectionError as exc:
        parser.error(str(exc))
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
