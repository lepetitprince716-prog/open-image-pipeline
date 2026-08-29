from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import video_director  # noqa: E402
import video_review  # noqa: E402
from loramake.video_direction import (  # noqa: E402
    DIRECTION_SCHEMA,
    REVIEW_DIMENSIONS,
    REVIEW_SCHEMA,
    build_review_record,
    normalize_direction,
)


def _brief() -> dict:
    return {
        "project": {
            "id": "moonlit-pv",
            "title": "Moonlit mission",
            "description": "A compact task-introduction PV.",
            "driver": "codex",
            "asset_namespace": "original_character",
            "resolution": "1080x1920",
            "fps": 30,
            "duration_seconds": 4,
        },
        "scenes": [
            {
                "id": "arrival",
                "title": "Arrival",
                "shots": [
                    {
                        "id": "s01",
                        "title": "Moon silhouette",
                        "start": 0,
                        "end": 1.5,
                        "shot_size": "extreme_wide",
                        "visual": "A lone silhouette against a large moon.",
                        "implementation": "still_motion",
                        "start_state": "The roof is empty and still.",
                        "end_state": "The silhouette is readable.",
                        "action": "Wind moves the coat hem.",
                        "camera": "Slow push-in.",
                        "screen_direction": "center",
                        "continuity_locks": ["moon position", "single red accent"],
                        "generation": {
                            "seed_image": "assets/s01.png",
                            "identity_ref": None,
                            "prompt": "",
                            "still_motion": {"preset": "slow_push_in", "strength": 0.045},
                        },
                        "sound_cues": [
                            {"at": 0, "kind": "ambience", "description": "night wind"}
                        ],
                        "acceptance": ["silhouette reads at phone size"],
                    },
                    {
                        "id": "s02",
                        "title": "Eye close-up",
                        "start": 1.5,
                        "end": 4,
                        "shot_size": "extreme_close_up",
                        "visual": "One eye catches a restrained red reflection.",
                        "implementation": "wan_i2v",
                        "start_state": "Eye open, gaze steady.",
                        "end_state": "Blink completes and gaze returns.",
                        "action": "One slow blink.",
                        "camera": "Subtle push-in.",
                        "screen_direction": "camera",
                        "continuity_locks": ["eye side", "red accent"],
                        "generation": {
                            "seed_image": "assets/s02.png",
                            "identity_ref": "assets/identity.png",
                            "seed": 123,
                            "prompt": "one slow blink, otherwise still",
                            "res": "480x832",
                            "length": 41,
                            "pos_suffix": "flat 2D anime, clean lineart, subtle motion",
                        },
                        "sound_cues": [],
                        "acceptance": ["exactly one blink", "identity remains stable"],
                    },
                ],
            }
        ],
    }


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_normalize_direction_builds_stable_project_scene_shot_contract(tmp_path: Path) -> None:
    direction = normalize_direction(_brief(), base_dir=tmp_path)

    assert direction["schema"] == DIRECTION_SCHEMA
    assert direction["project"]["duration_frames"] == 120
    assert direction["project"]["aspect_ratio"] == "9:16"
    assert [scene["id"] for scene in direction["scenes"]] == ["arrival"]
    first, second = direction["scenes"][0]["shots"]
    assert first["timeline"] == {
        "start_frame": 0,
        "end_frame": 45,
        "duration_frames": 45,
        "start_timecode": "00:00:00.000",
        "end_timecode": "00:00:01.500",
    }
    assert second["generation"]["seed"] == 123
    assert second["generation"]["seed_image"] == str((tmp_path / "assets/s02.png").resolve())
    assert set(second) >= {
        "start_state",
        "end_state",
        "action",
        "camera",
        "continuity_locks",
        "generation",
        "sound_cues",
        "acceptance",
    }
    assert normalize_direction(_brief(), base_dir=tmp_path) == direction


def test_director_plan_writes_offline_exact_commands_and_is_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = _write_json(tmp_path / "brief.json", _brief())
    project_dir = tmp_path / "project"

    # The planning entry point has no execution collaborator to patch: attempting to
    # import subprocess or a socket during this call is itself a regression.
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"subprocess", "socket", "httpx", "requests"}:
            raise AssertionError(f"offline planner imported execution/network module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert video_director.main(
        ["plan", "--brief", str(brief_path), "--project-dir", str(project_dir)]
    ) == 0

    assert {path.name for path in project_dir.iterdir()} == {
        "commands.sh",
        "direction.json",
        "shot_list.md",
        "storyboard_manifest.json",
    }
    direction_raw = (project_dir / "direction.json").read_bytes()
    direction = json.loads(direction_raw)
    manifest = json.loads((project_dir / "storyboard_manifest.json").read_text(encoding="utf-8"))
    wan = next(row for row in manifest["shots"] if row["shot_id"] == "s02")
    still = next(row for row in manifest["shots"] if row["shot_id"] == "s01")
    assert manifest["direction"]["sha256"] == hashlib.sha256(direction_raw).hexdigest()
    assert wan["env"] == {"DRIVING_MODEL": "codex"}
    assert wan["argv"][:2] == ["python3", str((ROOT / "scripts/z3_video.py").resolve())]
    assert wan["argv"][wan["argv"].index("--seed") + 1] == "123"
    assert wan["argv"][wan["argv"].index("--res") + 1] == "480x832"
    assert wan["argv"][wan["argv"].index("--pos-suffix") + 1] == direction["scenes"][0]["shots"][1][
        "generation"
    ]["pos_suffix"]
    assert wan["asset_state"] == "planned"
    assert wan["command_state"] == "needs_source_asset"
    assert "zoompan=" in still["argv"][still["argv"].index("-vf") + 1]
    assert still["argv"][still["argv"].index("-frames:v") + 1] == "45"
    commands = (project_dir / "commands.sh").read_text(encoding="utf-8")
    assert "# blocked exact command:" in commands
    assert "DRIVING_MODEL=codex" in commands
    assert "--neg" in commands and "--out" in commands
    assert "|" not in commands  # commands are literal argv, not shell pipelines

    with pytest.raises(FileExistsError):
        video_director.main(
            ["plan", "--brief", str(brief_path), "--project-dir", str(project_dir)]
        )
    assert (project_dir / "direction.json").read_bytes() == direction_raw


def test_direction_schema_is_strict_and_rejects_contract_drift(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (SCRIPTS / "schema/video_direction.schema.json").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    direction = normalize_direction(_brief(), base_dir=tmp_path)

    validator.validate(direction)
    drifted = json.loads(json.dumps(direction))
    drifted["scenes"][0]["shots"][0]["aesthetic_score"] = 9.5
    assert list(validator.iter_errors(drifted))
    missing = json.loads(json.dumps(direction))
    del missing["scenes"][0]["shots"][0]["acceptance"]
    assert list(validator.iter_errors(missing))


@pytest.mark.parametrize(
    ("motion", "filter_marker"),
    [
        ({"preset": "slow_push_in", "strength": 0.045}, "zoompan="),
        ({"preset": "impact_jolt", "strength": 0.012}, "if(eq(n,1)"),
        (
            {
                "preset": "status_reveal",
                "strength": 0.85,
                "region": {"x": 0.7, "y": 0.3, "width": 0.02, "height": 0.4},
                "color": "#8F1028",
            },
            "drawbox=",
        ),
    ],
)
def test_still_motion_presets_compile_declared_motion_not_a_frozen_loop(
    tmp_path: Path, motion: dict, filter_marker: str
) -> None:
    brief = _brief()
    brief["scenes"][0]["shots"] = [brief["scenes"][0]["shots"][0]]
    brief["scenes"][0]["shots"][0]["end"] = 4
    brief["scenes"][0]["shots"][0]["generation"]["still_motion"] = motion
    direction = normalize_direction(brief, base_dir=tmp_path)
    direction_path = tmp_path / "direction.json"
    direction_raw = json.dumps(direction, ensure_ascii=False, indent=2).encode()
    direction_path.write_bytes(direction_raw)
    from loramake.video_direction import build_storyboard_manifest

    manifest = build_storyboard_manifest(
        direction,
        direction_path=direction_path,
        direction_sha256=hashlib.sha256(direction_raw).hexdigest(),
        repo_root=ROOT,
        output_root=tmp_path,
    )
    argv = manifest["shots"][0]["argv"]
    assert filter_marker in argv[argv.index("-vf") + 1]
    assert argv[argv.index("-frames:v") + 1] == "120"


def test_still_motion_requires_an_explicit_motion_contract(tmp_path: Path) -> None:
    brief = _brief()
    del brief["scenes"][0]["shots"][0]["generation"]["still_motion"]
    with pytest.raises(ValueError, match="still_motion"):
        normalize_direction(brief, base_dir=tmp_path)


def _review_args(
    direction_path: Path, project_dir: Path, record_id: str, clip_path: Path
) -> list[str]:
    args = [
        "record",
        "--direction",
        str(direction_path),
        "--project-dir",
        str(project_dir),
        "--shot-id",
        "s02",
        "--timecode",
        "00:00:02.000",
        "--clip",
        str(clip_path),
        "--record-id",
        record_id,
        "--reviewer",
        "director",
        "--summary",
        "Blink lands, but the final frame needs another look.",
        "--dimension-note",
        "motion=one complete blink",
    ]
    values = {
        "motion": "pass",
        "camera": "pass",
        "beat": "pass",
        "identity": "partial",
        "continuity": "pass",
        "flicker": "uncertain",
        "line_boiling": "pass",
        "palette": "pass",
    }
    for dimension in REVIEW_DIMENSIONS:
        args.extend([f"--{dimension.replace('_', '-')}", values[dimension]])
    return args


def test_review_record_is_frame_addressed_atomic_append_only_and_has_no_composite_score(
    tmp_path: Path,
) -> None:
    brief_path = _write_json(tmp_path / "brief.json", _brief())
    project_dir = tmp_path / "project"
    video_director.main(["plan", "--brief", str(brief_path), "--project-dir", str(project_dir)])
    direction_path = project_dir / "direction.json"
    clip_path = tmp_path / "reviewed-s02.mp4"
    clip_path.write_bytes(b"reviewed clip fixture")

    assert video_review.main(_review_args(direction_path, project_dir, "review-a", clip_path)) == 0
    first_path = project_dir / "reviews/s02/000060-review-a.json"
    first_raw = first_path.read_bytes()
    record = json.loads(first_raw)
    assert record["schema"] == REVIEW_SCHEMA
    assert record["project_id"] == "moonlit-pv"
    assert record["scene_id"] == "arrival"
    assert record["shot_id"] == "s02"
    assert record["timecode"] == {
        "value": "00:00:02.000",
        "frame": 60,
        "relative_frame": 15,
    }
    assert record["direction"]["sha256"] == hashlib.sha256(direction_path.read_bytes()).hexdigest()
    assert record["clip"] == {
        "path": str(clip_path.resolve()),
        "sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        "bytes": clip_path.stat().st_size,
    }
    assert tuple(record["verdicts"]) == REVIEW_DIMENSIONS
    assert record["verdicts"]["motion"] == {"verdict": "pass", "note": "one complete blink"}
    assert "score" not in json.dumps(record).lower()

    with pytest.raises(FileExistsError):
        video_review.main(_review_args(direction_path, project_dir, "review-a", clip_path))
    assert first_path.read_bytes() == first_raw

    assert video_review.main(_review_args(direction_path, project_dir, "review-b", clip_path)) == 0
    assert (project_dir / "reviews/s02/000060-review-b.json").is_file()
    assert first_path.read_bytes() == first_raw


def test_review_schema_requires_each_independent_verdict_and_rejects_score(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    direction_path = _write_json(tmp_path / "direction.json", normalize_direction(_brief(), base_dir=tmp_path))
    direction = json.loads(direction_path.read_text(encoding="utf-8"))
    clip_path = tmp_path / "reviewed.mp4"
    clip_path.write_bytes(b"reviewed clip")
    record = build_review_record(
        direction,
        direction_path=direction_path,
        shot_id="s02",
        timecode="00:00:02.000",
        clip_path=clip_path,
        verdicts={dimension: "pass" for dimension in REVIEW_DIMENSIONS},
        reviewer="director",
        record_id="schema-fixture",
        created_at="2026-07-16T12:00:00+00:00",
    )
    schema = json.loads((SCRIPTS / "schema/video_review.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(record)

    missing = json.loads(json.dumps(record))
    del missing["verdicts"]["palette"]
    assert list(validator.iter_errors(missing))
    scored = json.loads(json.dumps(record))
    scored["aesthetic_score"] = 8
    assert list(validator.iter_errors(scored))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda brief: brief["scenes"][0]["shots"][1].update(start=1), "overlaps"),
        (lambda brief: brief["scenes"][0]["shots"][1].update(id="s01"), "duplicate shot"),
        (lambda brief: brief["scenes"][0]["shots"][0].update(end=1.51), "exact frame"),
        (lambda brief: brief["project"].update(aspect_ratio="16:9"), "must match resolution"),
    ],
)
def test_direction_rejects_ambiguous_timeline_and_identity_drift(mutate, message, tmp_path: Path) -> None:
    brief = _brief()
    mutate(brief)
    with pytest.raises(ValueError, match=message):
        normalize_direction(brief, base_dir=tmp_path)


def test_review_rejects_unknown_shot_out_of_range_time_and_incomplete_dimensions(tmp_path: Path) -> None:
    direction_path = _write_json(tmp_path / "direction.json", normalize_direction(_brief(), base_dir=tmp_path))
    direction = json.loads(direction_path.read_text(encoding="utf-8"))
    clip_path = tmp_path / "reviewed.mp4"
    clip_path.write_bytes(b"reviewed clip")
    verdicts = {dimension: "pass" for dimension in REVIEW_DIMENSIONS}

    with pytest.raises(ValueError, match="unknown shot"):
        build_review_record(
            direction,
            direction_path=direction_path,
            shot_id="missing",
            timecode="00:00:02.000",
            clip_path=clip_path,
            verdicts=verdicts,
        )
    with pytest.raises(ValueError, match="outside shot"):
        build_review_record(
            direction,
            direction_path=direction_path,
            shot_id="s02",
            timecode="00:00:01.000",
            clip_path=clip_path,
            verdicts=verdicts,
        )
    incomplete = dict(verdicts)
    del incomplete["palette"]
    with pytest.raises(ValueError, match="exactly the review dimensions"):
        build_review_record(
            direction,
            direction_path=direction_path,
            shot_id="s02",
            timecode="00:00:02.000",
            clip_path=clip_path,
            verdicts=incomplete,
        )
