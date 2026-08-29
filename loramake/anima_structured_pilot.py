"""Strict contracts for the isolated Anima identity + pose Pilot.

The compiler lane is deliberately text-blind: an LLM may choose only registered
IDs and layout enums.  The exact user source text travels beside that selection,
is hashed without normalization, and is deterministically joined with the
server-owned safety token and identity trigger.  No component in this module
rewrites, translates, expands, or repairs prompt text.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from loramake.config import ROOT

REGISTRY_PATH = ROOT / "scripts" / "anima_structured_pilot.json"
REGISTRY_SCHEMA = "loramake.anima-structured-registry.v1"
SELECTION_SCHEMA = "loramake.anima-structured-selection.v1"
TASK_SCHEMA = "loramake.anima-structured-task.v1"
FEATURE_ENV = "LORAMAKE_ANIMA_STRUCTURED_PILOT_ENABLED"
MAX_SOURCE_CHARS = 4000
MAX_SEED = (1 << 63) - 1


class ContractError(ValueError):
    """The caller supplied data outside the frozen Pilot contract."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_digest(task: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(task))


def _schema_variant(registry_schema: str, kind: str) -> str:
    """Derive the selection/task schema id from a family's registry schema id."""
    return registry_schema.replace("-registry.", f"-{kind}.")


def load_registry(path: Path = REGISTRY_PATH, *, schema: str = REGISTRY_SCHEMA) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError([f"Pilot registry not found: {path}"]) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError([f"Pilot registry is unreadable: {type(exc).__name__}: {exc}"]) from exc
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise ContractError([f"Pilot registry schema must be {schema}"])
    if not isinstance(data.get("profile"), dict):
        raise ContractError(["Pilot registry profile must be an object"])
    if not isinstance(data.get("identities"), dict) or not data["identities"]:
        raise ContractError(["Pilot registry identities must be a non-empty object"])
    if not isinstance(data.get("poses"), dict) or not data["poses"]:
        raise ContractError(["Pilot registry poses must be a non-empty object"])
    if "directions" in data and (not isinstance(data["directions"], dict) or not data["directions"]):
        raise ContractError(["Pilot registry directions, when present, must be a non-empty object"])
    return data


def pilot_enabled(env: Mapping[str, str] | None = None, registry: dict[str, Any] | None = None,
                  *, env_var: str = FEATURE_ENV) -> bool:
    values = os.environ if env is None else env
    raw = values.get(env_var)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    data = registry or load_registry()
    return bool(data.get("enabled_by_default", False))


def _strict_keys(value: object, expected: set[str], label: str, problems: list[str],
                 optional: frozenset[str] = frozenset()) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        problems.append(f"{label} must be an object")
        return None
    actual = set(value)
    for key in sorted(expected - actual):
        problems.append(f"{label}.{key} is required")
    for key in sorted(actual - expected - optional):
        problems.append(f"{label}.{key} is not allowed")
    return value


def _valid_seed(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_SEED


def validate_selection(selection: object, *, registry: dict[str, Any] | None = None) -> list[str]:
    data = registry or load_registry()
    problems: list[str] = []
    item = _strict_keys(selection, {"schema", "identity_id", "pose_id", "layout"}, "selection", problems,
                        optional=frozenset({"direction_id"}))
    if item is None:
        return problems
    expected_schema = _schema_variant(data["schema"], "selection")
    if item.get("schema") != expected_schema:
        problems.append(f"selection.schema must be {expected_schema}")
    identity_id = item.get("identity_id")
    if identity_id not in data["identities"]:
        problems.append("selection.identity_id is not registered for this Pilot")
    pose_id = item.get("pose_id")
    if pose_id not in data["poses"]:
        problems.append("selection.pose_id is not registered for this Pilot")
    if "direction_id" in item and item["direction_id"] not in data.get("directions", {}):
        problems.append("selection.direction_id is not registered for this Pilot")
    layout = _strict_keys(item.get("layout"), {"canvas", "framing", "placement"}, "selection.layout", problems)
    if layout is not None:
        for key in ("canvas", "framing", "placement"):
            if layout.get(key) not in data["layout"][key]:
                problems.append(f"selection.layout.{key} is not an allowed value")
    return problems


def _source_problems(source_text: object) -> list[str]:
    if not isinstance(source_text, str):
        return ["source_text must be a string"]
    problems: list[str] = []
    if not source_text.strip():
        problems.append("source_text must contain non-whitespace text")
    if len(source_text) > MAX_SOURCE_CHARS:
        problems.append(f"source_text must be at most {MAX_SOURCE_CHARS} characters")
    if "\x00" in source_text:
        problems.append("source_text must not contain NUL")
    return problems


def compile_task(source_text: object, selection: object, seed: object, *,
                 registry: dict[str, Any] | None = None) -> dict[str, Any]:
    data = registry or load_registry()
    problems = _source_problems(source_text)
    problems.extend(validate_selection(selection, registry=data))
    if not _valid_seed(seed):
        problems.append(f"seed must be an integer in 0..{MAX_SEED}")
    if problems:
        raise ContractError(problems)
    assert isinstance(source_text, str) and isinstance(selection, dict) and isinstance(seed, int)
    identity_id = selection["identity_id"]
    pose_id = selection["pose_id"]
    identity = data["identities"][identity_id]
    pose = data["poses"][pose_id]
    task = {
        "schema": _schema_variant(data["schema"], "task"),
        "pilot": True,
        "source": {"text": source_text, "sha256": sha256_text(source_text)},
        "profile": data["profile"]["id"],
        "identity": {"lora_id": identity_id, "strength": identity["strength"]},
        "pose": {
            "asset_id": pose_id,
            "format": pose["format"],
            "strength": pose["strength"],
            "start_percent": pose["start_percent"],
            "end_percent": pose["end_percent"],
        },
        "layout": dict(selection["layout"]),
        "seed": seed,
        "sfw": True,
    }
    if "direction_id" in selection:
        task["direction"] = {"id": selection["direction_id"]}
    problems = validate_task(task, registry=data, verify_local_assets=True)
    if problems:
        raise ContractError(problems)
    return task


def _same_number(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return float(actual) == float(expected)


def _resolve_local_asset(raw_path: object, label: str, problems: list[str]) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        problems.append(f"{label}.path must be a repository-relative string")
        return None
    candidate = (ROOT / raw_path).resolve(strict=False)
    if not candidate.is_relative_to(ROOT.resolve()):
        problems.append(f"{label}.path escapes the repository")
        return None
    return candidate


def _verify_asset(entry: dict[str, Any], label: str, problems: list[str]) -> None:
    path = _resolve_local_asset(entry.get("path"), label, problems)
    if path is None:
        return
    try:
        stat = path.stat()
    except OSError:
        problems.append(f"{label} file is missing: {entry.get('path')}")
        return
    if not path.is_file():
        problems.append(f"{label} is not a regular file: {entry.get('path')}")
        return
    if stat.st_size != entry.get("bytes"):
        problems.append(f"{label} byte size does not match the registry")
        return
    if sha256_file(path) != entry.get("sha256"):
        problems.append(f"{label} sha256 does not match the registry")


def validate_task(task: object, *, registry: dict[str, Any] | None = None,
                  verify_local_assets: bool = True) -> list[str]:
    data = registry or load_registry()
    problems: list[str] = []
    item = _strict_keys(
        task,
        {"schema", "pilot", "source", "profile", "identity", "pose", "layout", "seed", "sfw"},
        "task",
        problems,
        optional=frozenset({"direction"}),
    )
    if item is None:
        return problems
    expected_schema = _schema_variant(data["schema"], "task")
    if item.get("schema") != expected_schema:
        problems.append(f"task.schema must be {expected_schema}")
    if item.get("pilot") is not True:
        problems.append("task.pilot must be true")
    if item.get("profile") != data["profile"]["id"]:
        problems.append("task.profile is not the frozen Pilot profile")
    if item.get("sfw") is not True:
        problems.append("task.sfw must be true")
    if not _valid_seed(item.get("seed")):
        problems.append(f"task.seed must be an integer in 0..{MAX_SEED}")

    source = _strict_keys(item.get("source"), {"text", "sha256"}, "task.source", problems)
    if source is not None:
        problems.extend(_source_problems(source.get("text")))
        text = source.get("text")
        if isinstance(text, str) and source.get("sha256") != sha256_text(text):
            problems.append("task.source.sha256 does not match the exact UTF-8 source text")

    identity = _strict_keys(item.get("identity"), {"lora_id", "strength"}, "task.identity", problems)
    identity_entry = None
    if identity is not None:
        identity_entry = data["identities"].get(identity.get("lora_id"))
        if identity_entry is None:
            problems.append("task.identity.lora_id is not registered for this Pilot")
        elif not _same_number(identity.get("strength"), identity_entry["strength"]):
            problems.append("task.identity.strength must equal the registry value")

    pose = _strict_keys(
        item.get("pose"),
        {"asset_id", "format", "strength", "start_percent", "end_percent"},
        "task.pose",
        problems,
    )
    pose_entry = None
    if pose is not None:
        pose_entry = data["poses"].get(pose.get("asset_id"))
        if pose_entry is None:
            problems.append("task.pose.asset_id is not registered for this Pilot")
        else:
            for key in ("format", "strength", "start_percent", "end_percent"):
                expected = pose_entry[key]
                actual = pose.get(key)
                matches = actual == expected if isinstance(expected, str) else _same_number(actual, expected)
                if not matches:
                    problems.append(f"task.pose.{key} must equal the registry value")

    layout = _strict_keys(item.get("layout"), {"canvas", "framing", "placement"}, "task.layout", problems)
    if layout is not None:
        for key in ("canvas", "framing", "placement"):
            if layout.get(key) not in data["layout"][key]:
                problems.append(f"task.layout.{key} is not an allowed value")

    if "direction" in item:
        direction = _strict_keys(item["direction"], {"id"}, "task.direction", problems)
        if direction is not None and direction.get("id") not in data.get("directions", {}):
            problems.append("task.direction.id is not registered for this Pilot")

    if verify_local_assets:
        if identity_entry is not None:
            _verify_asset(identity_entry["ccip_anchor"], "identity.ccip_anchor", problems)
        if pose_entry is not None:
            _verify_asset(pose_entry, "pose", problems)
    return problems


def require_task(task: object, *, registry: dict[str, Any] | None = None,
                 verify_local_assets: bool = True) -> dict[str, Any]:
    problems = validate_task(task, registry=registry, verify_local_assets=verify_local_assets)
    if problems:
        raise ContractError(problems)
    assert isinstance(task, dict)
    return task


def effective_conditioning(task: Mapping[str, Any], *, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    data = registry or load_registry()
    require_task(task, registry=data, verify_local_assets=False)
    profile = data["profile"]
    identity = data["identities"][task["identity"]["lora_id"]]
    prefix = profile["positive_prefix"] if "positive_prefix" in profile else profile["safety_token"]
    components = [prefix, identity["trigger"]]
    if identity.get("identity_tags"):
        # Registry-owned appearance lock: on SDXL-family bases the trigger alone does not
        # reliably pin hair/eye colors against a strong style or scaffold context.
        components.append(identity["identity_tags"])
    if "direction" in task:
        components.append(data["directions"][task["direction"]["id"]]["positive_text"])
    components.append(task["source"]["text"])
    text = ", ".join(components)
    return {
        "joiner": ", ",
        "components": components,
        "text": text,
        "sha256": sha256_text(text),
        "source_is_exact_suffix": text.endswith(task["source"]["text"]),
    }


def effective_negative_text(task: Mapping[str, Any], *, registry: dict[str, Any] | None = None) -> str:
    data = registry or load_registry()
    require_task(task, registry=data, verify_local_assets=False)
    negative = data["profile"]["negative_text"]
    if "direction" in task:
        addition = data["directions"][task["direction"]["id"]].get("negative_text_add", "")
        if addition:
            negative = f"{negative}, {addition}"
    return negative


def resolve_runtime(task: Mapping[str, Any], *, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    data = registry or load_registry()
    require_task(task, registry=data, verify_local_assets=True)
    identity = data["identities"][task["identity"]["lora_id"]]
    pose = data["poses"][task["pose"]["asset_id"]]
    runtime = {
        "profile": data["profile"],
        "identity": identity,
        "pose": {**pose, "absolute_path": str((ROOT / pose["path"]).resolve())},
        "ccip_anchor_path": str((ROOT / identity["ccip_anchor"]["path"]).resolve()),
        "conditioning": effective_conditioning(task, registry=data),
        "negative_text": effective_negative_text(task, registry=data),
        "task_sha256": task_digest(task),
    }
    if "direction" in task:
        runtime["direction"] = data["directions"][task["direction"]["id"]]
    return runtime


def public_metadata(*, registry: dict[str, Any] | None = None,
                    env: Mapping[str, str] | None = None) -> dict[str, Any]:
    data = registry or load_registry()
    return {
        "schema": REGISTRY_SCHEMA,
        "enabled": pilot_enabled(env=env, registry=data),
        "feature_env": FEATURE_ENV,
        "selection_schema": SELECTION_SCHEMA,
        "task_schema": TASK_SCHEMA,
        "profile": {
            "id": data["profile"]["id"],
            "box": data["profile"]["box"],
            "width": data["profile"]["width"],
            "height": data["profile"]["height"],
            "base_sha256": data["profile"]["base"]["sha256"],
            "lllite_sha256": data["profile"]["lllite"]["sha256"],
        },
        "identities": [
            {"id": key, "strength": entry["strength"], "trigger": entry["trigger"]}
            for key, entry in data["identities"].items()
        ],
        "poses": [
            {
                "id": key,
                "description": entry["description"],
                "width": entry["width"],
                "height": entry["height"],
                "format": entry["format"],
                "source_url": entry["source_url"],
                "source_license": entry["source_license"],
            }
            for key, entry in data["poses"].items()
        ],
        "layout": data["layout"],
    }
