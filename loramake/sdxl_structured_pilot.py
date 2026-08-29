"""Strict contracts for the SDXL-family structured Pilot (iLustMix on ZZZ).

This is the SDXL sibling of :mod:`loramake.anima_structured_pilot`: the same
text-blind compiler lane (an LLM may choose only registered IDs and layout
enums; the exact user source text is hashed without normalization) bound to an
SDXL registry.  All validation logic is shared with the Anima module — only the
registry path, schema ids, and feature flag differ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loramake.anima_structured_pilot import (
    ContractError,
    canonical_json,
    sha256_file,
    sha256_text,
    task_digest,
)
from loramake.anima_structured_pilot import compile_task as _compile_task
from loramake.anima_structured_pilot import effective_conditioning as _effective_conditioning
from loramake.anima_structured_pilot import effective_negative_text as _effective_negative_text
from loramake.anima_structured_pilot import load_registry as _load_registry
from loramake.anima_structured_pilot import pilot_enabled as _pilot_enabled
from loramake.anima_structured_pilot import require_task as _require_task
from loramake.anima_structured_pilot import resolve_runtime as _resolve_runtime
from loramake.anima_structured_pilot import validate_selection as _validate_selection
from loramake.anima_structured_pilot import validate_task as _validate_task
from loramake.config import ROOT

__all__ = [
    "ContractError",
    "FEATURE_ENV",
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA",
    "SELECTION_SCHEMA",
    "TASK_SCHEMA",
    "canonical_json",
    "compile_task",
    "effective_conditioning",
    "effective_negative_text",
    "load_registry",
    "pilot_enabled",
    "require_task",
    "resolve_runtime",
    "sha256_file",
    "sha256_text",
    "task_digest",
    "validate_selection",
    "validate_task",
]

REGISTRY_PATH = ROOT / "scripts" / "sdxl_structured_pilot.json"
REGISTRY_SCHEMA = "loramake.sdxl-structured-registry.v1"
SELECTION_SCHEMA = "loramake.sdxl-structured-selection.v1"
TASK_SCHEMA = "loramake.sdxl-structured-task.v1"
FEATURE_ENV = "LORAMAKE_SDXL_STRUCTURED_PILOT_ENABLED"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return _load_registry(path, schema=REGISTRY_SCHEMA)


def pilot_enabled(env: Mapping[str, str] | None = None, registry: dict[str, Any] | None = None) -> bool:
    return _pilot_enabled(env, registry or load_registry(), env_var=FEATURE_ENV)


def validate_selection(selection: object, *, registry: dict[str, Any] | None = None) -> list[str]:
    return _validate_selection(selection, registry=registry or load_registry())


def validate_task(task: object, *, registry: dict[str, Any] | None = None,
                  verify_local_assets: bool = True) -> list[str]:
    return _validate_task(task, registry=registry or load_registry(),
                          verify_local_assets=verify_local_assets)


def require_task(task: object, *, registry: dict[str, Any] | None = None,
                 verify_local_assets: bool = True) -> dict[str, Any]:
    return _require_task(task, registry=registry or load_registry(),
                         verify_local_assets=verify_local_assets)


def compile_task(source_text: object, selection: object, seed: object, *,
                 registry: dict[str, Any] | None = None) -> dict[str, Any]:
    return _compile_task(source_text, selection, seed, registry=registry or load_registry())


def effective_conditioning(task: Mapping[str, Any], *,
                           registry: dict[str, Any] | None = None) -> dict[str, Any]:
    return _effective_conditioning(task, registry=registry or load_registry())


def effective_negative_text(task: Mapping[str, Any], *,
                            registry: dict[str, Any] | None = None) -> str:
    return _effective_negative_text(task, registry=registry or load_registry())


def resolve_runtime(task: Mapping[str, Any], *,
                    registry: dict[str, Any] | None = None) -> dict[str, Any]:
    return _resolve_runtime(task, registry=registry or load_registry())
