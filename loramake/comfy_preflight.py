#!/usr/bin/env python3
"""Loader-asset preflight: reject a run before any GPU work when a name is absent on the box.

Extracted from ``loramake.comfy`` (C3). This module owns the decision and the exact operator-
facing rejection messages; it owns no transport. The three things the decision needs from the
driver — the batch enumeration, the live single re-probe, and the "this answer came from the
disk cache" predicate — arrive as callables, so ``loramake.comfy`` binds them per call against
its own globals and a ``comfy.object_info_enum`` monkeypatch still reaches the probe.

``loramake.comfy.preflight_assets`` keeps the frozen public signature and wraps
``preflight_assets`` here; ``missing_assets``, ``suggest_assets`` and ``_suggest_suffix`` are
re-exported as the same objects.
"""
from __future__ import annotations

import difflib
import sys
from collections.abc import Callable, Sequence

EnumMany = Callable[[Sequence[tuple[str, str]], str], dict[tuple[str, str], set[str]]]
EnumLive = Callable[[str, str, str], set[str]]
AnsweredFromDisk = Callable[[tuple[str, str, str]], bool]


def missing_assets(known: set[str], needed: list[str]) -> list[str]:
    """Names in `needed` absent from `known`. An empty `known` (probe failed) returns [] — skip the guard."""
    if not known:
        return []
    return [n for n in needed if n not in known]


def suggest_assets(name: str, known: set[str], limit: int = 3) -> list[str]:
    """Closest known loader names for an absent one, so a typo'd or subfolder-less name fails
    with actionable candidates. Basename containment beats fuzzy matching because loader
    names carry Windows subfolder prefixes the caller often omits."""
    def basename(n: str) -> str:
        return n.lower().replace("\\", "/").rsplit("/", 1)[-1]

    query = basename(name)
    stem = query.rsplit(".", 1)[0]
    contained = sorted(k for k in known
                       if stem and (stem in basename(k) or basename(k).rsplit(".", 1)[0] in query))
    if contained:
        return contained[:limit]
    # Fuzzy over extension-less stems: the shared ".safetensors" suffix would otherwise pull
    # unrelated names over the cutoff.
    by_stem = {basename(k).rsplit(".", 1)[0]: k for k in known}
    return [by_stem[c] for c in difflib.get_close_matches(stem, list(by_stem), n=limit, cutoff=0.6)]


def _suggest_suffix(names: list[str], known: set[str]) -> str:
    """`; closest: ...` appendix for a preflight rejection (empty when nothing is close)."""
    candidates: list[str] = []
    for name in names:
        for hit in suggest_assets(name, known):
            if hit not in candidates:
                candidates.append(hit)
    return f"; closest: {', '.join(candidates[:3])}" if candidates else ""


def preflight_assets(arch: str, ckpt: str, lora_names: list[str], *,
                     controlnets: "Sequence[str]" = (), upscalers: "Sequence[str]" = (),
                     clips: "Sequence[str]" = (), vaes: "Sequence[str]" = (),
                     model_patches: "Sequence[str]" = (),
                     lora_enums: set[str] | None = None,
                     box: str,
                     enum_many: EnumMany,
                     enum_live: EnumLive,
                     answered_from_disk: AnsweredFromDisk) -> None:
    """The preflight decision. See ``loramake.comfy.preflight_assets`` for the public contract.

    ``box`` is the already-normalized target; ``enum_many(probes, box)`` batches the
    ``/object_info`` enumerations, ``enum_live(node, field, box)`` performs the single live
    re-probe that must bypass every cache, and ``answered_from_disk(key)`` reports whether the
    batched answer for ``(box, node, field)`` came from the disk cache.
    """
    ckpt_node, ckpt_field = (("CheckpointLoaderSimple", "ckpt_name") if arch == "sdxl"
                             else ("UNETLoader", "unet_name"))
    probes: list[tuple[str, str]] = [(ckpt_node, ckpt_field)]
    if lora_enums is None:
        probes.append(("LoraLoader", "lora_name"))
    if controlnets:
        probes.append(("ControlNetLoader", "control_net_name"))
    if upscalers:
        probes.append(("UpscaleModelLoader", "model_name"))
    if clips:
        probes.append(("CLIPLoader", "clip_name"))
    if vaes:
        probes.append(("VAELoader", "vae_name"))
    if model_patches:
        probes.append(("ModelPatchLoader", "name"))
    enums = enum_many(probes, box)

    def _known_and_missing(node: str, field: str, needed: list[str],
                           known: set[str] | None = None) -> tuple[set[str], list[str]]:
        if known is None:
            known = enums.get((node, field), set())
        miss = missing_assets(known, needed)
        if miss and answered_from_disk((box, node, field)):
            live = enum_live(node, field, box)
            if live:
                known = live
                miss = missing_assets(known, needed)
        return known, miss

    known_loras_in = set(lora_enums) if lora_enums is not None else None
    known_ckpts, missing_ckpt = _known_and_missing(ckpt_node, ckpt_field, [ckpt])
    known_loras, missing_loras = _known_and_missing("LoraLoader", "lora_name", lora_names,
                                                    known=known_loras_in)
    if missing_ckpt:
        sys.exit(f"ERROR: checkpoint not found on this box ({box}): {missing_ckpt[0]!r} "
                 f"(checked {ckpt_node}.{ckpt_field}; {len(known_ckpts)} available"
                 f"{_suggest_suffix(missing_ckpt, known_ckpts)})")
    if missing_loras:
        sys.exit(f"ERROR: LoRA(s) not found on this box ({box}): {missing_loras} "
                 f"(checked LoraLoader.lora_name; {len(known_loras)} available"
                 f"{_suggest_suffix(missing_loras, known_loras)})")
    if controlnets:
        known, miss = _known_and_missing("ControlNetLoader", "control_net_name", list(controlnets))
        if miss:
            sys.exit(f"ERROR: ControlNet model(s) not found on this box ({box}): {miss} "
                     f"(checked ControlNetLoader.control_net_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
    if upscalers:
        known, miss = _known_and_missing("UpscaleModelLoader", "model_name", list(upscalers))
        if miss:
            sys.exit(f"ERROR: upscale model(s) not found on this box ({box}): {miss} "
                     f"(checked UpscaleModelLoader.model_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)}) — "
                     f"set BOXA_UPSCALER to one of the available names")
    if clips:
        known, miss = _known_and_missing("CLIPLoader", "clip_name", list(clips))
        if miss:
            sys.exit(f"ERROR: CLIP model(s) not found on this box ({box}): {miss} "
                     f"(checked CLIPLoader.clip_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
    if vaes:
        known, miss = _known_and_missing("VAELoader", "vae_name", list(vaes))
        if miss:
            sys.exit(f"ERROR: VAE model(s) not found on this box ({box}): {miss} "
                     f"(checked VAELoader.vae_name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)})")
    if model_patches:
        known, miss = _known_and_missing("ModelPatchLoader", "name", list(model_patches))
        if miss:
            sys.exit(f"ERROR: model patch(es) not found on this box ({box}): {miss} "
                     f"(checked ModelPatchLoader.name; {len(known)} available"
                     f"{_suggest_suffix(miss, known)}) — LLLite weights live in models/model_patches/ "
                     f"on this box; an empty enumeration means the directory itself is missing")
