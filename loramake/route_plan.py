"""Route-plan seam of the gen.py router: resource compilation and machine-readable plans.

Extracted from ``scripts/gen.py`` (Phase C item C1) as a leaf: this module imports the
standard library and ``loramake.model_resources`` only, and it never locates anything
through ``__file__``. Everything the router owns arrives as a parameter, bound by the
``gen`` facade at call time: the engine-table entry, the box host table, the quirks-card
lookup, the already-evaluated catalog freshness gate, the engine directory, the outputs
root, the effective host, the config hash, the timestamp and the shared-gaming-host
predicate. That keeps the NSFW mirror honest: ``nsfw/scripts/gen.py`` passes its own
nsfw-local roots, so a plan built from the mirror stays inside ``nsfw/`` (a leaf that
derived OUTPUTS from ``loramake.config`` would send it into the SFW tree).

The function bodies are those of the original router functions; only the names of the
injected collaborators changed. Sampling constants and graph construction live elsewhere
and are untouched.
"""
from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable, Collection
from pathlib import Path

from loramake import model_resources as _mr


def compile_resources(spec: dict,
                      intent: str,
                      requested_box: str | None = None,
                      base: str | None = None,
                      loras: list[str] | None = None,
                      style_loras: list[str] | None = None,
                      pose_ref: str | None = None,
                      index_path=None,
                      *,
                      card_lookup: Callable[[str], dict | None],
                      allowed_boxes: Collection[str],
                      catalog_problems: list[str]) -> _mr.ResourcePlan:
    """Compile resource constraints and eligible boxes for an intent/brief.

    Constraints considered:
      * route-table boxa_only / boxb_only
      * base model_quirks card per-box registration and family
      * LoRA catalog: family compatibility and box availability
      * pose_ref forces BOXB

    ``spec`` is the router's engine-table entry for ``intent`` (``{}`` when unknown);
    ``card_lookup`` resolves a ``--base`` stem to its quirks card; ``allowed_boxes`` is the
    router's box host table; ``catalog_problems`` is the LoRA catalog freshness gate the
    caller already evaluated for ``index_path`` (non-empty means fail closed).
    """
    if catalog_problems:
        return _mr.ResourcePlan(
            family=spec.get("resource_family"),
            eligible_boxes=frozenset(),
            base_trace={},
            lora_traces=[],
            style_lora_traces=[],
            pose_ref=pose_ref,
            problems=catalog_problems,
        )
    requested = requested_box or "auto"
    if requested == "boxb" and spec.get("boxa_only"):
        why = spec.get("boxa_only_why") or ("the Forge deployment lives on BOXA only" if spec.get("forge")
                                          else "the Anima router lines are pinned to BOXA by routing policy")
        return _mr.ResourcePlan(
            family=spec.get("resource_family"),
            eligible_boxes=frozenset({"boxa"}),
            base_trace={},
            lora_traces=[],
            style_lora_traces=[],
            pose_ref=pose_ref,
            problems=[f"'{intent}' is BOXA-only ({why}); drop --box boxb."],
        )
    if requested == "boxa" and spec.get("boxb_only"):
        return _mr.ResourcePlan(
            family=spec.get("resource_family"),
            eligible_boxes=frozenset({"boxb"}),
            base_trace={},
            lora_traces=[],
            style_lora_traces=[],
            pose_ref=pose_ref,
            problems=[f"'{intent}' is BOXB-only; use --box boxb."],
        )

    base_card = None
    pre_problems: list[str] = []
    if base and isinstance(base, str):
        card = card_lookup(base)
        if not card:
            pre_problems.append(f"--base: no quirks card matches {base!r}")
        else:
            base_card = dict(card, stem=base)

    plan = _mr.compile_resources(
        spec,
        base_card,
        loras,
        style_loras,
        pose_ref,
        requested_box or "auto",
        allowed_boxes=allowed_boxes,
        index_path=index_path,
    )

    if pre_problems:
        plan = _mr.ResourcePlan(
            family=plan.family,
            eligible_boxes=plan.eligible_boxes,
            base_trace=plan.base_trace,
            lora_traces=plan.lora_traces,
            style_lora_traces=plan.style_lora_traces,
            pose_ref=plan.pose_ref,
            problems=pre_problems + plan.problems,
        )

    return _mr.ResourcePlan(
        family=plan.family,
        eligible_boxes=plan.eligible_boxes,
        base_trace=plan.base_trace,
        lora_traces=plan.lora_traces,
        style_lora_traces=plan.style_lora_traces,
        pose_ref=plan.pose_ref,
        problems=plan.problems,
    )


def route_gates(box: str, host: str, args: argparse.Namespace, *,
                shared_gaming_host: Callable[[str], bool]) -> list[str]:
    """Pre-run gates the router applies before launching on ``box`` at ``host``.

    ``shared_gaming_host`` is the router's predicate for the box that may be running a game
    with an empty ComfyUI queue (it needs the resolved BOXA endpoint, which the leaf does not own).
    """
    gates = []
    wait = "queue empty"
    if shared_gaming_host(host):
        wait += " and VRAM >= 40% free"
    gates.append(f"--run first waits for {wait} on {box} (--no-wait overrides)")
    if not args.no_free:
        gates.append("after the engine exits the router POSTs /free to release VRAM (--no-free overrides)")
    return gates


def route_post_steps(intent: str, args: argparse.Namespace) -> list[str]:
    """Post-run steps the router performs for ``intent`` under ``args``."""
    if intent == "artistic":
        steps = ["QC defect prefilter"]
        if getattr(args, "sfw", False):
            steps.append("wd14 SFW content gate")
        if getattr(args, "ref", None):
            steps.append("CCIP identity keep vs --ref")
        steps.append("survivors + contact sheet for a by-eye pick (no scalar auto-crown)")
        return steps
    if intent == "sequence":
        base = "CCIP identity HOLD/DRIFT report vs --ref"
        return [base + " with hard filter" if getattr(args, "ccip_keep", False) else base]
    if getattr(args, "pick_best", False):
        return ["QC-score all candidates and keep the single best-of-N"]
    if intent == "promote":
        return ["draft -> finished Anima img2img; output is an engine PNG set"]
    return ["raw engine outputs; add --json-out PATH for a machine-readable result"]


def resource_plan_to_fields(plan: _mr.ResourcePlan, box: str) -> dict:
    """Convert a compiled _mr.ResourcePlan into compact, box-selected route plan fields."""
    def _box_trace(trace: dict) -> dict:
        if trace.get("canonical") is None:
            return {
                "requested": trace["requested"],
                "canonical": None,
                "resolved": trace.get("name") or trace["requested"],
                "strength": trace.get("strength"),
                "available": None,
                "family_ok": trace.get("family_ok"),
                "warning": trace.get("warning", ""),
            }
        resolved = trace["resolved_boxb" if box == "boxb" else "resolved_boxa"]
        available = trace["available_boxb" if box == "boxb" else "available_boxa"]
        return {
            "requested": trace["requested"],
            "canonical": trace["canonical"],
            "resolved": resolved,
            "strength": trace.get("strength"),
            "available": available,
            "family_ok": trace.get("family_ok"),
            "warning": trace.get("warning", ""),
        }

    return {
        "resource_family": plan.family,
        "eligible_boxes": sorted(plan.eligible_boxes),
        "base_trace": plan.base_trace,
        "lora_trace": [_box_trace(t) for t in plan.lora_traces],
        "style_lora_trace": [_box_trace(t) for t in plan.style_lora_traces],
        "pose_ref": plan.pose_ref,
    }


def build_route_plan(intent: str, box: str, box_reason: str, env: dict[str, str],
                     argv: list[str], args: argparse.Namespace,
                     readiness: dict | None = None,
                     resource_plan: _mr.ResourcePlan | None = None,
                     *,
                     spec: dict,
                     engine_dir: Path,
                     outputs_root: Path,
                     host: str,
                     config_hash: str,
                     generated_at: str,
                     schema: str,
                     shared_gaming_host: Callable[[str], bool]) -> dict:
    """Machine-readable route plan: everything a fresh agent needs to fire or explain the route.

    ``spec`` is the router's engine-table entry for ``intent``; ``engine_dir`` is the directory
    holding the engine scripts (``scripts/`` or the NSFW mirror); ``outputs_root`` is that
    tree's output root; ``host`` is the effective endpoint the caller resolved; ``config_hash``,
    ``generated_at`` and ``schema`` are the router's stable digest, timestamp and plan schema id;
    ``shared_gaming_host`` feeds :func:`route_gates`.
    """
    command = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + \
        " ".join(shlex.quote(a) for a in argv)
    plan = {
        "schema": schema,
        "intent": intent,
        "engine": spec["script"],
        "engine_script": str(engine_dir / spec["script"]),
        "family": spec.get("resource_family"),
        "box": box,
        "box_reason": box_reason,
        "host": host,
        "outputs_root": str(outputs_root),
        "pre_run_gates": route_gates(box, host, args, shared_gaming_host=shared_gaming_host),
        "post_run": route_post_steps(intent, args),
        "command": command,
        "argv": argv,
        "env": env,
        "config_hash": config_hash,
        "generated_at": generated_at,
    }
    if resource_plan is not None:
        plan.update(resource_plan_to_fields(resource_plan, box))
    if readiness is not None:
        plan["readiness"] = readiness
    return plan
