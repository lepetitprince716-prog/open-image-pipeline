"""Revision contracts: which dimension a change touches, how small it may be, what stays frozen.

The model is the one already written down in the AI drawing knowledge base
(``13-出图系统Harness/image_revision_axes.md``): ten revision dimensions, a seven-rung
minimal-intervention ladder, and a freeze contract expressed as ``changed_dimensions`` /
``frozen_dimensions`` / ``minimal_delta``.

This module owns the contract only.  It classifies brief fields into dimensions, checks a
proposal against the freeze declaration, and applies an approved delta to a parent brief.  It
never talks to ComfyUI, never reads an image, and never decides *what* the user wants changed
— that judgement belongs to the chat agent, which submits a proposal here so the server can
check it deterministically instead of trusting prose.

The point of the whole exercise is the freeze half.  A proposal that says it only changes the
pose must not quietly move the checkpoint, so every field in ``minimal_delta`` is mapped back
to the dimension that owns it and rejected when that dimension was declared frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# --- dimensions ------------------------------------------------------------------------

SUBJECT = "subject_identity"
COMPOSITION = "composition"
POSE = "pose"
STYLE = "style"
LIGHTING = "lighting"
DETAIL = "detail"
MODEL_LORA = "model_lora"
SAMPLER = "sampler"
WORKFLOW = "workflow"
CONTENT = "content"

DIMENSIONS = (
    SUBJECT, COMPOSITION, POSE, STYLE, LIGHTING, DETAIL,
    MODEL_LORA, SAMPLER, WORKFLOW, CONTENT,
)

# Every brief field belongs to exactly one dimension, so a freeze check is a lookup rather
# than a judgement call.  Two assignments are deliberate and worth stating:
#   * ``style_lora`` is a style resource, so swapping it is a STYLE change.  Its weight rides
#     in ``loras`` and is therefore MODEL_LORA — changing what a style looks like and changing
#     how hard it is applied are different intents.
#   * ``res``/``aspect`` sit under COMPOSITION rather than SAMPLER: a user asking for a taller
#     frame is reframing, not retuning.
FIELD_DIMENSION: dict[str, str] = {
    # subject / identity
    "prompt": SUBJECT,
    "character": SUBJECT,
    "outfit": SUBJECT,
    "action": SUBJECT,
    "ref": SUBJECT,
    # framing
    "composition": COMPOSITION,
    "camera": COMPOSITION,
    "aspect": COMPOSITION,
    "res": COMPOSITION,
    "scene": COMPOSITION,
    # pose / performance
    "pose": POSE,
    "pose_ref": POSE,
    "expression": POSE,
    # style / medium
    "artist": STYLE,
    "medium": STYLE,
    "style_mode": STYLE,
    "style_lora": STYLE,
    "direction": STYLE,
    # light / colour
    "lighting": LIGHTING,
    "color_mood": LIGHTING,
    # detail quality
    "detail": DETAIL,
    "neg_add": DETAIL,
    "detailer": DETAIL,
    # resources
    "base": MODEL_LORA,
    "lora": MODEL_LORA,
    "loras": MODEL_LORA,
    # sampling
    "steps": SAMPLER,
    "cfg": SAMPLER,
    "sampler": SAMPLER,
    "scheduler": SAMPLER,
    "seed": SAMPLER,
    "denoise": SAMPLER,
    "pag": SAMPLER,
    "no_freeu": SAMPLER,
    # topology / routing
    "tier": WORKFLOW,
    "box": WORKFLOW,
    "provider": WORKFLOW,
    # content boundary
    "sfw": CONTENT,
    "nsfw": CONTENT,
}

# --- ladder ----------------------------------------------------------------------------

# The seven rungs, cheapest first.  A proposal declares the rung it lands on; the fields it
# touches must not reach past it.  Rung 7 rewrites the graph and is refused unless the caller
# passes an explicit authorisation, because no brief field can express it.
RUNG_FIELDS: dict[int, frozenset[str]] = {
    1: frozenset({"prompt", "character", "outfit", "action", "composition", "camera",
                  "pose", "expression", "artist", "medium", "lighting", "color_mood",
                  "direction", "scene"}),
    2: frozenset({"neg_add"}),
    3: frozenset({"detail", "detailer", "pose_ref", "denoise"}),
    4: frozenset({"steps", "cfg", "sampler", "scheduler", "seed", "pag", "no_freeu"}),
    5: frozenset({"lora", "loras", "style_lora", "style_mode", "ref"}),
    6: frozenset({"base"}),
    7: frozenset({"tier", "box", "provider", "aspect", "res"}),
}

MAX_RUNG = 7
TOPOLOGY_RUNG = 7

# The content boundary is not a revision dimension and deliberately sits on no rung.  The
# knowledge base puts content authority with the user, so "make it warmer" must never be able
# to walk an image across the SFW line as a side effect of tuning something else.  Changing it
# is a new generation the user asks for, not a delta a proposal may carry.
UNREVISABLE_FIELDS = frozenset({"sfw", "nsfw"})

# Evidence grading, from the knowledge base.  A proposal built on nothing but generic priors
# has to say so out loud rather than presenting itself with the same confidence as one backed
# by an approved local record.
CONFIDENCE_LEVELS = ("strong", "moderate", "weak", "heuristic")
HEURISTIC_DISCLAIMER = "此建议暂无本机实证支撑"


def dimension_of(field_name: str) -> str | None:
    """The dimension that owns ``field_name``, or None when the field is not classified."""
    return FIELD_DIMENSION.get(field_name)


def rung_of(field_name: str) -> int | None:
    """The cheapest ladder rung that can express a change to ``field_name``."""
    for rung in sorted(RUNG_FIELDS):
        if field_name in RUNG_FIELDS[rung]:
            return rung
    return None


def dimensions_touched(delta: dict[str, Any]) -> set[str]:
    """Dimensions actually written by ``delta``, regardless of what it claims."""
    return {dim for dim in (dimension_of(key) for key in delta) if dim is not None}


@dataclass
class RevisionProposal:
    """One bounded change to one image.

    ``scope`` is "whole" or "region"; a region-scoped proposal carries a mask only when the
    delta lands on a rung that actually needs to repaint pixels (see ``needs_mask``).
    """

    changed_dimensions: list[str]
    frozen_dimensions: list[str]
    minimal_delta: dict[str, Any]
    rung: int
    scope: str = "whole"
    needs_mask: bool = False
    confidence: str = "heuristic"
    confidence_note: str = ""
    revision_id: str = ""
    user_feedback: str = ""
    risk_notes: list[str] = field(default_factory=list)


def validate(
    proposal: RevisionProposal,
    *,
    known_fields: Iterable[str] | None = None,
    allow_topology: bool = False,
) -> list[str]:
    """Problems with ``proposal``.  Empty list means it may be applied.

    ``known_fields`` restricts the delta to the fields the target tier actually accepts, so a
    proposal cannot smuggle in a key that the brief validator would later reject with a much
    less useful message.
    """
    problems: list[str] = []

    # This runs on untrusted input and its contract is to return problems, so a malformed
    # shape has to become a message rather than an exception. Anything that is not a list of
    # strings is reported and then treated as empty: a non-list is not iterable, a nested list
    # is unhashable in the set intersection below, and a mixed str/int list cannot be sorted --
    # all three used to surface as a 500 from a validator whose whole job is saying no politely.
    def _dimension_list(value: object, field: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            problems.append(f"{field} must be a list of dimension names")
            return []
        names = [item for item in value if isinstance(item, str)]
        if len(names) != len(value):
            problems.append(f"{field} must contain only dimension names")
        return names

    changed = _dimension_list(proposal.changed_dimensions, "changed_dimensions")
    frozen = _dimension_list(proposal.frozen_dimensions, "frozen_dimensions")

    if not changed:
        problems.append("changed_dimensions must name at least one dimension")
    for dim in changed + frozen:
        if dim not in DIMENSIONS:
            problems.append(f"unknown dimension: {dim}")

    overlap = sorted(set(changed) & set(frozen))
    if overlap:
        problems.append(f"dimensions cannot be both changed and frozen: {', '.join(overlap)}")

    # A revision that claims to move more than two dimensions at once is not a revision, it is
    # a regeneration.  The knowledge base states the rule as one primary plus at most one
    # supporting dimension.
    if len(changed) > 2:
        problems.append(
            f"a single revision may change at most 2 dimensions, got {len(changed)}: "
            f"{', '.join(sorted(changed))}"
        )

    if not isinstance(proposal.minimal_delta, dict) or not proposal.minimal_delta:
        problems.append("minimal_delta must be a non-empty object")
        return problems

    if proposal.scope not in ("whole", "region"):
        problems.append('scope must be "whole" or "region"')

    if not isinstance(proposal.rung, int) or isinstance(proposal.rung, bool):
        problems.append("rung must be an integer")
    elif not 1 <= proposal.rung <= MAX_RUNG:
        problems.append(f"rung must be in 1..{MAX_RUNG}")

    if proposal.confidence not in CONFIDENCE_LEVELS:
        problems.append(f"confidence must be one of {', '.join(CONFIDENCE_LEVELS)}")
    elif proposal.confidence == "heuristic" and HEURISTIC_DISCLAIMER not in proposal.confidence_note:
        problems.append(
            f"a heuristic proposal must state {HEURISTIC_DISCLAIMER!r} in confidence_note"
        )

    allowed = set(known_fields) if known_fields is not None else None

    for key in sorted(proposal.minimal_delta):
        if key in UNREVISABLE_FIELDS:
            problems.append(
                f"minimal_delta.{key} moves the content boundary, which a revision may never "
                f"do; the user decides content on a fresh generation"
            )
            continue
        dim = dimension_of(key)
        if dim is None:
            problems.append(f"minimal_delta.{key} is not a classified brief field")
            continue
        if dim in frozen:
            problems.append(
                f"minimal_delta.{key} writes the frozen dimension {dim}"
            )
        elif dim not in changed:
            problems.append(
                f"minimal_delta.{key} touches {dim}, which is not in changed_dimensions"
            )
        if allowed is not None and key not in allowed:
            problems.append(f"minimal_delta.{key} is not accepted by the target tier")

        field_rung = rung_of(key)
        if field_rung is None:
            problems.append(f"minimal_delta.{key} is not placed on the ladder")
        elif isinstance(proposal.rung, int) and not isinstance(proposal.rung, bool):
            if field_rung > proposal.rung:
                problems.append(
                    f"minimal_delta.{key} needs rung {field_rung} but the proposal declares "
                    f"rung {proposal.rung}"
                )

    if proposal.rung == TOPOLOGY_RUNG and not allow_topology:
        problems.append(
            "rung 7 rewrites the workflow and needs explicit authorisation; "
            "solve it on a lower rung or pass allow_topology"
        )

    if proposal.needs_mask and proposal.scope != "region":
        problems.append("needs_mask requires scope=region")

    return problems


def apply_delta(parent_brief: dict[str, Any], proposal: RevisionProposal) -> dict[str, Any]:
    """A new brief: ``parent_brief`` with the proposal's delta written over it.

    The parent is not mutated.  A delta value of None drops the field, which is how a
    proposal clears an inherited knob (for example dropping a style LoRA) without needing a
    separate verb.  Callers must run :func:`validate` first — this function trusts its input.
    """
    child = dict(parent_brief)
    for key, value in proposal.minimal_delta.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    return child


def frozen_fields(frozen_dimensions: Iterable[str]) -> set[str]:
    """Every brief field a freeze declaration protects.  Useful for diffing two briefs."""
    frozen = set(frozen_dimensions)
    return {name for name, dim in FIELD_DIMENSION.items() if dim in frozen}


def verify_frozen(
    parent_brief: dict[str, Any],
    child_brief: dict[str, Any],
    frozen_dimensions: Iterable[str],
) -> list[str]:
    """Fields that moved between two briefs despite belonging to a frozen dimension.

    This is the belt to :func:`validate`'s braces: validate checks what a proposal *says* it
    will touch, this checks what actually changed once the brief was built.  Anything that
    reaches ComfyUI should have passed both.
    """
    violations: list[str] = []
    for name in sorted(frozen_fields(frozen_dimensions)):
        if parent_brief.get(name) != child_brief.get(name):
            violations.append(
                f"{name} ({FIELD_DIMENSION[name]}) changed while its dimension was frozen"
            )
    return violations
