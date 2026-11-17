"""Composition vocabulary, coverage sampler and prompt compiler for the content explorer.

Plan section 14 (C-1 / C-2): a card is one value per composition axis (shot, angle,
placement, subjects, ratio) plus one per content dimension (subject, setting, action,
time, mood). ``sample_deck`` draws cards so that no two cards, including cards from
earlier decks, share more than ``max_overlap`` dimensions; ``compile_prompt`` turns a
card into the tag string of one line.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

SCHEMA = "loramake.composition/v1"
AXES = ("shot", "angle", "placement", "subjects", "ratio")
CONTENT = ("subject", "setting", "action", "time", "mood")
DIMS = AXES + CONTENT
DEFAULT_MAX_OVERLAP = 3
DEFAULT_TRIES = 20000
SAMPLED_SUBJECTS = ("solo", "duo", "group")  # a female character is always the subject


class CompositionError(ValueError):
    pass


def load_vocabulary(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise CompositionError(f"vocabulary schema must be {SCHEMA}")
    axes = data.get("axes")
    content = data.get("content")
    if not isinstance(axes, dict) or not isinstance(content, dict):
        raise CompositionError("vocabulary needs axes and content")
    for axis in AXES:
        if not isinstance(axes.get(axis), dict) or not axes[axis]:
            raise CompositionError(f"axis {axis} missing or empty")
    for dim in CONTENT:
        if not isinstance(content.get(dim), list) or not content[dim]:
            raise CompositionError(f"content dimension {dim} missing or empty")
    return data


def values(vocab: dict, dim: str) -> list[str]:
    return list(vocab["axes"][dim]) if dim in AXES else list(vocab["content"][dim])


def overlap(a: dict, b: dict) -> int:
    return sum(1 for dim in DIMS if a.get(dim) is not None and a.get(dim) == b.get(dim))


def sample_deck(vocab: dict, *, count: int, seed: int, history: list[dict] | tuple = (),
                max_overlap: int = DEFAULT_MAX_OVERLAP, tries: int = DEFAULT_TRIES) -> list[dict]:
    """Cards that overlap every earlier card (history and this deck) in at most max_overlap dims."""
    rng = random.Random(seed)
    existing = [dict(card) for card in history]
    cards: list[dict] = []
    attempts = 0
    while len(cards) < count and attempts < tries:
        attempts += 1
        pick = {dim: rng.choice(values(vocab, dim)) for dim in DIMS}
        pick["subjects"] = rng.choice(SAMPLED_SUBJECTS)
        if all(overlap(pick, other) <= max_overlap for other in existing + cards):
            cards.append(pick)
    if len(cards) < count:
        raise CompositionError(f"only {len(cards)} of {count} cards fit within overlap {max_overlap} after {tries} tries")
    return cards


def coverage(cards: list[dict]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {dim: {} for dim in DIMS}
    for card in cards:
        for dim in DIMS:
            value = card.get(dim)
            if value is not None:
                table[dim][value] = table[dim].get(value, 0) + 1
    return table


def size(vocab: dict, card: dict) -> tuple[int, int]:
    ratio = vocab["axes"]["ratio"][card["ratio"]]
    return int(ratio["width"]), int(ratio["height"])


def compile_prompt(vocab: dict, card: dict, line: str = "nai") -> str:
    """Tag string for one line: subjects, content, then the composition phrases."""
    axes = vocab["axes"]
    parts = [axes["subjects"][card["subjects"]][line]]
    if card["subjects"] != "none":
        parts.extend([card["subject"], card["action"]])
    parts.extend([card["setting"], card["time"], f"{card['mood']} atmosphere"])
    for axis in ("shot", "angle", "placement"):
        phrase = axes[axis][card[axis]].get(line, "")
        if phrase:
            parts.append(phrase)
    return ", ".join(dedup_tokens(part for part in parts if part))


def build_deck(vocab: dict, *, deck_id: str, line: str, count: int, seed: int, artist_card: str,
               history: list[dict] | tuple = (), max_overlap: int = DEFAULT_MAX_OVERLAP) -> dict:
    picks = sample_deck(vocab, count=count, seed=seed, history=history, max_overlap=max_overlap)
    cards = []
    for index, pick in enumerate(picks, start=1):
        width, height = size(vocab, pick)
        card_id = f"c{index:02d}_{pick['shot']}_{pick['angle']}_{pick['subjects']}"
        cards.append({
            "id": card_id, **pick, "width": width, "height": height, "seed": seed * 100 + index,
            "prompt": compile_prompt(vocab, pick, line),
            "caption": f"[{card_id}] {pick['shot']} / {pick['angle']} / {pick['placement']} / {pick['subjects']} / {pick['ratio']} · "
                       f"{pick['subject']} · {pick['setting']} · {pick['action']} · {pick['time']} · {pick['mood']}",
        })
    return {"schema": "loramake.content-deck/v1", "deck": deck_id, "line": line, "artist_card": artist_card,
            "seed": seed, "max_overlap": max_overlap, "cards": cards}


SERIES_SCHEMA = "loramake.composition-series/v1"


def load_series(path: Path) -> dict:
    """Series presets keyed by id (see configs/composition/series.json)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SERIES_SCHEMA:
        raise CompositionError(f"series file schema must be {SERIES_SCHEMA!r}")
    if not data.get("series"):
        raise CompositionError(f"{path} defines no series")
    return data["series"]


def build_series_deck(vocab: dict, spec: dict, *, deck_id: str, line: str, seed: int,
                      artist_card: str) -> dict:
    """One subject walked along a single content axis, everything else held.

    Unlike build_deck this samples nothing: the preset authors both the held dims and the
    per-step light_source / moment, because those are the de-AI commitments a series exists
    to make explicit (docs/DEAI_CREATIVE_METHOD.md §1). The light source and the accent join
    the prompt because they are renderable; the moment stays on the card and in the caption,
    since a beat like "she has not turned the page yet" is prose a tag line cannot carry — it
    is there for the person judging the series, not for the sampler.
    """
    walk = spec.get("walk")
    if walk not in CONTENT:
        raise CompositionError(f"series walk axis must be one of {CONTENT}, got {walk!r}")
    steps = spec.get("steps") or []
    if not steps:
        raise CompositionError("series preset has no steps")
    hold = dict(spec.get("hold") or {})
    if walk in hold:
        raise CompositionError(f"series holds {walk!r} and also walks it")
    # A hold key outside DIMS is a typo, not an extra dimension. Left unchecked it would
    # ride onto every card while the dimension it meant to hold stayed unheld.
    unknown = sorted(set(hold) - set(DIMS))
    if unknown:
        raise CompositionError(f"series holds unknown dimension(s): {', '.join(unknown)}")
    # A held value gets the same check as a walked one. Only the shipped presets are
    # test-validated, and a typo in a held value rides into every prompt in the series
    # rather than just one card.
    for dim, value in hold.items():
        # values() discriminates on AXES, the same constant DIMS is built from. Branching
        # on the vocabulary's own keys instead would route a content dimension to the axes
        # pool the moment an extra axes key shadowed its name.
        if value not in values(vocab, dim):
            raise CompositionError(f"series holds {dim}={value!r}, which is not in the vocabulary")
    missing = [dim for dim in DIMS if dim != walk and dim not in hold]
    if missing:
        raise CompositionError(f"series preset does not hold {', '.join(missing)}")
    accent = spec.get("accent")
    if not accent:
        raise CompositionError("series preset needs one palette accent for the whole series")

    cards = []
    for index, step in enumerate(steps, start=1):
        for field in ("value", "light_source", "moment"):
            if not str(step.get(field) or "").strip():
                raise CompositionError(f"series step {index} is missing {field}")
        # Only the shipped presets are test-validated against the vocabulary. A hand-authored
        # value that is merely misspelled would otherwise reach the prompt as a tag nothing
        # in the vocabulary backs.
        if step["value"] not in vocab["content"][walk]:
            raise CompositionError(
                f"series step {index}: {walk}={step['value']!r} is not in the vocabulary")
        pick = {**hold, walk: step["value"]}
        width, height = size(vocab, pick)
        card_id = f"s{index:02d}_{_slug(step['value'])}"
        prompt = ", ".join(dedup_tokens([compile_prompt(vocab, pick, line),
                                         step["light_source"], accent]))
        cards.append({
            "id": card_id, **pick, "width": width, "height": height, "seed": seed * 100 + index,
            "light_source": step["light_source"], "moment": step["moment"],
            "palette": {"accent": accent}, "artist_block": artist_card,
            "prompt": prompt,
            "caption": f"[{card_id}] {walk}={step['value']} · {pick['shot']} / {pick['angle']} · "
                       f"{pick['subject']} · {step['moment']}",
        })
    return {"schema": "loramake.content-deck/v1", "deck": deck_id, "line": line,
            "artist_card": artist_card, "seed": seed, "series": {"walk": walk, "accent": accent},
            "cards": cards}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


WILD_DIMS = ("style", "character")


def banned_styles(vocab: dict) -> list[str]:
    """Media the user rejected outright: never sampled again, whatever the deck asks for."""
    groups = vocab.get("style_groups") or {}
    return list(groups.get("excluded_by_user") or [])


def verdict_totals(verdict_files: list[dict]) -> dict[str, dict[str, int]]:
    """love / yes / no counts per style over every verdict file."""
    totals: dict[str, dict[str, int]] = {}
    for data in verdict_files:
        for card in (data.get("cards") or {}).values():
            style = card.get("style")
            verdict = card.get("verdict")
            if not style or verdict not in ("love", "yes", "no"):
                continue
            totals.setdefault(style, {"love": 0, "yes": 0, "no": 0})[verdict] += 1
    return totals


def styles_to_ban(totals: dict[str, dict[str, int]]) -> list[str]:
    """A medium is banned once it has been rejected and never earned a love or a yes."""
    return sorted(style for style, t in totals.items() if t["no"] >= 1 and t["love"] == 0 and t["yes"] == 0)
CHARM_DIMS = ("expression", "gesture", "motif", "concept")


def weighted_choice(rng: random.Random, options: list[str], weights: dict | None) -> str:
    """rng.choice over options, or a weighted draw when the vocabulary gives weights for them."""
    if not weights:
        return rng.choice(options)
    table = [max(0.0, float(weights.get(option, 0))) for option in options]
    if sum(table) <= 0:
        return rng.choice(options)
    return rng.choices(options, weights=table, k=1)[0]
COMPANION_CHANCE = 0.4
WILD_SUBJECTS = ("solo", "duo", "group")


def sample_wild_deck(vocab: dict, *, count: int, seed: int, history: list[dict] | tuple = (),
                     max_overlap: int = DEFAULT_MAX_OVERLAP, tries: int = DEFAULT_TRIES,
                     styles: list[str] | None = None, per_style: int = 1) -> list[dict]:
    """Cards that each carry a far style around a female character.

    ``styles`` restricts the deck to those media (default: every style in the vocabulary)
    and ``per_style`` repeats each medium that many times with different characters and
    compositions. The subject is always one of the character archetypes (sometimes with a
    companion), never a scene without her. Composition and content dimensions keep the
    coverage rule of ``sample_deck``.
    """
    known = vocab.get("styles") or {}
    banned = set(banned_styles(vocab))
    chosen = [style for style in (list(styles) if styles else list(known)) if style not in banned]
    unknown = [style for style in chosen if style not in known]
    characters = vocab.get("characters") or []
    if unknown:
        raise CompositionError(f"unknown styles: {unknown}")
    if not chosen or not characters:
        raise CompositionError("vocabulary has no styles or characters for a wild deck")
    if per_style < 1:
        raise CompositionError("per_style must be at least 1")
    if count > len(chosen) * per_style:
        raise CompositionError(f"wild deck of {count} needs at most {len(chosen) * per_style} cards "
                               f"({len(chosen)} styles x {per_style})")
    rng = random.Random(seed)
    order = []
    for _ in range(per_style):
        block = chosen[:]
        rng.shuffle(block)
        order.extend(block)
    existing = [dict(card) for card in history]
    cards: list[dict] = []
    attempts = 0
    while len(cards) < count and attempts < tries:
        attempts += 1
        weights = vocab.get("wild_weights") or {}
        pick = {dim: weighted_choice(rng, values(vocab, dim), weights.get(dim)) for dim in DIMS}
        pick["style"] = order[len(cards)]
        pick["character"] = rng.choice(characters)
        pick["subject"] = pick["character"]
        pick["subjects"] = weighted_choice(rng, list(WILD_SUBJECTS), weights.get("subjects"))
        for dim in CHARM_DIMS:
            options = vocab["content"].get(dim) or []
            pick[dim] = weighted_choice(rng, options, weights.get(dim)) if options else None
        pick["companion"] = rng.choice(vocab.get("companions") or []) if rng.random() < COMPANION_CHANCE else None
        if all(overlap(pick, other) <= max_overlap for other in existing + cards):
            cards.append(pick)
    if len(cards) < count:
        raise CompositionError(f"only {len(cards)} of {count} wild cards fit within overlap {max_overlap}")
    return cards


def compile_wild_prompt(vocab: dict, card: dict, line: str = "nai") -> str:
    """Style phrase first, then the character, her action, companion, place, time, mood, composition."""
    style = vocab["styles"][card["style"]]
    axes = vocab["axes"]
    companion = card.get("companion")
    if line == "caption":
        shot = axes["shot"][card["shot"]].get("nai") or "medium shot"
        angle = axes["angle"][card["angle"]].get("nai", "")
        with_companion = f" with a {companion}" if companion else ""
        charm = ", ".join(card.get(dim) for dim in CHARM_DIMS if card.get(dim))
        charm = f", {charm}" if charm else ""
        return (f"{style['caption']} of a {card['character']} {card['action']}{with_companion} at {card['setting']}{charm}, "
                f"{card['time']}, {card['mood']} atmosphere, {shot}" + (f", {angle}" if angle else ""))
    parts = [style[line if line in style else "nai"], axes["subjects"][card["subjects"]][line], card["character"]]
    parts.extend(card.get(dim) for dim in ("expression", "gesture") if card.get(dim))
    parts.append(card["action"])
    if companion:
        parts.append(companion)
    parts.append(card["setting"])
    if card.get("concept"):
        parts.append(card["concept"])
    if card.get("motif"):
        parts.append(card["motif"])
    parts.extend([card["time"], f"{card['mood']} atmosphere"])
    for axis in ("shot", "angle", "placement"):
        phrase = axes[axis][card[axis]].get(line, "")
        if phrase:
            parts.append(phrase)
    return ", ".join(dedup_tokens(part for part in parts if part))


def dedup_tokens(parts) -> list[str]:
    """Drop repeated comma-separated tokens (case-insensitive), keeping first occurrences in order."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for token in str(part).split(","):
            token = token.strip()
            if token and token.lower() not in seen:
                seen.add(token.lower())
                out.append(token)
    return out


def build_wild_deck(vocab: dict, *, deck_id: str, line: str, count: int, seed: int,
                    history: list[dict] | tuple = (), max_overlap: int = DEFAULT_MAX_OVERLAP,
                    styles: list[str] | None = None, per_style: int = 1,
                    artists: list[str] | None = None, artist_share: float = 0.5) -> dict:
    """``artists`` are NovelAI artist cards assigned round-robin to a share of the cards, so a
    round compares the same media with and without an artist string."""
    picks = sample_wild_deck(vocab, count=count, seed=seed, history=history, max_overlap=max_overlap,
                             styles=styles, per_style=per_style)
    cards = []
    artist_slots = 0
    for index, pick in enumerate(picks, start=1):
        width, height = size(vocab, pick)
        card_id = f"w{index:02d}_{pick['style']}"
        # Even spread: card N takes an artist when the running share crosses a whole number,
        # so floor(count * share) cards get one and the rest stay bare (pr-gate #257 fix).
        if artists and math.floor(index * artist_share) > math.floor((index - 1) * artist_share):
            pick["artist_card"] = artists[artist_slots % len(artists)]
            artist_slots += 1
            card_id += "_" + pick["artist_card"].replace("_solo", "")
        else:
            pick["artist_card"] = None
        companion = f" + {pick['companion']}" if pick.get("companion") else ""
        cards.append({
            "id": card_id, **pick, "width": width, "height": height, "seed": seed * 100 + index,
            "prompt": compile_wild_prompt(vocab, pick, line),
            "caption_nl": compile_wild_prompt(vocab, pick, "caption"),
            "caption": f"[{card_id}] {pick['style']}{' @' + pick['artist_card'] if pick.get('artist_card') else ''} · "
                       f"{pick['character']}{companion} · {pick.get('concept') or ''} · {pick.get('expression') or ''} · "
                       f"{pick.get('gesture') or ''} · {pick['action']} · {pick['setting']} · "
                       f"{pick['shot']} / {pick['angle']} / {pick['subjects']} / {pick['ratio']}",
        })
    return {"schema": "loramake.content-deck/v1", "deck": deck_id, "line": line, "artist_card": None, "wild": True,
            "seed": seed, "max_overlap": max_overlap, "styles": list(styles) if styles else None,
            "per_style": per_style, "cards": cards}
