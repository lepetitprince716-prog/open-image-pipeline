#!/usr/bin/env python3
"""De-AI lint for concept-card decks and raw prompt pairs.

Checks every card (or a raw ``--base`` / ``--character`` pair) against the machine
rules in ``configs/deai/rules.json``: the five commitments (camera, light source,
moment, artist voice, single accent), the quality-tag ban, slop attractor tokens,
the mode-composition guard, the NAI saturation guard, the signature/text guard,
mechanism words, the small-figure scale guard, the youth guard and the artist-tag
lottery. Per-rule ``"match"`` in rules.json selects ``substring`` (default),
``word``, or ``prefix``.
The ``qwen21`` line adds a natural-language syntax rule (``qwen21_syntax``) and the
anti-text closing-sentence check (``qwen21_text_closer``); artist voice and the artist
lottery do not apply there (no @artist / style LoRA mechanism on that engine).
A line with a ``line_profiles`` entry in rules.json runs only the rules that entry lists
(``moss`` runs the youth guard and slop tokens until its recipe work settles the rest).
Findings are reported per card id as ``{rule, severity, message,
evidence}``; any ``error`` finding exits non-zero and ``--strict`` promotes
warnings to errors. Rule data lives in the JSON file — this script only does the
thin matching on top of it. Method authority: ``docs/DEAI_CREATIVE_METHOD.md``.

Usage:
    python3 scripts/deai_lint.py lint DECK.json [--rules RULES.json] [--strict] [--json]
    python3 scripts/deai_lint.py prompt --line nai --base "…" --character "…" \
        [--negative "…"] [--strict] [--json]
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from config import ROOT
import concept_cards
from concept_cards import DeckError

SCHEMA = "loramake-deai-rules/v1"
DEFAULT_RULES = ROOT / "configs" / "deai" / "rules.json"
LINES = ("nai", "anima", "illustrious", "qwen21", "moss")

# A negatively weighted section (-2::x, y ::) suppresses its tokens; it must not
# count as asking for the thing. Strip such spans before matching positive text.
# DOTALL: a wrapped -N::...:: span must still strip.
NEGATIVE_WEIGHT_RE = re.compile(r"-\d+(?:\.\d+)?::.*?::", re.DOTALL)
# Digit-leading artist handles (@448Ani) stay; Danbooru count tags (@1girl/@2girls) do not.
AT_ARTIST_RE = re.compile(r"@(?!\d+(?:girl|boy|others?|girls|boys)\b)[a-z0-9]")


class LintError(ValueError):
    """A user-facing lint setup problem (bad rules file, bad deck, bad args)."""


def load_rules(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LintError(f"cannot read rules: {exc}") from None
    except json.JSONDecodeError as exc:
        raise LintError(f"rules are not valid JSON: {exc}") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise LintError(f"rules schema must be {SCHEMA!r}")
    return data


def _strip_negative_weights(text: str) -> str:
    return NEGATIVE_WEIGHT_RE.sub(" ", text)


@functools.lru_cache(maxsize=None)
def _bounded_re(token: str, mode: str) -> re.Pattern:
    # Text is already lowercased; edges count only letters/digits, so tokens
    # with punctuation or digits ("close-up", "bird's-eye", "8k") behave right.
    # "word" bounds both edges; "prefix" bounds only the leading edge so safety
    # lists over-match derivatives (kid -> kids, child -> childhood/lolicon-style).
    trailing = r"(?![a-z0-9])" if mode == "word" else ""
    return re.compile(rf"(?<![a-z0-9]){re.escape(token)}{trailing}")


def _mode(rule: dict) -> str:
    mode = rule.get("match", "substring")
    if mode not in ("substring", "word", "prefix"):
        raise LintError(f"unknown match mode {mode!r}")
    return mode


def _hits(text: str, tokens: Iterable[str], mode: str = "substring") -> list[str]:
    if isinstance(tokens, str):
        raise LintError("token list must be a list, not a string")
    tokens = [t.lower() for t in tokens]
    if mode in ("word", "prefix"):
        hits = [t for t in tokens if _bounded_re(t, mode).search(text)]
        if mode == "prefix":
            # Report the most specific hit: "children", not "child, children".
            hits = [t for t in hits
                    if not any(o != t and o.startswith(t) for o in hits)]
        return hits
    return [t for t in tokens if t in text]


def _applies(rule: dict, line: str) -> bool:
    lines = rule.get("lines")
    return not lines or line in lines


def _for_line(rules: dict, line: str) -> dict:
    """The rule set a line's profile allows; a line without a profile keeps every rule.

    Profile ids are top-level rule names, or ``commitments.<name>`` for the five commitments.
    """
    profile = (rules.get("line_profiles") or {}).get(line)
    if profile is None:
        return rules
    only = profile.get("only") if isinstance(profile, dict) else None
    if not isinstance(only, list):
        raise LintError(f"line_profiles.{line}.only must be a list of rule ids")
    allowed = set(only)
    kept = {key: value for key, value in rules.items() if key in allowed}
    kept["commitments"] = {key: value for key, value in (rules.get("commitments") or {}).items()
                           if f"commitments.{key}" in allowed}
    return kept


def _finding(rule: str, severity: str, message: str, evidence: str = "") -> dict:
    return {"rule": rule, "severity": severity, "message": message, "evidence": evidence}


def _view(card: dict, index: int, line: str) -> dict:
    positive = _strip_negative_weights(
        f"{card.get('artist_block') or ''} {card.get('base') or ''} "
        f"{card.get('character') or ''}".lower()
    )
    return {
        "id": str(card.get("id") or f"card#{index}"),
        "line": line,
        "card": card,
        "positive": positive,
        "negative": str(card.get("negative_add") or "").lower(),
    }


def lint_subject(view: dict, rules: dict, prompt_mode: bool = False) -> list[dict]:
    """Run every applicable rule on one card view; return its findings."""
    pos, neg, line, card = view["positive"], view["negative"], view["line"], view["card"]
    rules = _for_line(rules, line)
    out: list[dict] = []

    def add(rule_id: str, rule: dict, severity: str | None = None, evidence: str = "") -> None:
        out.append(_finding(rule_id, severity or rule.get("severity", "warning"),
                            rule.get("message", ""), evidence))

    commit = rules.get("commitments", {})
    for key in ("camera", "light_source"):
        rule = commit.get(key) or {}
        mode = _mode(rule)
        if rule and _applies(rule, line) \
                and not _hits(pos, rule.get("tokens", []), mode):
            add(key, rule)

    rule = commit.get("moment") or {}
    if rule and not prompt_mode and not str(card.get("moment") or "").strip():
        add("moment", rule)

    rule = commit.get("artist_voice") or {}
    if rule and _applies(rule, line):
        if line == "nai":
            ok = "artist:" in pos
        else:
            ok = bool(AT_ARTIST_RE.search(pos)) or bool(
                _hits(pos, ("lora", "loras"), "word")
            ) or any(
                card.get(f) for f in ("style_lora", "lora", "style_loras")
            )
        if not ok:
            severity = rule.get("severity", "error")
            if prompt_mode and line != "nai":
                # A style LoRA rides at engine level and is invisible to prompt text.
                severity = "warning"
            add("artist_voice", rule, severity=severity)

    rule = commit.get("single_accent") or {}
    if rule and not prompt_mode:
        palette = card.get("palette")
        ok = (
            isinstance(palette, dict)
            and bool(str(palette.get("base") or "").strip())
            and bool(str(palette.get("accent") or "").strip())
        )
        if not ok:
            add("single_accent", rule)

    qt = rules.get("quality_tags") or {}
    qt_mode = _mode(qt)
    if qt and _applies(qt, line):
        severity = (qt.get("severity") or {}).get(line)
        hits = _hits(pos, qt.get("tokens", []), qt_mode)
        if severity and hits:
            add("quality_tags", qt, severity=severity, evidence=", ".join(hits))

    mc = rules.get("mode_composition") or {}
    mc_mode = _mode(mc)
    gaze = _hits(pos, mc.get("gaze_tokens", []), mc_mode)
    frame = _hits(pos, mc.get("frame_tokens", []), mc_mode)
    if mc and _applies(mc, line) and gaze and frame \
            and not _hits(pos, mc.get("escape_tokens", []), mc_mode):
        add("mode_composition", mc,
            evidence=f"gaze={gaze[0]!r} frame={frame[0]!r}, no escape token")

    sg = rules.get("saturation_guard") or {}
    sg_mode = _mode(sg)
    if sg and _applies(sg, line) and not _hits(pos, sg.get("positive_tokens", []), sg_mode) \
            and not _hits(neg, sg.get("negative_tokens", []), sg_mode):
        add("saturation_guard", sg)

    tg = rules.get("text_guard") or {}
    tg_mode = _mode(tg)
    if tg and _applies(tg, line) and not _hits(pos, tg.get("positive_tokens", []), tg_mode) \
            and not _hits(neg, tg.get("negative_tokens", []), tg_mode):
        add("text_guard", tg)

    for key in ("slop_tokens", "mechanism_words", "scale_guard", "youth_guard"):
        rule = rules.get(key) or {}
        mode = _mode(rule)
        if not _applies(rule, line):
            continue
        hits = _hits(pos, rule.get("tokens", []), mode)
        if hits:
            add(key, rule, evidence=", ".join(hits))

    pl = rules.get("pixel_lottery") or {}
    if pl and _applies(pl, line):
        max_tags = int(pl.get("max_artist_tags", 3))
        n_artist = pos.count("artist:") + len(AT_ARTIST_RE.findall(pos))
        if n_artist > max_tags:
            add("pixel_lottery", pl, evidence=f"{n_artist} artist tags > {max_tags}")

    # Qwen-Image-2.1: natural-language English only. Each pattern is one syntax family the
    # model ignores or misreads; the @artist form is a warning because it is merely unproven.
    qs = rules.get("qwen21_syntax") or {}
    if qs and _applies(qs, line):
        patterns = qs.get("patterns") or {}
        if not isinstance(patterns, dict):
            raise LintError("qwen21_syntax.patterns must be an object")
        for name, pattern in patterns.items():
            match = re.search(pattern, pos)
            if match:
                add("qwen21_syntax", qs, evidence=f"{name}: {match.group(0)!r}")
        if AT_ARTIST_RE.search(pos):
            out.append(_finding("qwen21_syntax", qs.get("at_artist_severity", "warning"),
                                qs.get("at_artist_message", ""), "@artist"))

    qc = rules.get("qwen21_text_closer") or {}
    if qc and _applies(qc, line) and not _hits(pos, qc.get("tokens", []), _mode(qc)):
        add("qwen21_text_closer", qc)

    return out


def _promote(findings: list[dict]) -> list[dict]:
    return [{**f, "severity": "error"} if f["severity"] == "warning" else f
            for f in findings]


def lint_deck(deck: dict, rules: dict, strict: bool = False) -> dict[str, list[dict]]:
    """Lint every card in a loaded deck; report only cards with findings."""
    line = deck.get("line", "nai")
    report: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for index, card in enumerate(deck.get("cards") or []):
        if not isinstance(card, dict):
            label = f"card#{index}"
            report[label] = [_finding("deck", "error", "card is not an object")]
            continue
        view = _view(card, index, line)
        label = view["id"]
        if label in seen or label in report:
            label = f"{label}#{index}"
        seen.add(label)
        findings = lint_subject(view, rules)
        if strict:
            findings = _promote(findings)
        if findings:
            report[label] = findings
    return report


def lint_prompt(line: str, base: str, character: str, negative: str,
                rules: dict, strict: bool = False) -> list[dict]:
    """Lint a raw prompt pair. Card-field commitments (moment, single accent) have
    no card to check and are skipped; prompt-level rules still apply."""
    card = {"id": "prompt", "base": base, "character": character}
    view = _view(card, 0, line)
    view["negative"] = negative.lower()
    findings = lint_subject(view, rules, prompt_mode=True)
    return _promote(findings) if strict else findings


def has_errors(report: dict[str, list[dict]] | list[dict]) -> bool:
    rows = report.values() if isinstance(report, dict) else [report]
    return any(f["severity"] == "error" for findings in rows for f in findings)


def _print_report(report: dict[str, list[dict]]) -> None:
    findings = errors = 0
    for label, rows in report.items():
        print(f"{label}:")
        for f in rows:
            findings += 1
            errors += f["severity"] == "error"
            suffix = f" (evidence: {f['evidence']})" if f.get("evidence") else ""
            print(f"  [{f['severity']}] {f['rule']}: {f['message']}{suffix}")
    print(f"cards_with_findings={len(report)} findings={findings} errors={errors}")


def _cmd_lint(args: argparse.Namespace) -> int:
    rules = load_rules(Path(args.rules))
    try:
        deck = concept_cards.load_deck(Path(args.deck))
    except DeckError as exc:
        raise LintError(str(exc)) from None
    report = lint_deck(deck, rules, strict=args.strict)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 1 if has_errors(report) else 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    rules = load_rules(Path(args.rules))
    findings = lint_prompt(args.line, args.base, args.character or "",
                           args.negative or "", rules, strict=args.strict)
    if args.json:
        print(json.dumps({"prompt": findings}, ensure_ascii=False, indent=2))
    else:
        _print_report({"prompt": findings} if findings else {})
    return 1 if has_errors(findings) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="De-AI lint for decks and prompts")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lint", help="lint every card in a concept deck")
    p.add_argument("deck")
    p.add_argument("--rules", default=str(DEFAULT_RULES))
    p.add_argument("--strict", action="store_true",
                   help="promote warnings to errors")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_lint)

    p = sub.add_parser("prompt", help="lint one raw base/character prompt pair")
    p.add_argument("--line", required=True, choices=LINES)
    p.add_argument("--base", required=True)
    p.add_argument("--character", default="")
    p.add_argument("--negative", default="")
    p.add_argument("--rules", default=str(DEFAULT_RULES))
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_prompt)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (LintError, OSError, json.JSONDecodeError) as exc:
        print(f"deai_lint.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
