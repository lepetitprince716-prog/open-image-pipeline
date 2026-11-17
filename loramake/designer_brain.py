"""Designer brain: one sentence in, a design card (and her first line) out.

The console cannot hold an API key (the user is on a subscription), so the brain runs the
local Claude Code binary headlessly (``claude -p``, the transport the Agent SDK wraps) and
asks for strict JSON. When the binary is missing, slow or returns garbage, a template card
is built from the sentence instead, so the page never blocks on the model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_BRAIN_BIN = Path.home() / ".local" / "bin" / "claude-shadowrocket"
BRAIN_TIMEOUT_S = 90.0
ID_RE = re.compile(r"[^a-z0-9_]+")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

CARD_INSTRUCTIONS = """You are a professional anime/game character designer. Turn the user's one-line brief into a
design card as STRICT JSON (no prose, no code fences). All visual fields are Danbooru-style
English tags; name_cn, concept, impression_keywords, register.description, outfits[].name_cn
and voice_line are Chinese. Rules: an adult woman; one signature element only; one accent
colour; silhouette-first; no style words (no artist names, no rendering terms); no narrative in
tags; the outfit needs at least two material words; no tears / pout / wide grin in expressions.

Schema:
{"id": "snake_case_ascii", "name_cn": "...", "intent": "one line", "concept": "one line",
 "impression_keywords": ["a","b","c"],
 "silhouette": ["tag","tag","tag"], "face": ["tag","tag"],
 "hair": {"color": "... hair", "length": "... hair", "shape": "...", "detail": "..."},
 "eyes": {"color": "... eyes", "shape": "... eyes"}, "skin_marks": ["..."],
 "signature_element": "...", "palette": {"base": ["colour","colour"], "accent": "colour"},
 "register": {"description": "...", "allowed_expressions": ["...","...","...","..."], "excluded": ["wide grin","tears","pout"]},
 "outfits": [{"id": "snake", "name_cn": "...", "tags": ["6 tags with >=2 materials"], "sfw": true,
              "coverage": "full", "motion_allow": ["static","seated","dynamic"], "neg_add": "", "status": "unvalidated"}],
 "accessories": [{"phrase": "...", "scene_class": "task", "placement": "..."}],
 "neg_guards": ["..."], "forbidden": ["wings","animal ears"],
 "voice_line": "她的一句自我介绍（中文，一句话）"}

Brief:
"""

VOICE_FALLBACK = "……你来了。灯还亮着，进来吧。"


def slugify(text: str, fallback: str | None = None) -> str:
    """ASCII id from free text; non-ASCII briefs fall back to a short content hash."""
    slug = ID_RE.sub("_", text.strip().lower()).strip("_")
    if slug:
        return slug[:40]
    return fallback or "c_" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:6]


def brain_bin() -> Path:
    return Path(os.environ.get("LORAMAKE_BRAIN_BIN", "").strip() or DEFAULT_BRAIN_BIN)


def _extract_json(text: str) -> dict | None:
    match = JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def ask(prompt: str, *, timeout_s: float = BRAIN_TIMEOUT_S) -> str | None:
    """One headless turn; the answer text or None when the brain is unavailable."""
    binary = brain_bin()
    if not binary.is_file():
        return None
    argv = [str(binary), "-p", prompt, "--output-format", "json"]
    model = os.environ.get("LORAMAKE_BRAIN_MODEL", "").strip()
    if model:
        argv += ["--model", model]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout.strip() or None
    result = envelope.get("result") if isinstance(envelope, dict) else None
    return result if isinstance(result, str) and result.strip() else None


def template_card(sentence: str, card_id: str | None = None) -> dict:
    """A lint-shaped card built from the sentence alone (the no-brain fallback)."""
    sentence = sentence.strip() or "an original adult woman"
    cid = card_id or slugify(sentence)
    return {
        "id": cid,
        "name_cn": sentence[:12],
        "intent": sentence,
        "concept": sentence[:80],
        "impression_keywords": ["沉静", "利落", "暖光"],
        "subject": "female",
        "target_lines": ["illustrious", "nai"],
        "silhouette": ["tall slim figure", "straight shoulders", "long coat outline"],
        "face": ["narrow chin", "small mouth"],
        "hair": {"color": "dark brown hair", "length": "long hair", "shape": "hair bun", "detail": "single hair stick"},
        "eyes": {"color": "amber eyes", "shape": "sharp eyes"},
        "skin_marks": [],
        "signature_element": "brass pocket watch on a chain",
        "palette": {"base": ["charcoal", "cream"], "accent": "amber"},
        "register": {
            "description": "calm and unhurried",
            "allowed_expressions": [
                "level gaze, closed mouth, looking at viewer",
                "narrowed eyes, closed mouth, looking afar",
                "light smile, closed mouth, looking at viewer",
                "furrowed brow, closed mouth, sideways glance",
            ],
            "excluded": ["wide grin", "tears", "pout"],
        },
        "outfits": [{
            "id": "default", "name_cn": "默认服装",
            "tags": ["long wool coat", "cotton collared shirt", "wool trousers", "leather boots", "leather belt", "matte fabric"],
            "sfw": True, "coverage": "full", "motion_allow": ["static", "seated", "dynamic"],
            "neg_add": "", "status": "unvalidated",
        }],
        "accessories": [],
        "neg_guards": ["glowing eyes"],
        "forbidden": ["wings", "animal ears"],
        "notes": ["template card: the designer brain was unavailable; refine by talking to her"],
        "status": "draft",
        "render_ledger": {"elements": {}, "rows": []},
    }


def compose_card(sentence: str, *, schema: str) -> tuple[dict, str, str]:
    """Return (card, voice_line, source) where source is ``brain`` or ``template``."""
    answer = ask(CARD_INSTRUCTIONS + sentence.strip())
    body = _extract_json(answer) if answer else None
    if body is None:
        card = template_card(sentence)
        card["schema"] = schema
        return card, VOICE_FALLBACK, "template"
    voice = str(body.pop("voice_line", "") or VOICE_FALLBACK)
    base = template_card(sentence, card_id=slugify(str(body.get("id") or sentence)))
    for key, value in body.items():
        if key in base and value not in (None, "", [], {}):
            base[key] = value
    base["schema"] = schema
    base["notes"] = [f"composed by the designer brain from: {sentence.strip()[:120]}"]
    return base, voice, "brain"


def reply(card: dict, voice: str, history: list[dict], text: str) -> dict:
    """Her answer to the user plus an optional design proposal.

    Returns {"say": str, "proposal": {"summary": str, "edit": str} | None}. The edit is an
    English image-edit instruction ("change only ..."); absent when the turn is pure talk.
    """
    prompt = (
        "You are two voices at once for an original anime character and her designer.\n"
        f"Character card (JSON): {json.dumps(card, ensure_ascii=False)[:4000]}\n"
        f"Her established voice line: {voice}\n"
        "Recent turns: " + json.dumps(history[-6:], ensure_ascii=False) + "\n"
        f"User says: {text}\n"
        "Answer as STRICT JSON: {\"say\": \"她的回答（中文，一到两句，符合人设）\", "
        "\"proposal\": null | {\"summary\": \"设计师提议（中文一句）\", "
        "\"edit\": \"English image-edit instruction: change ONLY <one visual thing>; keep everything else identical\"}}. "
        "Propose a visual change only when the user's words imply one; otherwise proposal is null."
    )
    answer = ask(prompt, timeout_s=60.0)
    body = _extract_json(answer) if answer else None
    if body is None:
        return {"say": voice, "proposal": None}
    say = str(body.get("say") or voice)
    proposal = body.get("proposal")
    if isinstance(proposal, dict) and str(proposal.get("edit") or "").strip():
        proposal = {"summary": str(proposal.get("summary") or ""), "edit": str(proposal["edit"]).strip()}
    else:
        proposal = None
    return {"say": say, "proposal": proposal}
