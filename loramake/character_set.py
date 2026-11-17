"""Standing-art set planning for one design card: views x expressions, prompt assembly.

The set is the "usable" tier of the character line: the same face (an anchor image
chosen in the identity round) across a few views and the basic six expressions on a
plain background. Main views render on NovelAI with the anchor as a Precise Reference;
expression differentials are face-mask repaints of the neutral bust on the BOXA box.

This module only plans: it assembles prompts from the design card (via
``scripts/character_design.py``'s element tokens) and the expression set, and writes
jobs, commands and an index. Nothing here renders, spends or talks to a box.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import character_design as cd  # noqa: E402

SCHEMA = "loramake.character-set/v1"
EXPRESSIONS_FILE = ROOT / "configs" / "composition" / "expressions.json"
SET_HOME = ROOT / "outputs" / "characters"

NAI_REFERENCE_MODEL = "nai-diffusion-4-5-full"  # Precise Reference is V4.5 only
DEFAULT_REFERENCE_STRENGTH = 1.0
DEFAULT_REFERENCE_FIDELITY = 0.5
FACE_SEGM = "anzhc_face"
FACE_MODE = "replace"
# M1 (2026-09-05): 0.75 with the colour-match pass left a pale rectangle around the face and
# redrew the eyes; 0.7 without colour-match and a wide feather changed only the expression.
FACE_DENOISE = 0.7
FACE_FEATHER_PX = 24
FACE_GROW = 10
FACE_FAMILY = "illustrious"
FACE_ELEMENTS = ("hair", "eyes", "face", "skin_marks")
# M1 (2026-09-05): every arm, including faces a human called a different person, passed the
# library's 0.21; same-face renders scored <= 0.025 and drifted ones >= 0.07. Provisional.
GATE_MAX_DIFF = 0.06


class SetError(ValueError):
    pass


def load_expressions(path: Path = EXPRESSIONS_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("basic") or not data.get("views"):
        raise SetError(f"{path} needs 'basic' expressions and 'views'")
    return data


def expression_list(card: dict, expressions: dict) -> list[dict]:
    """The basic six plus up to N character-specific slots from the card register."""
    items = [dict(e) for e in expressions["basic"]]
    slots = int(expressions.get("character_slots") or 0)
    allowed = list(card["register"].get("allowed_expressions") or [])
    for i, phrase in enumerate(allowed[:slots]):
        items.append({"id": f"register_{i}", "name_cn": f"角色表情 {i + 1}", "tags": [phrase]})
    return items


def anchor_of(card: dict, override: str | None = None) -> str:
    image = override or (card.get("anchor") or {}).get("image")
    if not image:
        raise SetError("no anchor: pass --anchor IMAGE or write card.anchor.image after the identity round")
    if not Path(image).is_file():
        raise SetError(f"anchor image not found: {image}")
    return str(image)


def identity_tokens(card: dict, frame: str, set_name: str, drop: tuple[str, ...] = ("expression",)) -> list[str]:
    """Card tokens for one frame and element set, minus the elements in ``drop``."""
    tokens: list[str] = []
    for key, toks in cd.element_tokens(card, frame, set_name):
        if key in drop:
            continue
        tokens.extend(toks)
    return _dedup(tokens)


def face_tokens(card: dict) -> list[str]:
    """Only the elements a face repaint must keep: hair colour, eyes, face, skin marks."""
    tokens: list[str] = []
    for key, toks in cd.element_tokens(card, "bust", "sig"):
        if key in FACE_ELEMENTS:
            tokens.extend(toks)
    return _dedup(tokens)


def _dedup(tokens) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def nai_job(card: dict, view: dict, expression: dict, outfit_id: str, anchor: str,
            expressions: dict, strength: float, fidelity: float) -> dict:
    """One nai_gen.py job: base = count + view + background, character = identity + expression."""
    count_tag, char_tag = cd.SUBJECTS[card["subject"]][0], cd.SUBJECTS[card["subject"]][1]
    set_name = f"outfit:{outfit_id}"
    body = identity_tokens(card, view["frame"], set_name)
    base = ", ".join([count_tag, "solo", *view["tags"], *expressions.get("background", [])])
    character = ", ".join([char_tag, *body, *expression["tags"]])
    compiled = cd.compile_card(card, "nai", frame=view["frame"], set_name=set_name)
    return {
        "model": NAI_REFERENCE_MODEL,
        "base": base,
        "characters": [{"prompt": character}],
        "negative": compiled["negative"],
        "character_references": [
            {"image": anchor, "type": "character", "strength": strength, "fidelity": fidelity}
        ],
    }


def nai_command(job_path: Path, seed: int, out_dir: Path, artist_card: str | None) -> str:
    argv = ["python3", "scripts/nai_gen.py", "--job", str(job_path), "--seed", str(seed),
            "--out", str(out_dir), "--anlas-budget", "150"]
    if artist_card:
        argv += ["--artist-mix", artist_card]
    return " ".join(shlex.quote(a) for a in argv)


def face_command(card: dict, bust: str, expression: dict, seed: int, loras: list[str],
                 negative: str) -> str:
    prompt = ", ".join([*face_tokens(card), *expression["tags"]])
    argv = ["python3", "scripts/region_fix.py", "--image", bust, "--segm", FACE_SEGM,
            "--mode", FACE_MODE, "--denoise", f"{FACE_DENOISE:g}", "--family", FACE_FAMILY,
            "--no-color-match", "--feather-px", str(FACE_FEATHER_PX), "--grow", str(FACE_GROW),
            "--box", "boxa", "--no-meta", "--character", card["id"], "--seed", str(seed),
            "--prompt", prompt, "--neg", negative]
    for lora in loras:
        argv += ["--lora", lora]
    return " ".join(shlex.quote(a) for a in argv)


def plan_set(card: dict, out_dir: Path, outfit_id: str, anchor: str, seed: int,
             artist_card: str | None, expressions: dict | None = None,
             strength: float = DEFAULT_REFERENCE_STRENGTH,
             fidelity: float = DEFAULT_REFERENCE_FIDELITY) -> dict:
    """Write the NAI main-view jobs and the index; the face differentials wait for the bust."""
    expressions = expressions or load_expressions()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SetError(f"{out_dir} is not empty; sets never mix, pick a new --out")
    outfit = cd.outfit_for(card, outfit_id)  # validates the id against the card
    if outfit.get("sfw") is not True:
        raise SetError(f"outfit {outfit['id']!r} is sfw:false; the set line is SFW-only")
    exprs = expression_list(card, expressions)
    neutral = exprs[0]
    items: list[dict] = []
    commands: list[str] = []
    files: list[tuple[Path, str]] = []
    for view in expressions["views"]:
        job = nai_job(card, view, neutral, outfit["id"], anchor, expressions, strength, fidelity)
        stem = f"{view['id']}__{neutral['id']}"
        job_path = out_dir / "jobs" / f"{stem}.job.json"
        render_dir = out_dir / "renders" / stem
        files.append((job_path, json.dumps(job, ensure_ascii=False, indent=2) + "\n"))
        commands.append(nai_command(job_path, seed, render_dir, artist_card))
        items.append({"kind": "nai", "view": view["id"], "expression": neutral["id"],
                      "frame": view["frame"], "seed": seed, "job": str(job_path),
                      "render_dir": str(render_dir), "prompt": job["characters"][0]["prompt"]})
    for expression in exprs[1:]:
        items.append({"kind": "boxa_face", "view": "bust", "expression": expression["id"],
                      "frame": "bust", "seed": seed, "tags": expression["tags"],
                      "base": None, "command": None})
    index = {
        "schema": SCHEMA, "design": card["id"], "name_cn": card["name_cn"],
        "card_sha256": cd.card_sha256(card), "outfit": outfit["id"], "anchor": anchor,
        "artist_card": artist_card, "reference": {"strength": strength, "fidelity": fidelity},
        "seed": seed, "items": items,
    }
    files.append((out_dir / "set_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"))
    files.append((out_dir / "commands_main.sh",
                  "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n"))
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = commands
    return index


IDENTITY_SCHEMA = "loramake.character-identity/v1"
IDENTITY_MAX = 12


def identity_job(card: dict, outfit_id: str | None, expression_index: int, expressions: dict) -> dict:
    """One neutral bust job for the identity round: card identity (no anchor yet) on the NAI free tier."""
    exprs = list((card.get("register") or {}).get("allowed_expressions") or [])
    if not exprs:
        raise SetError("card register has no allowed_expressions; the identity bust needs one visible expression")
    count_tag, char_tag = cd.SUBJECTS[card["subject"]][0], cd.SUBJECTS[card["subject"]][1]
    set_name = f"outfit:{outfit_id}" if outfit_id else "sig"
    body = identity_tokens(card, "bust", set_name)
    bust = next(v for v in expressions["views"] if v["id"] == "bust")
    expression = exprs[expression_index % len(exprs)]
    base = ", ".join([count_tag, "solo", *bust["tags"], *expressions.get("background", [])])
    character = ", ".join([char_tag, *body, expression])
    compiled = cd.compile_card(card, "nai", frame="bust", set_name=set_name)
    return {"model": NAI_REFERENCE_MODEL, "base": base, "characters": [{"prompt": character}],
            "negative": compiled["negative"]}


LOCAL_ENGINES = {
    # BOXA-first identity rounds (user, 2026-09-05: "try not to use nai first"): the Anima fast
    # line on BOXA, or the Illustrious quality line on the BOXB worker box, each with the artist
    # LoRA the probes picked. Briefs go through gen_job.py with --keep-into so seeds never
    # overwrite each other in the shared output tree.
    "anima": {"tier": "draft", "box": "boxa", "res": "square"},
    "illustrious": {"tier": "best", "box": "boxb", "res": "1024x1024"},
}


def identity_brief(card: dict, engine: str, seed: int, lora_specs: list[str],
                   outfit_id: str | None, expression_index: int, expressions: dict) -> dict:
    """One neutral-bust gen_job brief for a local identity round."""
    if engine not in LOCAL_ENGINES:
        raise SetError(f"engine must be one of {tuple(LOCAL_ENGINES)}; got {engine!r}")
    exprs = list((card.get("register") or {}).get("allowed_expressions") or [])
    if not exprs:
        raise SetError("card register has no allowed_expressions; the identity bust needs one visible expression")
    settings = LOCAL_ENGINES[engine]
    count_tag = cd.SUBJECTS[card["subject"]][0]
    set_name = f"outfit:{outfit_id}" if outfit_id else "sig"
    body = identity_tokens(card, "bust", set_name)
    bust = next(v for v in expressions["views"] if v["id"] == "bust")
    expression = exprs[expression_index % len(exprs)]
    prompt = ", ".join([count_tag, "solo", *body, expression, *bust["tags"], *expressions.get("background", [])])
    compiled = cd.compile_card(card, engine, frame="bust", set_name=set_name)
    brief = {"tier": settings["tier"], "box": settings["box"], "res": settings["res"], "n": 1,
             "seed": seed, "sfw": True, "prompt": prompt}
    if lora_specs:
        brief["lora"] = lora_specs[0]
        if len(lora_specs) > 1:
            brief["loras"] = lora_specs[1:]
    if compiled.get("negative") and engine == "illustrious":
        brief["neg_add"] = compiled["negative"]
    return brief


def plan_identity_local(card: dict, out_dir: Path, seeds: list[int], engine: str,
                        lora_specs: list[str], outfit_id: str | None = None,
                        expression_index: int = 0, expressions: dict | None = None) -> dict:
    """Write a local identity round: one gen_job brief per seed plus commands.sh (renders nothing)."""
    expressions = expressions or load_expressions()
    if not seeds:
        raise SetError("identity round needs at least one seed")
    if len(seeds) > IDENTITY_MAX:
        raise SetError(f"identity round is capped at {IDENTITY_MAX} images per round; got {len(seeds)} seeds")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SetError(f"{out_dir} is not empty; rounds never mix, pick a new --out")
    if engine not in card.get("target_lines", []):
        raise SetError(f"{engine} is not in the card's target_lines {card.get('target_lines')}")
    if outfit_id and cd.outfit_for(card, outfit_id).get("sfw") is not True:
        raise SetError(f"outfit {outfit_id!r} is sfw:false; the identity round is SFW-only")
    items, commands, files = [], [], []
    for seed in seeds:
        brief = identity_brief(card, engine, seed, lora_specs, outfit_id, expression_index, expressions)
        brief_path = out_dir / "briefs" / f"identity_bust__s{seed}.brief.json"
        result = out_dir / "results" / f"identity_bust__s{seed}.json"
        render_dir = out_dir / "renders" / f"s{seed}"
        files.append((brief_path, json.dumps(brief, ensure_ascii=False, indent=2) + "\n"))
        commands.append(" ".join(shlex.quote(a) for a in (
            "python3", "scripts/gen_job.py", str(brief_path), "--json-out", str(result),
            "--keep-into", str(render_dir))))
        items.append({"seed": seed, "brief": str(brief_path), "result": str(result), "render_dir": str(render_dir)})
    index = {"schema": IDENTITY_SCHEMA, "design": card["id"], "name_cn": card["name_cn"],
             "card_sha256": cd.card_sha256(card), "engine": engine, "loras": list(lora_specs),
             "outfit": outfit_id, "expression_index": expression_index, "seeds": list(seeds),
             "prompt": items and json.loads(files[0][1])["prompt"], "items": items}
    files.append((out_dir / "identity_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"))
    files.append((out_dir / "commands.sh", "#!/usr/bin/env bash\nset -uo pipefail\n" + "\n".join(commands) + "\n"))
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = commands
    return index


def plan_identity(card: dict, out_dir: Path, seeds: list[int], artist_card: str | None,
                  outfit_id: str | None = None, expression_index: int = 0,
                  expressions: dict | None = None) -> dict:
    """Write the identity round: one free NAI bust per seed, same prompt, ready for the anchor pick."""
    expressions = expressions or load_expressions()
    if not seeds:
        raise SetError("identity round needs at least one seed")
    if len(seeds) > IDENTITY_MAX:
        raise SetError(f"identity round is capped at {IDENTITY_MAX} images per round; got {len(seeds)} seeds")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SetError(f"{out_dir} is not empty; rounds never mix, pick a new --out")
    if outfit_id:
        outfit = cd.outfit_for(card, outfit_id)
        if outfit.get("sfw") is not True:
            raise SetError(f"outfit {outfit['id']!r} is sfw:false; the identity round is SFW-only")
    job = identity_job(card, outfit_id, expression_index, expressions)
    job_path = out_dir / "jobs" / "identity_bust.job.json"
    render_dir = out_dir / "renders"
    commands = [nai_command(job_path, seed, render_dir, artist_card) for seed in seeds]
    index = {"schema": IDENTITY_SCHEMA, "design": card["id"], "name_cn": card["name_cn"],
             "card_sha256": cd.card_sha256(card), "outfit": outfit_id, "artist_card": artist_card,
             "expression_index": expression_index, "seeds": list(seeds), "job": str(job_path),
             "render_dir": str(render_dir), "prompt": job["characters"][0]["prompt"]}
    files = [(job_path, json.dumps(job, ensure_ascii=False, indent=2) + "\n"),
             (out_dir / "identity_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"),
             (out_dir / "commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")]
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = commands
    return index


MAIN_VIEW_ENGINE = {
    # Main views on the local Illustrious line: the anchor rides along as the IPAdapter face
    # reference (gen.py sequence), the same artist LoRA keeps the style, and the neutral bust
    # for the face differentials comes from the plain quality line.
    "illustrious": {"sequence": {"tier": "sequence", "box": "boxb"}, "bust": {"tier": "best", "box": "boxb"},
                    "res": {"full": "896x1152", "bust": "1024x1024"}},
}


def main_view_brief(card: dict, view: dict, expression: dict, outfit_id: str, anchor: str,
                    engine: str, lora_specs: list[str], expressions: dict) -> dict:
    if engine not in MAIN_VIEW_ENGINE:
        raise SetError(f"local main views support {tuple(MAIN_VIEW_ENGINE)}; got {engine!r}")
    settings = MAIN_VIEW_ENGINE[engine]
    count_tag = cd.SUBJECTS[card["subject"]][0]
    set_name = f"outfit:{outfit_id}"
    body = identity_tokens(card, view["frame"], set_name)
    prompt = ", ".join([count_tag, "solo", *body, *expression["tags"], *view["tags"],
                        *expressions.get("background", [])])
    compiled = cd.compile_card(card, engine, frame=view["frame"], set_name=set_name)
    kind = "bust" if view["frame"] == "bust" else "sequence"
    brief = {**settings[kind], "res": settings["res"][view["frame"]], "n": 1, "sfw": True, "prompt": prompt}
    if kind == "sequence":
        brief["ref"] = anchor
    if lora_specs:
        brief["lora"] = lora_specs[0]
        if len(lora_specs) > 1:
            brief["loras"] = lora_specs[1:]
    if compiled.get("negative"):
        brief["neg_add"] = compiled["negative"]
    return brief


def plan_set_local(card: dict, out_dir: Path, outfit_id: str, anchor: str, seed: int, engine: str,
                   lora_specs: list[str], expressions: dict | None = None) -> dict:
    """Local main views: full and cowboy via IPAdapter on the anchor, the neutral bust plain."""
    expressions = expressions or load_expressions()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SetError(f"{out_dir} is not empty; sets never mix, pick a new --out")
    if engine not in card.get("target_lines", []):
        raise SetError(f"{engine} is not in the card's target_lines {card.get('target_lines')}")
    outfit = cd.outfit_for(card, outfit_id)
    if outfit.get("sfw") is not True:
        raise SetError(f"outfit {outfit['id']!r} is sfw:false; the set line is SFW-only")
    if not Path(anchor).is_file():
        raise SetError(f"anchor image not found: {anchor}")
    exprs = expression_list(card, expressions)
    neutral = exprs[0]
    items, commands, files = [], [], []
    for view in expressions["views"]:
        brief = main_view_brief(card, view, neutral, outfit["id"], anchor, engine, lora_specs, expressions)
        brief["seed"] = seed
        stem = f"{view['id']}__{neutral['id']}"
        brief_path = out_dir / "briefs" / f"{stem}.brief.json"
        result = out_dir / "results" / f"{stem}.json"
        render_dir = out_dir / "renders" / stem
        files.append((brief_path, json.dumps(brief, ensure_ascii=False, indent=2) + "\n"))
        commands.append(" ".join(shlex.quote(a) for a in (
            "python3", "scripts/gen_job.py", str(brief_path), "--json-out", str(result),
            "--keep-into", str(render_dir))))
        items.append({"kind": "local", "engine": engine, "view": view["id"], "expression": neutral["id"],
                      "frame": view["frame"], "seed": seed, "brief": str(brief_path), "result": str(result),
                      "render_dir": str(render_dir), "prompt": brief["prompt"]})
    for expression in exprs[1:]:
        items.append({"kind": "boxa_face", "view": "bust", "expression": expression["id"], "frame": "bust",
                      "seed": seed, "tags": expression["tags"], "base": None, "command": None})
    index = {"schema": SCHEMA, "design": card["id"], "name_cn": card["name_cn"], "card_sha256": cd.card_sha256(card),
             "outfit": outfit["id"], "anchor": anchor, "engine": engine, "loras": list(lora_specs),
             "artist_card": None, "seed": seed, "items": items}
    files.append((out_dir / "set_index.json", json.dumps(index, ensure_ascii=False, indent=2) + "\n"))
    files.append((out_dir / "commands_main.sh", "#!/usr/bin/env bash\nset -uo pipefail\n" + "\n".join(commands) + "\n"))
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index["commands"] = commands
    return index


def load_index(out_dir: Path) -> dict:
    path = out_dir / "set_index.json"
    if not path.is_file():
        raise SetError(f"no set_index.json under {out_dir}; run plan first")
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != SCHEMA:
        raise SetError(f"{path} is not a {SCHEMA} index")
    return index


def save_index(out_dir: Path, index: dict) -> None:
    (out_dir / "set_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plan_faces(card: dict, out_dir: Path, bust: str, loras: list[str], seed: int | None = None) -> dict:
    """Bind the neutral bust render to every face differential and write commands_faces.sh."""
    if not Path(bust).is_file():
        raise SetError(f"bust image not found: {bust}")
    index = load_index(out_dir)
    compiled = cd.compile_card(card, "nai", frame="bust", set_name=f"outfit:{index['outfit']}")
    commands: list[str] = []
    for item in index["items"]:
        if item["kind"] != "boxa_face":
            continue
        item_seed = seed if seed is not None else item["seed"]
        expression = {"id": item["expression"], "tags": item["tags"]}
        item["base"] = bust
        item["seed"] = item_seed
        item["command"] = face_command(card, bust, expression, item_seed, loras, compiled["negative"])
        commands.append(item["command"])
    if not commands:
        raise SetError("the index has no face differentials to bind")
    index["bust"] = bust
    index["loras"] = list(loras)
    save_index(out_dir, index)
    (out_dir / "commands_faces.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
    index["commands"] = commands
    return index


def caption_for(index: dict, item: dict, path: str) -> str:
    key = f"{index['design']}/set/{item['view']}/{item['expression']}"
    sha8 = str(index.get("card_sha256") or "")[:8]
    line = "nai" if item["kind"] == "nai" else "boxa"
    return f"[{key}] {index.get('name_cn', '')} · {line} · seed {item['seed']} · card {sha8} — {Path(path).name}"


def parse_ccip(output: str) -> dict[str, dict]:
    """Rows of consistency_check.py: ``0.1234  HOLD   name.png``."""
    rows: dict[str, dict] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("HOLD", "DRIFT"):
            try:
                rows[parts[2]] = {"ccip": float(parts[0]), "hold": parts[1] == "HOLD"}
            except ValueError:
                continue
    return rows


def gate_summary(rows: dict[str, dict], min_hold_ratio: float = 0.85) -> dict:
    held = sum(1 for r in rows.values() if r["hold"])
    ratio = held / len(rows) if rows else 0.0
    return {"held": held, "total": len(rows), "hold_ratio": round(ratio, 3),
            "pass": bool(rows) and ratio >= min_hold_ratio, "min_hold_ratio": min_hold_ratio}
