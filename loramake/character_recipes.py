"""Recipes behind the character atelier: prompts, sheet crops and the BOXA command lines.

Everything here was validated by hand on the tide keeper (2026-09-06): the GPT reference
sheet as identity truth, img2img restyle on the Illustrious line with a lineart ControlNet
and an IPAdapter face lock, and masked face repaints for expressions. Numbers are the ones
that held identity; change them only with a new side-by-side.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import OUTPUTS

ROOT = OUTPUTS.parent
SCRIPTS = ROOT / "scripts"
VENV_PY = ROOT / ".venv" / "bin" / "python"
BRIDGE_PY = ROOT / "comfyui-boxa" / "tools" / "visual_features" / ".venv" / "bin" / "python"
SHEET_SIZE = "1536x1024"
CANVAS = (832, 1216)
PAD = 40

# Art-book pages: mutually distant Illustrious artist LoRAs (CSD max-min pick, 2026-09-06).
BOOK_STYLES: tuple[dict, ...] = (
    {"id": "artist_hanabi_style", "label": "花火厚涂", "lora": "Illustrious\\style\\hanabi-09.safetensors"},
    {"id": "artist_vixon_s_illustrious_styles_ye_jji", "label": "灰调成熟", "lora": "Illustrious\\style\\ye_jjiILL.safetensors"},
    {"id": "artist_xianzi_xianxia_mature_woman_style", "label": "仙侠熟女", "lora": "Illustrious\\style\\6TWQ4QER4N7MX1NBGZ2ZX3A0H0.safetensors"},
    {"id": "artist_fantasy_forge", "label": "幻想锻炉", "lora": "Illustrious\\style\\0NZ9KMEEMSGGXKV1KM7G1SVNE0.safetensors"},
    {"id": "artist_nekomega", "label": "猫眼淡彩", "lora": "Illustrious\\anime\\IL_nekomega style-000050.safetensors"},
    {"id": "artist_cat_citron_styles", "label": "柠檬异色", "lora": "cat_citron_styles.safetensors"},
)
NEG_COMMON = ("text, watermark, signature, border, frame, letters, green skin, whipped cream, cream, "
              "white liquid, dessert, food, earrings, jewelry, blush, big eyes, child, teen")
NEG_BODY = "short legs, chibi, wide hips, large breasts"
NEG_HAIR = "ponytail, twintails, loose long hair, hair ornament, tassel"
EXPRESSIONS: tuple[tuple[str, str, str], ...] = (
    ("neutral", "澄まし", "level gaze, closed mouth, still, looking at viewer"),
    ("afar", "望远", "eyes narrowed against the wind, squinting, looking into the distance, side glance, closed mouth"),
    ("smile", "微笑", "gentle warm smile, closed-mouth smile, soft relaxed eyes, slight head tilt"),
    ("frown", "皱眉", "furrowed brow, closed mouth, sideways glance, displeased"),
    ("surprise", "惊讶", "surprised, eyebrows raised high, wide open eyes, mouth slightly open, startled"),
    ("tired", "疲惫", "exhausted, half-closed eyes, drooping eyelids, eyes downcast, sigh"),
)


@dataclass(frozen=True)
class Command:
    label: str
    argv: list[str]
    expect_dir: Path  # where the engine drops its PNGs
    expect_glob: str  # newest match after the run is the product


def _strs(values) -> list[str]:
    return [v for v in (values or []) if isinstance(v, str) and v.strip()]


def _hair(card: dict) -> list[str]:
    hair = card.get("hair") or {}
    return [hair.get(k) for k in ("color", "length", "shape", "detail") if hair.get(k)]


def _eyes(card: dict) -> list[str]:
    eyes = card.get("eyes") or {}
    return [eyes.get(k) for k in ("color", "shape") if eyes.get(k)]


def outfit_tags(card: dict) -> list[str]:
    outfits = card.get("outfits") or []
    return _strs((outfits[0] if outfits else {}).get("tags")) if outfits else []


def identity_prompt(card: dict) -> str:
    parts = ["1girl", "solo", "mature adult woman", *_hair(card), *_eyes(card), *_strs(card.get("skin_marks")),
             *_strs(card.get("face")), "calm expression"]
    return ", ".join(dict.fromkeys(p for p in parts if p))


def sheet_prompt(card: dict) -> str:
    """The GPT reference-sheet brief: identity + one outfit + layout, and no text at all."""
    palette = card.get("palette") or {}
    base = ", ".join(_strs(palette.get("base"))) or "muted neutrals"
    accent = palette.get("accent") or "one warm accent"
    register = (card.get("register") or {}).get("description") or "calm and composed"
    signature = card.get("signature_element") or "one signature accessory"
    outfit = ", ".join(outfit_tags(card)) or "a simple, plausible everyday outfit"
    accessories = "; ".join(f"{a.get('phrase')} ({a.get('placement')})" for a in card.get("accessories") or []
                            if isinstance(a, dict) and a.get("phrase"))
    avoid = ", ".join(_strs(card.get("neg_guards")) + _strs(card.get("forbidden")))
    return "\n".join([
        f"Landscape 3:2 official anime character reference sheet for an original adult woman: {card.get('intent') or card.get('concept') or ''}.",
        "Role of this image: the base identity sheet every later illustration is checked against, so keep it clean, readable and consistent across panels.",
        "",
        f"Style: polished Japanese anime illustration with refined clean linework, cel shading with soft gradients, a muted palette of {base} with a single {accent} accent, pale neutral studio background, no scenery.",
        "",
        "Identity (identical in every panel):",
        "- Adult woman in her late twenties, not teen-coded, not cute-styled; " + register + ".",
        "- Silhouette: " + ", ".join(_strs(card.get("silhouette"))) + ".",
        "- Face: " + ", ".join(_strs(card.get("face")) + _strs(card.get("skin_marks"))) + ".",
        "- Eyes: " + ", ".join(_eyes(card)) + ".",
        "- Hair: " + ", ".join(_hair(card)) + ".",
        f"- Signature element: {signature}; it appears in every full-body view.",
        "",
        "Clothing (one outfit, shown consistently): " + outfit + "."
        + (f" Accessories: {accessories}." if accessories else ""),
        "Physical logic: fabrics drape by their material, accessories hang with real weight, nothing merges into the body.",
        "",
        "Layout:",
        "- Left two thirds: front view, side view and back view, full body, same height, feet visible, standing relaxed.",
        "- Right third, top: six small expression portraits, restrained and in character: "
        + "; ".join(_strs((card.get("register") or {}).get("allowed_expressions"))[:4]) + "; quiet surprise; tired with eyes lowered.",
        "- Right third, bottom: close-up part breakdown of the hair, the signature element and one clothing detail; and a small palette row.",
        "- No text, labels, captions, logos or watermarks anywhere in the image.",
        "",
        "Avoid: childlike face, big moe eyes, wide grin, tears, pout, fantasy armor, cluttered accessories, busy background, "
        + (avoid + ", " if avoid else "") + "any written characters.",
    ])


def edit_prompt(card: dict, instruction: str) -> str:
    keep = ", ".join(_hair(card) + _eyes(card) + _strs(card.get("skin_marks")) + [card.get("signature_element") or ""]
                     + outfit_tags(card))
    return "\n".join([
        f"Surgical edit of the attached character reference sheet. {instruction.strip()}",
        "",
        "Keep EVERYTHING ELSE identical to the attached image: the overall layout and every panel position, the three "
        f"full-body poses, the expressions, the part breakdowns, the palette swatches, the pale studio background, and: {keep}.",
        "Do not add text, labels or logos. Avoid: childlike or teen face, big moe eyes, makeup, changed composition.",
    ])


# ---------------------------------------------------------------- sheet crops

def _detect(image: Path) -> dict:
    """Person / head boxes from the imgutils bridge; {} when the bridge is unavailable."""
    if not BRIDGE_PY.is_file():
        return {}
    try:
        proc = subprocess.run([str(BRIDGE_PY), str(SCRIPTS / "imgutils_bridge.py"), "detect", str(image)],
                              capture_output=True, text=True, timeout=300)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or not results:
        return {}
    return next(iter(results.values())) or {}


def _boxes(found: dict, key: str, min_conf: float = 0.4) -> list[tuple[int, int, int, int]]:
    out = []
    for item in found.get(key) or []:
        bbox = item.get("bbox") if isinstance(item, dict) else None
        if bbox and len(bbox) == 4 and float(item.get("confidence", 0)) >= min_conf:
            out.append(tuple(int(v) for v in bbox))
    return out


def crop_views(sheet: Path, out_dir: Path) -> dict[str, Path]:
    """Cut the front/side/back figures and the neutral bust out of a reference sheet.

    Persons are taken from the detector (tallest three, left to right); the bust is the head
    box of the front figure grown to a portrait. Falls back to the fixed sheet layout when
    the detector finds nothing.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(sheet).convert("RGB")
    width, height = image.size
    found = _detect(sheet)
    persons = sorted(_boxes(found, "persons"), key=lambda b: b[3] - b[1], reverse=True)[:3]
    persons = sorted(persons, key=lambda b: b[0])
    if len(persons) < 2:
        persons = [(int(width * 0.03), int(height * 0.11), int(width * 0.26), int(height * 0.98)),
                   (int(width * 0.25), int(height * 0.11), int(width * 0.42), int(height * 0.98)),
                   (int(width * 0.40), int(height * 0.11), int(width * 0.59), int(height * 0.98))]
    names = ("front", "side", "back")
    crops: dict[str, Path] = {}
    for name, (x0, y0, x1, y1) in zip(names, persons, strict=False):
        pad = int((y1 - y0) * 0.03)
        box = (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))
        path = out_dir / f"{name}_full.png"
        image.crop(box).save(path)
        crops[name] = path
    front = persons[0]
    heads = [h for h in _boxes(found, "heads") if front[0] <= (h[0] + h[2]) / 2 <= front[2]]
    heads.sort(key=lambda h: h[1])
    if heads:
        hx0, hy0, hx1, hy1 = heads[0]
        hw = hx1 - hx0
        box = (max(0, hx0 - hw // 2), max(0, hy0 - hw // 3), min(width, hx1 + hw // 2), min(height, hy1 + hw))
    else:
        fw = front[2] - front[0]
        box = (front[0], front[1], front[2], front[1] + int(fw * 1.3))
    bust = image.crop(box)
    bust_path = out_dir / "bust.png"
    bust.save(bust_path)
    bust.resize((bust.width * 4, bust.height * 4), Image.LANCZOS).save(out_dir / "bust_x4.png")
    crops["bust"] = bust_path
    crops["bust_x4"] = out_dir / "bust_x4.png"
    return crops


def pad_to_canvas(src: Path, dst: Path, canvas: tuple[int, int] = CANVAS) -> Path:
    """Centre a figure crop on a white canvas of the render aspect (img2img crops otherwise)."""
    from PIL import Image

    figure = Image.open(src).convert("RGB")
    tw, th = canvas
    scale = min((th - 2 * PAD) / figure.height, (tw - 2 * PAD) / figure.width)
    figure = figure.resize((max(1, int(figure.width * scale)), max(1, int(figure.height * scale))), Image.LANCZOS)
    board = Image.new("RGB", canvas, (255, 255, 255))
    board.paste(figure, ((tw - figure.width) // 2, (th - figure.height) // 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    board.save(dst)
    return dst


# ---------------------------------------------------------------- BOXA commands

def _boxa_out_dir() -> Path:
    from .config import OUTPUT_CHARACTER
    return OUTPUTS / OUTPUT_CHARACTER / "by_model" / "claude" / "boxa_quality"


def _common(prompt: str, face: Path, lora: str, weight: float, neg: str, denoise: float,
            control_strength: float, control_end: float, seed: int, label: str, res: str = "896x1152") -> list[str]:
    return [str(VENV_PY), str(SCRIPTS / "boxa_quality_gen.py"), "--family", "illustrious", "--prompt", prompt,
            "--no-wildcard", "--res", res, "--n", "1", "--seed", str(seed), "--max",
            "--lora", f"{lora}:{weight}", "--neg-add", neg,
            "--control", "lineart", "--control-strength", str(control_strength), "--control-end", str(control_end),
            "--ip-image", str(face), "--ip", "face", "--denoise", str(denoise), "--label", label]


def book_command(card: dict, style: dict, bust_x4: Path, seed: int) -> Command:
    """One art-book page: restyle the GPT bust (img2img 0.6, lineart 0.4, LoRA 1.0)."""
    prompt = identity_prompt(card) + ", " + ", ".join(outfit_tags(card)[:3]) + \
        ", portrait, bust, looking at viewer, simple background, white background"
    label = f"book_{style['id'][-12:]}_{seed}"
    argv = _common(prompt, bust_x4, style["lora"], 1.0, NEG_COMMON, 0.6, 0.4, 0.6, seed, label)
    argv += ["--init-image", str(bust_x4), "--control-image", str(bust_x4)]
    return Command(label, argv, _boxa_out_dir(), f"*_{label}_*.png")


def view_command(card: dict, view: str, padded: Path, bust_x4: Path, lora: str, seed: int) -> Command:
    """A full-body view (front / side) in the chosen style: img2img 0.5, lineart 0.55, hair locked."""
    hair = ", ".join(_hair(card))
    pose = {"front": "front view, looking at viewer", "side": "from side, profile view, looking ahead, hands in pockets"}[view]
    prompt = ("1girl, solo, tall slim adult woman, straight shoulders, long legs, small bust, "
              + hair + ", " + ", ".join(_eyes(card)) + ", " + ", ".join(_strs(card.get("skin_marks")))
              + ", calm expression, " + ", ".join(outfit_tags(card)) + f", {card.get('signature_element') or ''}"
              + f", full body, standing, {pose}, simple background, white background")
    label = f"view_{view}_{seed}"
    argv = _common(prompt, bust_x4, lora, 1.0, f"{NEG_HAIR}, {NEG_COMMON}, {NEG_BODY}", 0.5, 0.55, 0.7, seed, label,
                   res=f"{CANVAS[0]}x{CANVAS[1]}")
    argv += ["--init-image", str(padded), "--control-image", str(padded)]
    return Command(label, argv, _boxa_out_dir(), f"*_{label}_*.png")


def face_command(card: dict, image: Path, lora: str, seed: int, character: str,
                 expression: str | None = None) -> Command:
    """Masked face repaint: refine 0.5 restores freckles / maturity; replace 0.7 swaps the expression."""
    mode, denoise = ("replace", "0.7") if expression else ("refine", "0.5")
    prompt = ("mature adult woman face, late twenties, " + ", ".join(_strs(card.get("face")) + _strs(card.get("skin_marks"))
              + _eyes(card) + ["fine eyebrows"] + _hair(card)))
    if expression:
        prompt += ", " + expression
    argv = [str(VENV_PY), str(SCRIPTS / "region_fix.py"), "--image", str(image), "--segm", "anzhc_face",
            "--mode", mode, "--denoise", denoise, "--phase", "run", "--box", "boxa", "--n", "1", "--seed", str(seed),
            "--lora", f"{lora}:0.9", "--no-color-match", "--feather-px", "24", "--grow", "10", "--no-meta",
            "--character", character, "--prompt", prompt,
            "--neg", "text, watermark, child, teen, big eyes, blush, makeup, jewelry, wide grin, tears, pout, teeth"]
    out_dir = OUTPUTS / character / "by_model" / "claude" / "region_fix"
    return Command(f"face_{expression or 'refine'}_{seed}", argv, out_dir, f"rf_{image.stem}_{mode}_d{denoise}_*_{seed}.png")


def run_command(command: Command, timeout_s: float = 900.0) -> Path:
    """Run one engine command and return the PNG it produced (newest match)."""
    before = {p: p.stat().st_mtime for p in command.expect_dir.glob(command.expect_glob)} if command.expect_dir.is_dir() else {}
    proc = subprocess.run(command.argv, capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT),
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-3:])
        raise RuntimeError(f"{command.label}: engine exited {proc.returncode}: {tail}")
    fresh = [p for p in command.expect_dir.glob(command.expect_glob)
             if p.stat().st_mtime > before.get(p, -1) and not p.name.endswith("_fx.png")]
    if not fresh:
        raise RuntimeError(f"{command.label}: engine finished without a new PNG")
    return max(fresh, key=lambda p: p.stat().st_mtime)


def free_vram() -> bool:
    from .comfy import free_vram as _free
    try:
        return bool(_free())
    except Exception:
        return False


def matte(src: Path, dst: Path) -> bool:
    """isnetis subject matting through the imgutils bridge (RGBA PNG); False when the model is absent."""
    if not BRIDGE_PY.is_file():
        return False
    try:
        proc = subprocess.run([str(BRIDGE_PY), str(SCRIPTS / "imgutils_bridge.py"), "segment", str(src), str(dst)],
                              capture_output=True, text=True, timeout=300)
        return bool(json.loads(proc.stdout.strip().splitlines()[-1]).get("ok"))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return False


if __name__ == "__main__":  # quick manual check: print the sheet prompt of a card
    card = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(sheet_prompt(card))
